//! [`RunLoop`]'s `Starting` phase: spawn the child, pump its stdout/stderr,
//! race the health probe against an early exit or an incoming command, and
//! hand off to [`RunLoop::run_running_loop`] once it passes.

use std::{
    os::unix::process::ExitStatusExt,
    time::{Duration, Instant},
};

use tracing::{error, info, warn};

use crate::{
    db::Database,
    supervise::{
        RunLoop, STARTING_SIGTERM_GRACE, StartingOutcome, Step, drain,
        handle::{StartFailure, StartFailureKind, StartOutcome},
        health::{HealthConfig, HealthOutcome, wait_healthy},
        logs::{spawn_pump_stderr, spawn_pump_stdout},
        render_argv,
        state::{DisableReason, Event as StateEvent, ServiceState, transition},
    },
    system::ManagedChild,
};

/// If a child dies with SIGKILL within this window from spawn, we treat it
/// as an OOM kill and bump the rolling safety factor for the next attempt.
const OOM_KILL_WINDOW: Duration = Duration::from_secs(30);

/// 31-bit mask for `run_id`. `run_id` is derived from wall-clock millis; we
/// clip to a positive `i64` so it round-trips through SQLite's `INTEGER`
/// without sign surprises.
const RUN_ID_MASK: i64 = 0x7FFF_FFFF;

impl RunLoop {
    /// The whole Starting → Running → (Draining|Idle|Failed|Disabled)
    /// pipeline. State transitions within this body never escape back to the
    /// outer dispatcher; we only return when the child has been cleaned up and
    /// the next outer-loop state is either a terminal variant or Idle.
    pub(crate) async fn handle_active_lifecycle(&mut self) -> Step {
        // Pull the latest ServiceConfig for this spawn. `render_argv`,
        // `HealthConfig`, and the per-command branches below all read
        // fields that a reload may have changed (context, override_tensor,
        // cache_type_k/v, sampling, health probe settings, etc.).
        let current = self.current_svc();
        // When the placement engine has run (`packed_for_spawn = Some`), use
        // its computed Allocation for CUDA_VISIBLE_DEVICES rendering —
        // `self.init.allocation` is built from `placement_override` at
        // registry-init time and is empty for any estimator-driven service,
        // which would otherwise leave the child with `CUDA_VISIBLE_DEVICES=`
        // and silently fall back to CPU.
        let spawn_alloc = self
            .packed_for_spawn
            .as_ref()
            .map(|p| &p.allocation)
            .unwrap_or(&self.init.allocation);
        let spawn_cfg = match render_argv(
            &current,
            spawn_alloc,
            self.packed_for_spawn.as_ref().map(|p| &p.args),
        ) {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "placeholder substitution failed; aborting spawn");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("placeholder substitution failed: {e}"),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Failed { retry_count: 0 });
                return Step::Continue;
            }
        };
        let cmdline = format!("{} {}", spawn_cfg.binary, spawn_cfg.args.join(" "));
        info!(service = %self.init.identity.name, binary = %spawn_cfg.binary, "spawning child");
        let mut child = match self.deps.system.process_spawner.spawn(&spawn_cfg).await {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "spawn failed");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::LaunchFailed,
                        message: format!("{e}"),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Failed { retry_count: 0 });
                return Step::Continue;
            }
        };

        let pid = child.id().unwrap_or(0) as i32;
        self.deps
            .observation
            .register(&self.init.identity.name, pid as u32);
        self.deps.observation.set_cgroup_parent(
            &self.init.identity.name,
            current.tracking.cgroup_parent.clone(),
        );
        let spawn_time = Instant::now();
        let run_id = crate::tracking::now_unix_ms() & RUN_ID_MASK;
        let allocation_json = serde_json::to_string(
            &self
                .init
                .allocation
                .bytes
                .iter()
                .map(|(k, v)| (k.as_display(), *v))
                .collect::<std::collections::BTreeMap<_, _>>(),
        )
        .unwrap_or_default();
        insert_running_row(
            &self.deps.db,
            self.init.service_id,
            run_id,
            pid as i64,
            cmdline.clone(),
            allocation_json,
        )
        .await;
        self.set_running_ids(run_id, pid);

        if let Some(stdout) = child.take_stdout() {
            spawn_pump_stdout(
                stdout,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }
        if let Some(stderr) = child.take_stderr() {
            spawn_pump_stderr(
                stderr,
                self.init.service_id,
                run_id,
                self.deps.batcher.clone(),
            );
        }

        // When no health check path is configured, transition to Running
        // immediately after spawn. The service is assumed ready as soon as
        // the child process exists. Used by oneshots that don't expose an
        // HTTP health endpoint.
        if let Some(http_path) = &current.health.http_path {
            let health_cfg = HealthConfig {
                url: format!(
                    "http://127.0.0.1:{}{}",
                    self.init.identity.private_port, http_path
                ),
                probe_interval: Duration::from_millis(current.health.probe_interval_ms),
                timeout: Duration::from_millis(current.health.timeout_ms),
            };

            let cancel_rx_h = self.cancel_rx.clone();
            let health_task = tokio::spawn(wait_healthy(health_cfg, cancel_rx_h));
            tokio::pin!(health_task);

            loop {
                tokio::select! {
                    exit = child.wait() => {
                        return self.on_child_exit_during_start(exit, spawn_time);
                    }
                    outcome = &mut health_task => {
                        match self.on_health_outcome(outcome, &mut *child, run_id).await {
                            StartingOutcome::Continue => {}
                            StartingOutcome::Break => break,
                            StartingOutcome::Exit => return Step::Exit,
                        }
                    }
                    cmd = self.rx.recv() => {
                        match self.on_starting_command(cmd, &mut *child, run_id).await {
                            StartingOutcome::Continue => {}
                            StartingOutcome::Break => break,
                            StartingOutcome::Exit => return Step::Exit,
                        }
                    }
                }
            }
        } else {
            // No health check: transition to Running immediately.
            match self
                .on_health_outcome(Ok(HealthOutcome::Healthy), &mut *child, run_id)
                .await
            {
                StartingOutcome::Continue => {}
                StartingOutcome::Break => {}
                StartingOutcome::Exit => return Step::Exit,
            }
        }
        Step::Continue
    }

    /// Child exited while we were still in Starting (before the health
    /// probe passed). Detects OOM, updates state, and notifies waiters.
    pub(crate) fn on_child_exit_during_start(
        &mut self,
        exit: std::io::Result<std::process::ExitStatus>,
        spawn_time: Instant,
    ) -> Step {
        warn!(?exit, "child exited during starting");
        self.deps
            .allocations
            .lock()
            .remove(&self.init.identity.name);
        self.deps.observation.clear(&self.init.identity.name);
        self.emit_allocation_changed();

        // Detect OOM kill: process died within 30 s and was killed by
        // SIGKILL (kernel OOM killer or cgroup limit).
        let runtime = spawn_time.elapsed();
        let was_sigkill = exit
            .as_ref()
            .ok()
            .and_then(|s| s.signal())
            .map(|sig| sig == libc::SIGKILL)
            .unwrap_or(false);
        if runtime < OOM_KILL_WINDOW && was_sigkill {
            self.oom_attempts += 1;
            if self.oom_attempts >= 2 {
                warn!(service = %self.init.identity.name, attempts = self.oom_attempts, "OOM retry limit reached; disabling");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::Oom,
                        message: "disabled after repeated OOM kills".into(),
                    }));
                }
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::Oom,
                });
            } else {
                warn!(service = %self.init.identity.name, "OOM kill detected; bumping rolling factor for retry");
                self.deps
                    .rolling
                    .bump_for_oom_retry(&self.init.identity.name);
                // Return to Idle so the next Ensure triggers a re-estimated
                // start with the bumped safety factor.
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::Oom,
                        message: "OOM kill; retrying with larger reservation".into(),
                    }));
                }
                self.set_state(ServiceState::Idle);
            }
            return Step::Continue;
        }

        if let Some(bus) = self.start_bus_carry.take() {
            let _ = bus.send(StartOutcome::Err(StartFailure {
                kind: StartFailureKind::LaunchFailed,
                message: "child exited during starting".into(),
            }));
        }
        self.set_state(ServiceState::Failed { retry_count: 0 });
        Step::Continue
    }

    /// Handle the result of the health probe task. On Healthy we transition
    /// to Running and notify waiters; on other outcomes we tear the child
    /// down and update state.
    pub(crate) async fn on_health_outcome(
        &mut self,
        outcome: Result<HealthOutcome, tokio::task::JoinError>,
        child: &mut dyn ManagedChild,
        run_id: i64,
    ) -> StartingOutcome {
        match outcome {
            Ok(HealthOutcome::Healthy) => {
                let next = transition(&self.read_state(), StateEvent::HealthPassed);
                self.set_state(next);
                // The child answered, so its weights are resident and any
                // memory peak sampled from here on is a peak of the whole
                // model rather than of a partial load. This holds because
                // llama.cpp flips `is_ready` only after `load_model` returns
                // and 503s every endpoint until then; an older build that
                // whitelisted `/v1/models` during loading would quietly turn
                // this into "the HTTP server bound".
                self.rolling_base.run_became_ready();

                // Reset the idle window at the moment the service becomes
                // ready. Without this, a stale `last_activity` (left over
                // from the request that preceded the most recent drain) can
                // make the idle deadline already elapsed on the first poll
                // of `run_running_loop`'s select, racing the waiter's
                // post-`await_ensure` ping and draining the freshly-spawned
                // child before it can serve the request that started it.
                *self.init.last_activity.lock() = tokio::time::Instant::now();

                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Ok);
                }

                self.run_running_loop(child, run_id).await
            }
            Ok(HealthOutcome::TimedOut) => {
                warn!(service = %self.init.identity.name, "health timed out; disabling");
                if let Some(bus) = self.start_bus_carry.take() {
                    let _ = bus.send(StartOutcome::Err(StartFailure {
                        kind: StartFailureKind::HealthTimeout,
                        message: "health check timed out".into(),
                    }));
                }
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                self.set_state(ServiceState::Disabled {
                    reason: DisableReason::HealthTimeout,
                });
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Break
            }
            Ok(HealthOutcome::Cancelled) | Err(_) => {
                self.deps
                    .allocations
                    .lock()
                    .remove(&self.init.identity.name);
                self.emit_allocation_changed();
                drain::sigterm_then_sigkill(child, STARTING_SIGTERM_GRACE).await;
                self.run_shutdown_command().await;
                StartingOutcome::Exit
            }
        }
    }
}

/// Insert a `running_services` row.
async fn insert_running_row(
    db: &Database,
    service_id: i64,
    run_id: i64,
    pid: i64,
    command_line: String,
    allocation: String,
) {
    use crate::db::models::RunningService;

    let row = RunningService {
        service_id,
        run_id,
        pid,
        spawned_at: crate::tracking::now_unix_ms(),
        command_line,
        allocation,
        state: "starting".to_string(),
    };
    if let Err(e) = db.insert_running(&row).await {
        warn!(error = %e, "running_services insert failed");
    }
}
