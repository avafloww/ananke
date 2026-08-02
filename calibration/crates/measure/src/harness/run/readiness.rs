//! Wait for a server to finish loading, or decide why it never will.
//!
//! Three outcomes have to be told apart, because each writes a different record:
//! the server answered (`ok`), the process is gone (`failed-to-load`, and its log
//! tail says why), or neither happened inside the timeout (`timeout`, which for a
//! 200 GiB `--no-mmap` load is half an hour rather than a formality).
//!
//! `/health` is the signal because llama.cpp answers it 503 while the model is
//! still loading and 200 once the slots exist. Health is checked *before* the
//! exit status so that a server which answered and then died still counts as
//! loaded — its log carries the buffer sizes, which is the measurement.

use std::time::Duration;

use crate::harness::{
    run::watchdog::SwapWatchdog,
    sys::{Child, Deps},
};

pub(crate) const HEALTH_POLL: Duration = Duration::from_secs(2);

#[derive(Debug, PartialEq)]
pub(crate) enum Readiness {
    /// Serving, after this many seconds since the spawn.
    Loaded {
        load_seconds: f64,
    },
    /// The process exited on its own, with this status.
    Exited(i32),
    TimedOut,
    /// The box started paging; how far past the baseline swap had grown.
    Swapping(f64),
}

/// One server being waited on: which process, on which port, and for how long.
pub(crate) struct ReadinessWait<'a> {
    /// Polled for an early exit, so a server that dies loading is told apart from
    /// one that is merely slow.
    pub child: &'a mut dyn Child,
    pub port: u16,
    /// The clock reading taken at the spawn, which `timeout` runs from.
    pub spawned_at: Duration,
    pub timeout: Duration,
    pub watchdog: &'a mut SwapWatchdog,
}

pub(crate) fn wait_for_ready(deps: &Deps, wait: ReadinessWait<'_>) -> Readiness {
    let ReadinessWait {
        child,
        port,
        spawned_at,
        timeout,
        watchdog,
    } = wait;
    let deadline = spawned_at + timeout;
    loop {
        if deps.http.healthy(port) {
            return Readiness::Loaded {
                load_seconds: (deps.clock.elapsed().saturating_sub(spawned_at)).as_secs_f64(),
            };
        }
        if let Some(status) = child.exit_status() {
            return Readiness::Exited(status);
        }
        if let Some(grown) = watchdog.check(deps.procfs.as_ref()) {
            return Readiness::Swapping(grown);
        }
        if deps.clock.elapsed() > deadline {
            return Readiness::TimedOut;
        }
        deps.clock.sleep(HEALTH_POLL);
    }
}

/// Wait until nothing holds the port.
///
/// A server must be *fully* exited before the next run starts. A leftover process
/// wins the port bind and every later cell then measures that same process at
/// every point — which is how a whole sweep gets invalidated without a single
/// error being logged.
pub(crate) fn wait_for_port(deps: &Deps, port: u16, timeout: Duration) -> bool {
    let deadline = deps.clock.elapsed() + timeout;
    while deps.clock.elapsed() < deadline {
        if deps.http.port_free(port) {
            return true;
        }
        deps.clock.sleep(Duration::from_secs(1));
    }
    false
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes, SpawnRequest};

    fn fakes(spawner: FakeSpawner, procfs: FakeProcFs, http: FakeHttp) -> Fakes {
        Fakes::new(spawner, procfs, FakeGpu::new(), http)
    }

    fn child(deps: &Deps) -> Box<dyn Child> {
        let argv = ["llama-server".to_owned()];
        deps.spawner
            .spawn(SpawnRequest {
                argv: &argv,
                env: Vec::new(),
                log_path: Path::new("/dev/null"),
            })
            .expect("the fake spawner never fails")
    }

    #[test]
    fn a_server_that_answers_after_some_polls_is_loaded_at_that_time() {
        let fakes = fakes(
            FakeSpawner::new(),
            FakeProcFs::new(),
            FakeHttp::new().loading_for(4),
        );
        let deps = fakes.deps();
        let mut child = child(&deps);
        let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), 4.0);
        let readiness = wait_for_ready(
            &deps,
            ReadinessWait {
                child: child.as_mut(),
                port: 18099,
                spawned_at: Duration::ZERO,
                timeout: Duration::from_secs(1800),
                watchdog: &mut watchdog,
            },
        );
        // Four unhealthy polls, each followed by a two-second wait.
        assert_eq!(readiness, Readiness::Loaded { load_seconds: 8.0 });
    }

    #[test]
    fn a_process_that_is_already_gone_is_a_load_failure_not_a_timeout() {
        let fakes = fakes(
            FakeSpawner::new().exited_with(1),
            FakeProcFs::new(),
            FakeHttp::new().never_healthy(),
        );
        let deps = fakes.deps();
        let mut child = child(&deps);
        let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), 4.0);
        assert_eq!(
            wait_for_ready(
                &deps,
                ReadinessWait {
                    child: child.as_mut(),
                    port: 18099,
                    spawned_at: Duration::ZERO,
                    timeout: Duration::from_secs(1800),
                    watchdog: &mut watchdog,
                },
            ),
            Readiness::Exited(1)
        );
        assert_eq!(
            deps.clock.elapsed(),
            Duration::ZERO,
            "the exit is noticed on the first poll rather than waited out"
        );
    }

    #[test]
    fn a_server_that_never_answers_times_out_at_the_deadline() {
        let fakes = fakes(
            FakeSpawner::new(),
            FakeProcFs::new(),
            FakeHttp::new().never_healthy(),
        );
        let deps = fakes.deps();
        let mut child = child(&deps);
        let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), 4.0);
        assert_eq!(
            wait_for_ready(
                &deps,
                ReadinessWait {
                    child: child.as_mut(),
                    port: 18099,
                    spawned_at: Duration::ZERO,
                    timeout: Duration::from_secs(60),
                    watchdog: &mut watchdog,
                },
            ),
            Readiness::TimedOut
        );
        assert!(deps.clock.elapsed() > Duration::from_secs(60));
    }

    /// The load is where swap grows on a hybrid, so the watchdog has to be able
    /// to end this phase rather than only the phases after it.
    #[test]
    fn swap_growth_during_the_load_ends_the_wait() {
        let fakes = fakes(
            FakeSpawner::new(),
            FakeProcFs::new().with_swap_growth_gib(1.0),
            FakeHttp::new().never_healthy(),
        );
        let deps = fakes.deps();
        let mut child = child(&deps);
        let mut watchdog = SwapWatchdog::start(deps.procfs.as_ref(), 2.0);
        assert_eq!(
            wait_for_ready(
                &deps,
                ReadinessWait {
                    child: child.as_mut(),
                    port: 18099,
                    spawned_at: Duration::ZERO,
                    timeout: Duration::from_secs(1800),
                    watchdog: &mut watchdog,
                },
            ),
            Readiness::Swapping(3.0)
        );
    }

    #[test]
    fn a_busy_port_is_waited_out_rather_than_raced() {
        let busy = fakes(
            FakeSpawner::new(),
            FakeProcFs::new(),
            FakeHttp::new().port_busy(),
        );
        let deps = busy.deps();
        assert!(!wait_for_port(&deps, 18099, Duration::from_secs(180)));
        assert!(deps.clock.elapsed() >= Duration::from_secs(180));

        let free = fakes(FakeSpawner::new(), FakeProcFs::new(), FakeHttp::new());
        assert!(wait_for_port(&free.deps(), 18099, Duration::from_secs(180)));
    }
}
