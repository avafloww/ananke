//! The per-service resolver task: samples VRAM, reconciles the pledge book,
//! enforces the over-ceiling watchdog, and resolves growth contention.

use std::{collections::VecDeque, time::Duration};

use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::{
    allocator::{
        AllocationTable,
        balloon::{
            BalloonConfig, WINDOW_SIZE,
            contention::{ContentionAction, overcommitted_gpus_for, resolve_contention},
            growth::detect_growth,
            pledge::reconcile_pledge,
        },
    },
    config::{Lifecycle, manager::ConfigManager},
    daemon::events::EventBus,
    devices::snapshotter::SharedSnapshot,
    supervise::{drain::DrainReason, registry::ServiceRegistry},
    tracking::observation::ObservationTable,
};

/// If the dynamic service exceeds `max_mb * 110 %` for this long, fast-kill it.
const OVER_CEILING_GRACE: Duration = Duration::from_secs(30);

/// Headroom above `max_mb` tolerated before `OVER_CEILING_GRACE` applies, as
/// permille (110 ‰ = +10 %, i.e. 1.10 ×).
const OVER_CEILING_PERMILLE: u64 = 1100;

const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// Inputs to [`spawn_resolver`] beyond the per-service `BalloonConfig` —
/// the shared collaborators every resolver task needs. Bundled into a
/// struct so the spawn signature stays under clippy's argument limit and
/// callers can build the inputs once and clone.
pub struct ResolverDeps {
    pub observation: ObservationTable,
    pub registry: ServiceRegistry,
    pub allocations: std::sync::Arc<Mutex<AllocationTable>>,
    pub events: EventBus,
    /// Live snapshot used to compute per-GPU pledge totals against
    /// physical capacity — the contention resolver only fires when a GPU
    /// the service holds is actually over-pledged.
    pub snapshot: SharedSnapshot,
    /// Live `ConfigManager` so the contention resolver can look up a
    /// peer's lifecycle to apply the persistent-vs-on-demand tie-break.
    pub config: std::sync::Arc<ConfigManager>,
    pub shutdown: watch::Receiver<bool>,
}

/// Spawn a balloon-resolver task for a dynamic service.
///
/// The task runs until `deps.shutdown` fires `true`. Each `SAMPLE_INTERVAL` it:
/// 1. Reads the observed VRAM peak and slides the sample window forward.
/// 2. Reconciles the pledge book against the recent peak so other services'
///    fit decisions see realistic usage rather than the stale `min_mb`
///    floor (rate-limited by [`crate::allocator::balloon::should_update_pledge`]).
/// 3. Enforces the `max_mb * 110 %` ceiling for `OVER_CEILING_GRACE`.
/// 4. Resolves contention by fast-killing the lower-priority side when
///    growth pressure is detected and a borrower is present.
pub fn spawn_resolver(
    service_name: SmolStr,
    cfg: BalloonConfig,
    svc_priority: u8,
    svc_lifecycle: Lifecycle,
    deps: ResolverDeps,
) -> tokio::task::JoinHandle<()> {
    let ResolverDeps {
        observation,
        registry,
        allocations,
        events,
        snapshot,
        config,
        mut shutdown,
    } = deps;
    tokio::spawn(async move {
        let mut window: VecDeque<u64> = VecDeque::with_capacity(WINDOW_SIZE);
        let mut exceeded_since: Option<std::time::Instant> = None;
        let mut tick = tokio::time::interval(SAMPLE_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        return;
                    }
                }
                _ = tick.tick() => {}
            }

            // VRAM-only peak: the dynamic pledge models GPU bytes, not
            // the python interpreter's RSS. Combining the two used to
            // inflate the pledge with multi-GB CPU footprint and trigger
            // false over-commit signals during normal SDXL inference.
            let observed = observation.read_peak_vram(&service_name);
            if window.len() == WINDOW_SIZE {
                window.pop_front();
            }
            window.push_back(observed);

            reconcile_pledge(&service_name, &window, &cfg, &allocations, &events);

            // If observed > max_mb by >10% for >30 s, fast-kill self.
            //
            // The resolver does NOT terminate after firing — fast_kill
            // drains the supervisor, which then re-ensures via its
            // normal lifecycle. The resolver task lives for the whole
            // daemon run (one task per service, spawned at provision
            // time) and re-arms across the kill: window cleared, grace
            // timer reset, observation cleared by the drain itself.
            // Returning here would orphan the resolver for the rest of
            // the daemon's lifetime, leaving subsequent runs of the
            // service without pledge tracking or contention guarding.
            let ceiling = cfg.max_mb * 1024 * 1024 * OVER_CEILING_PERMILLE / 1000;
            if observed > ceiling {
                if let Some(since) = exceeded_since {
                    if since.elapsed() > OVER_CEILING_GRACE {
                        warn!(
                            service = %service_name,
                            observed,
                            max_bytes = cfg.max_mb * 1024 * 1024,
                            "balloon: max_vram_gb exceeded by >10% for >30s; fast-killing dynamic service",
                        );
                        if let Some(handle) = registry.get(&service_name) {
                            handle.fast_kill(DrainReason::Eviction).await;
                        }
                        window.clear();
                        exceeded_since = None;
                        continue;
                    }
                } else {
                    exceeded_since = Some(std::time::Instant::now());
                }
            } else {
                exceeded_since = None;
            }

            // Contention resolver. Pre-condition for firing:
            // 1. Growth detected in the recent sample window.
            // 2. A GPU the service holds is OOM-pressured — physical
            //    NVML free has dropped below `OOM_MARGIN_BYTES`. The
            //    earlier "pledge sum > total" gate over-fired: pledges
            //    are upper-bound reservations (estimator predictions
            //    pad ~10–20 %, dynamic services hold the recent-peak
            //    high-water mark), so two services can pledge to >100 %
            //    without ever actually filling the GPU. Only the kernel
            //    knows when the next allocation is about to fail, and
            //    NVML free is the signal it gives us.
            //
            // When the gate fires, we identify a peer to resolve against
            // using priority + lifecycle: strict numeric priority always
            // wins; at tied priority, on-demand yields to persistent.
            let floor = cfg.min_mb * 1024 * 1024 + cfg.margin_bytes;
            if detect_growth(&window, floor) {
                let reservations = allocations.lock().clone();
                let snap_now = snapshot.read().clone();
                let overcommitted = overcommitted_gpus_for(&service_name, &reservations, &snap_now);
                if overcommitted.is_empty() {
                    debug!(
                        service = %service_name,
                        observed,
                        "balloon: growth detected but no GPU is over-committed; deferring to pledge book",
                    );
                } else {
                    let cfg_snapshot = config.effective();
                    let resolution = resolve_contention(
                        &service_name,
                        svc_priority,
                        svc_lifecycle,
                        &reservations,
                        &overcommitted,
                        &registry,
                        &cfg_snapshot.services,
                    );
                    drop(cfg_snapshot);
                    match resolution {
                        ContentionAction::EvictPeer { peer } => {
                            info!(
                                service = %service_name,
                                peer = %peer,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU; evicting lower-ranked peer",
                            );
                            if let Some(handle) = registry.get(&peer) {
                                handle.fast_kill(DrainReason::Eviction).await;
                            }
                            window.clear();
                        }
                        ContentionAction::YieldSelf { to } => {
                            info!(
                                service = %service_name,
                                to = %to,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU; yielding to higher-ranked peer",
                            );
                            if let Some(handle) = registry.get(&service_name) {
                                handle.fast_kill(DrainReason::Eviction).await;
                            }
                            // Don't terminate — see the over-ceiling
                            // branch above. The resolver re-arms once
                            // the supervisor finishes draining; the
                            // window resets so we don't immediately
                            // re-fire on the same stale samples.
                            window.clear();
                        }
                        ContentionAction::NoCandidate => {
                            debug!(
                                service = %service_name,
                                gpus = ?overcommitted,
                                "balloon: over-committed GPU but no peer to resolve against",
                            );
                        }
                    }
                }
            } else {
                debug!(
                    service = %service_name,
                    observed,
                    window_len = window.len(),
                    "balloon: no growth detected",
                );
            }
        }
    })
}
