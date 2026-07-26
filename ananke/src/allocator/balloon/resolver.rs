//! The per-service resolver task: samples the device the service's
//! reservation sits on, reconciles the pledge book,
//! enforces the over-ceiling watchdog, and resolves growth contention.

use std::time::Duration;

use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::{
    allocator::{
        AllocationTable,
        balloon::{
            BalloonConfig,
            ceiling::{CeilingAction, ceiling_action, ceiling_bytes},
            contention::{ContentionAction, overcommitted_gpus_for, resolve_contention},
            growth::detect_growth,
            pledge::{pledged_slot, reconcile_pledge},
            window::SampleWindow,
        },
    },
    config::{DeviceSlot, Lifecycle, PlacementPolicy, manager::ConfigManager},
    daemon::events::EventBus,
    devices::snapshotter::SharedSnapshot,
    supervise::{drain::DrainReason, registry::ServiceRegistry},
    tracking::observation::ObservationTable,
};

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
/// 1. Reads the service's *current* usage on the pledged device — VRAM for a
///    GPU service, host RSS for a CPU-pinned one — and slides the window
///    forward. Current rather than peak so the window's max is a recent peak
///    that decays, and so a breach can be seen to subside.
/// 2. Reconciles the pledge book against that recent peak so other services'
///    fit decisions see realistic usage rather than the stale `min_mb`
///    floor (rate-limited by [`crate::allocator::balloon::should_update_pledge`]).
/// 3. Enforces the `max_mb * 110 %` ceiling against a sustained breach.
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
        let mut window = SampleWindow::new();
        // `tokio::time::Instant`, not `std::time::Instant`: the grace window
        // has to advance with the same clock as `tick`, or it is unreachable
        // under `start_paused = true` and the whole watchdog goes untested.
        let mut exceeded_since: Option<tokio::time::Instant> = None;
        let ceiling = ceiling_bytes(cfg.max_mb);
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

            // Sample the single component the reservation actually consumes,
            // never the combined footprint. On a GPU service the pledge
            // models GPU bytes, not the python interpreter's RSS: combining
            // the two used to inflate the pledge with multi-GB CPU footprint
            // and trigger false over-commit signals during normal SDXL
            // inference. On a CPU-pinned service the reservation *is* host
            // RAM, so the mirror-image reading applies.
            //
            // Which one that is has to be re-read each tick rather than fixed
            // at provision time: the slot is chosen per reservation, so a
            // service can land on a GPU one run and the CPU the next. The
            // window resets itself when that happens.
            let slot = pledged_slot_of(&allocations, &service_name);
            let observed = match slot {
                Some(DeviceSlot::Cpu) => observation.read_current_rss(&service_name),
                // No row yet (idle/draining) reads VRAM, matching the
                // pre-existing behaviour for a service that hasn't pledged.
                Some(DeviceSlot::Gpu(_)) | None => observation.read_current_vram(&service_name),
            };
            let ceiling_enforceable =
                cpu_intended(&config, &service_name) == matches!(slot, Some(DeviceSlot::Cpu));
            if window.push(slot, observed) {
                // The retained breach was measured against a device this
                // service no longer holds; it says nothing about this one.
                exceeded_since = None;
            }

            reconcile_pledge(&service_name, window.samples(), &cfg, &allocations, &events);

            // Fast-kill self once current usage has stayed above
            // `max_mb * 110 %` for the whole grace period. The reading is
            // current, so a spike that subsides disarms the timer instead of
            // latching — which is what made this a kill/respawn loop when the
            // input was a high-water mark.
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
            // A GPU service that landed on the CPU did so because the
            // snapshot reported no GPUs — an NVML init failure, which
            // CONTRIBUTING flags as a real NixOS condition. Its `max_mb` is
            // denominated in VRAM, so comparing it against python's RSS would
            // fast-kill a service that is behaving perfectly. Enforce the
            // ceiling only where the reservation's device matches what the
            // operator actually asked for.
            if !ceiling_enforceable {
                exceeded_since = None;
            }
            let action = if ceiling_enforceable {
                ceiling_action(observed, ceiling, exceeded_since.map(|at| at.elapsed()))
            } else {
                CeilingAction::Disarm
            };
            match action {
                CeilingAction::Disarm => exceeded_since = None,
                CeilingAction::Arm => exceeded_since = Some(tokio::time::Instant::now()),
                CeilingAction::Wait => {}
                CeilingAction::Kill => {
                    warn!(
                        service = %service_name,
                        observed,
                        max_bytes = cfg.max_mb * 1024 * 1024,
                        "balloon: max_reserve_gb exceeded by >10% for >30s; fast-killing dynamic service",
                    );
                    if let Some(handle) = registry.get(&service_name) {
                        handle.fast_kill(DrainReason::Eviction).await;
                    }
                    window.clear();
                    exceeded_since = None;
                    continue;
                }
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
            if detect_growth(window.samples(), floor) {
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
                    window_len = window.samples().len(),
                    "balloon: no growth detected",
                );
            }
        }
    })
}

/// Read the slot a service's reservation currently sits on, holding the
/// allocation lock only for the lookup. Split out so the guard is provably
/// dropped before the `.await`-bearing body that follows.
fn pledged_slot_of(
    allocations: &Mutex<AllocationTable>,
    service_name: &SmolStr,
) -> Option<DeviceSlot> {
    allocations.lock().get(service_name).and_then(pledged_slot)
}

/// Whether the operator has declared this service CPU-placed, read from the
/// *live* config rather than frozen at provision time.
///
/// A reload can flip `placement` without restarting the daemon — the resolver
/// task outlives the edit, and `reconciler` handles only add/remove because
/// the supervisor re-reads config at Ensure time. Freezing this would leave a
/// service whose placement changed with a permanently unenforceable ceiling.
///
/// CPU intent covers both ways of expressing it: a `cpu-only` policy, or a
/// `placement_override` that pins every slot to the CPU. An unknown service
/// (removed by a reload before its resolver noticed) reports `false`, matching
/// the `None`-slot reading of VRAM.
fn cpu_intended(config: &ConfigManager, service_name: &SmolStr) -> bool {
    config
        .effective()
        .services
        .iter()
        .find(|s| s.name == *service_name)
        .is_some_and(|svc| {
            matches!(svc.placement_policy, PlacementPolicy::CpuOnly)
                || (!svc.placement_override.is_empty()
                    && svc
                        .placement_override
                        .keys()
                        .all(|slot| matches!(slot, DeviceSlot::Cpu)))
        })
}
