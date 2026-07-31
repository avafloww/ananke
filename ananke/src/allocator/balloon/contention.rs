//! Picking which peer yields when a balloon wants to grow onto a
//! over-committed GPU.

use smol_str::SmolStr;

use crate::{
    allocator::AllocationTable,
    config::{
        Lifecycle,
        validate::{DEFAULT_SERVICE_PRIORITY, DeviceSlot},
    },
    supervise::registry::ServiceRegistry,
};

/// Outcome of the contention resolver's peer pick.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ContentionAction {
    /// We outrank a peer on the over-committed GPU; evict it.
    EvictPeer { peer: SmolStr },
    /// A peer outranks us; yield by fast-killing self.
    YieldSelf { to: SmolStr },
    /// GPU is over-committed but no peer is a valid contention partner
    /// (e.g. we're alone on it, or every peer is itself this service).
    NoCandidate,
}

/// Headroom (in bytes) the balloon's next growth tick may consume.
/// A GPU whose pledge sum leaves less than this much slack is treated
/// as over-committed — the balloon won't be able to climb without
/// eating into a peer's reservation. Aligned with
/// [`crate::allocator::balloon::BalloonConfig::margin_bytes`] by convention.
const OVERCOMMIT_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// GPUs the named service holds an allocation on where the sum of all
/// pledges already eats into the [`OVERCOMMIT_MARGIN_BYTES`] growth
/// headroom. Empty when there's still slack on every GPU we touch.
///
/// The check is arithmetic over pledges rather than over NVML
/// `free_bytes`. Reading physical free memory is strictly safer but
/// symmetrically wrong: when a balloon *wants* to grow (detected
/// separately by `detect_growth`), peer pledges that arithmetically
/// don't fit alongside it are never evicted until the GPU is on the
/// literal edge of OOM. What keeps the arithmetic form from over-firing
/// is the caller's `detect_growth` gate — a stale recent-peak pledge
/// with no actual climb in the sample window won't trigger eviction,
/// even if the pledge sum arithmetically exceeds the GPU total.
pub(crate) fn overcommitted_gpus_for(
    service_name: &SmolStr,
    reservations: &AllocationTable,
    snapshot: &crate::devices::DeviceSnapshot,
) -> Vec<u32> {
    let mine: std::collections::BTreeSet<u32> = reservations
        .get(service_name)
        .map(|row| {
            row.keys()
                .filter_map(|s| match s {
                    DeviceSlot::Gpu(id) => Some(*id),
                    DeviceSlot::Cpu => None,
                })
                .collect()
        })
        .unwrap_or_default();
    if mine.is_empty() {
        return Vec::new();
    }
    snapshot
        .gpus
        .iter()
        .filter(|g| mine.contains(&g.id))
        .filter(|g| {
            let pledged_bytes_on_gpu = pledged_bytes_on_slot(reservations, DeviceSlot::Gpu(g.id));
            pledged_bytes_on_gpu + OVERCOMMIT_MARGIN_BYTES > g.total_bytes
        })
        .map(|g| g.id)
        .collect()
}

/// Sum every service's pledged MiB on `slot`, returning bytes. Used by
/// the contention check to compare reservations against GPU capacity
/// without involving NVML.
fn pledged_bytes_on_slot(reservations: &AllocationTable, slot: DeviceSlot) -> u64 {
    reservations
        .values()
        .filter_map(|row| row.get(&slot).copied())
        .sum::<u64>()
        * 1024
        * 1024
}

/// Pick a contention peer on one of `overcommitted_gpus` and decide whose
/// turn it is to leave. Pure-data over the inputs so unit tests can drive
/// every branch without spawning supervisors.
pub(crate) fn resolve_contention(
    service_name: &SmolStr,
    svc_priority: u8,
    svc_lifecycle: Lifecycle,
    reservations: &AllocationTable,
    overcommitted_gpus: &[u32],
    registry: &ServiceRegistry,
    services: &[crate::config::ServiceConfig],
) -> ContentionAction {
    let lifecycle_of = |name: &SmolStr| -> Lifecycle {
        services
            .iter()
            .find(|s| s.name.as_str() == name.as_str())
            .map(|s| s.lifecycle)
            .unwrap_or(Lifecycle::OnDemand)
    };
    let priority_of = |name: &SmolStr| -> u8 {
        services
            .iter()
            .find(|s| s.name.as_str() == name.as_str())
            .map(|s| s.priority)
            .unwrap_or(DEFAULT_SERVICE_PRIORITY)
    };
    for peer_name in reservations.keys() {
        if peer_name.as_str() == service_name.as_str() {
            continue;
        }
        // Peer must hold a pledge on at least one over-committed GPU.
        let peer_row = match reservations.get(peer_name) {
            Some(r) => r,
            None => continue,
        };
        let intersects = overcommitted_gpus
            .iter()
            .any(|id| peer_row.get(&DeviceSlot::Gpu(*id)).copied().unwrap_or(0) > 0);
        if !intersects {
            continue;
        }
        if registry.get(peer_name).is_none() {
            continue;
        }
        let peer_priority = priority_of(peer_name);
        let peer_lifecycle = lifecycle_of(peer_name);
        if svc_priority > peer_priority {
            return ContentionAction::EvictPeer {
                peer: peer_name.clone(),
            };
        }
        if svc_priority < peer_priority {
            return ContentionAction::YieldSelf {
                to: peer_name.clone(),
            };
        }
        // Tied numeric priority — lifecycle breaks the tie.
        match (svc_lifecycle, peer_lifecycle) {
            (Lifecycle::OnDemand, Lifecycle::Persistent) => {
                return ContentionAction::YieldSelf {
                    to: peer_name.clone(),
                };
            }
            (Lifecycle::Persistent, Lifecycle::OnDemand) => {
                return ContentionAction::EvictPeer {
                    peer: peer_name.clone(),
                };
            }
            // Both same lifecycle — the dynamic side (us, by definition of
            // running this resolver) yields. Avoids two persistent peers
            // oscillating; in practice only one of them is dynamic, so this
            // branch is rare.
            _ => {
                return ContentionAction::YieldSelf {
                    to: peer_name.clone(),
                };
            }
        }
    }
    ContentionAction::NoCandidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::AllocationTable;

    fn snap_with_free(id: u32, total_gb: u64, free_bytes: u64) -> crate::devices::DeviceSnapshot {
        crate::devices::DeviceSnapshot {
            gpus: vec![crate::devices::GpuSnapshot {
                id,
                name: format!("GPU {id}"),
                total_bytes: total_gb * 1024 * 1024 * 1024,
                free_bytes,
            }],
            cpu: None,
            taken_at_ms: 0,
        }
    }

    fn alloc_table(entries: &[(&str, u32, u64)]) -> AllocationTable {
        let mut t = AllocationTable::new();
        for (name, gpu, mb) in entries {
            t.entry(SmolStr::new(*name))
                .or_default()
                .insert(DeviceSlot::Gpu(*gpu), *mb);
        }
        t
    }

    /// Pledge sum exceeds the GPU total — `overcommitted_gpus_for`
    /// reports the GPU as over-committed regardless of what NVML says.
    /// The "kernel won't OOM, so don't fire" reasoning lives at the
    /// *resolver* level instead, in the caller's
    /// `detect_growth` gate: when the balloon's observed window isn't
    /// climbing, the resolver returns early without consulting this
    /// function. Pledge-overcommit + growth → evict; pledge-overcommit
    /// alone → wait.
    #[test]
    fn overcommit_yes_when_pledges_exceed_total() {
        // GPU 1 has 24 GiB total; NVML free is unrelated to this check
        // now (we report based on pledges), so any value works here.
        let snap = snap_with_free(1, 24, 2 * 1024 * 1024 * 1024);
        let table = alloc_table(&[("comfy", 1, 14 * 1024), ("qwen", 1, 12 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert_eq!(
            gpus,
            vec![1],
            "comfy (14 GiB) + qwen (12 GiB) = 26 GiB > 24 GiB — over-committed"
        );
    }

    /// Pledge sum eats into the `OVERCOMMIT_MARGIN_BYTES` (512 MiB)
    /// growth headroom but doesn't yet exceed the GPU total. Still
    /// over-committed: the next growth tick has nowhere to go.
    #[test]
    fn overcommit_yes_when_pledges_within_margin_of_total() {
        let snap = snap_with_free(1, 24, 4 * 1024 * 1024 * 1024);
        // 23.75 GiB pledged on a 24 GiB card → 256 MiB slack, below
        // the 512 MiB growth margin.
        let table = alloc_table(&[("comfy", 1, 12 * 1024), ("qwen", 1, 12 * 1024 - 256)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert_eq!(gpus, vec![1]);
    }

    /// Pledges leave more than `OVERCOMMIT_MARGIN_BYTES` of headroom.
    /// The balloon can keep growing without stealing from a peer, so
    /// the GPU is not over-committed even if NVML reports tight free
    /// space (that would be the *physical* footprint of unrelated
    /// processes, not something the pledge book promised away).
    #[test]
    fn overcommit_no_when_pledges_leave_headroom() {
        let snap = snap_with_free(1, 24, 100 * 1024 * 1024);
        let table = alloc_table(&[("comfy", 1, 7 * 1024), ("qwen", 1, 12 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert!(
            gpus.is_empty(),
            "7 GiB + 12 GiB = 19 GiB, 5 GiB of slack — plenty of room"
        );
    }

    /// Pressure on a GPU we don't hold isn't our problem.
    #[test]
    fn overcommit_filters_to_held_gpus() {
        let snap = crate::devices::DeviceSnapshot {
            gpus: vec![
                crate::devices::GpuSnapshot {
                    id: 1,
                    name: "GPU 1".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                    free_bytes: 10 * 1024 * 1024 * 1024, // plenty
                },
                crate::devices::GpuSnapshot {
                    id: 2,
                    name: "GPU 2".into(),
                    total_bytes: 24 * 1024 * 1024 * 1024,
                    free_bytes: 50 * 1024 * 1024, // pressured, but not ours
                },
            ],
            cpu: None,
            taken_at_ms: 0,
        };
        let table = alloc_table(&[
            ("comfy", 1, 4 * 1024),
            ("a", 2, 20 * 1024),
            ("b", 2, 10 * 1024),
        ]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("comfy"), &table, &snap);
        assert!(gpus.is_empty());
    }

    /// A service with no allocation has nothing to defend.
    #[test]
    fn overcommit_empty_when_service_has_no_allocation() {
        let snap = snap_with_free(1, 24, 50 * 1024 * 1024);
        let table = alloc_table(&[("a", 1, 30 * 1024)]);
        let gpus = overcommitted_gpus_for(&SmolStr::new("ghost"), &table, &snap);
        assert!(gpus.is_empty());
    }

    fn svc(name: &str, priority: u8, lifecycle: Lifecycle) -> crate::config::ServiceConfig {
        let mut s = crate::config::validate::test_fixtures::minimal_service(name);
        s.priority = priority;
        s.lifecycle = lifecycle;
        s
    }

    /// Synthesise a minimal `ServiceRegistry` with present-but-dead
    /// handles for the named services. The contention resolver only
    /// checks registry membership (`registry.get(...).is_some()`), not
    /// handle health, so this is sufficient for the unit-level pure-data
    /// tests.
    fn with_handles(names: &[&str]) -> ServiceRegistry {
        // We can't construct a real SupervisorHandle here without the full
        // supervisor stack; build a registry with synthetic entries by
        // taking handles from a tiny side-helper that spawns a no-op
        // supervisor. For the pure-data scope of these tests, presence is
        // all that matters, so just clone the same handle into each slot.
        let registry = ServiceRegistry::new();
        let handle = std::sync::Arc::new(crate::supervise::SupervisorHandle::stub_for_test());
        for n in names {
            registry.insert(SmolStr::new(*n), handle.clone());
        }
        registry
    }

    /// Strict numeric priority always wins: an on-demand requester at
    /// priority 70 evicts a tied-lifecycle peer at priority 50.
    #[test]
    fn resolve_strict_priority_wins() {
        let services = vec![
            svc("hi-prio", 70, Lifecycle::OnDemand),
            svc("low-prio", 50, Lifecycle::Persistent),
        ];
        let table = alloc_table(&[("hi-prio", 1, 14 * 1024), ("low-prio", 1, 12 * 1024)]);
        let registry = with_handles(&["hi-prio", "low-prio"]);
        let action = resolve_contention(
            &SmolStr::new("hi-prio"),
            70,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::EvictPeer {
                peer: SmolStr::new("low-prio"),
            }
        );
    }

    /// At tied numeric priority, on-demand requester yields to a
    /// persistent peer (the lifecycle tie-break). Reproduces the user's
    /// scenario: ComfyUI (on-demand) vs Qwen (persistent), both priority 50.
    #[test]
    fn resolve_on_demand_yields_to_persistent_at_tied_priority() {
        let services = vec![
            svc("comfy", 50, Lifecycle::OnDemand),
            svc("qwen", 50, Lifecycle::Persistent),
        ];
        let table = alloc_table(&[("comfy", 1, 10 * 1024), ("qwen", 1, 12 * 1024)]);
        let registry = with_handles(&["comfy", "qwen"]);
        let action = resolve_contention(
            &SmolStr::new("comfy"),
            50,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::YieldSelf {
                to: SmolStr::new("qwen")
            }
        );
    }

    /// Reverse: a persistent dynamic service (rare but possible) at tied
    /// priority evicts an on-demand peer.
    #[test]
    fn resolve_persistent_evicts_on_demand_at_tied_priority() {
        let services = vec![
            svc("dyn-persistent", 50, Lifecycle::Persistent),
            svc("on-demand-peer", 50, Lifecycle::OnDemand),
        ];
        let table = alloc_table(&[
            ("dyn-persistent", 1, 14 * 1024),
            ("on-demand-peer", 1, 12 * 1024),
        ]);
        let registry = with_handles(&["dyn-persistent", "on-demand-peer"]);
        let action = resolve_contention(
            &SmolStr::new("dyn-persistent"),
            50,
            Lifecycle::Persistent,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(
            action,
            ContentionAction::EvictPeer {
                peer: SmolStr::new("on-demand-peer"),
            }
        );
    }

    /// No peer holds an allocation on the over-committed GPU → NoCandidate
    /// (the resolver leaves the situation alone; the kernel will OOM
    /// whichever side is allocating, which is semantically correct).
    #[test]
    fn resolve_no_candidate_when_alone_on_overcommit_gpu() {
        let services = vec![svc("comfy", 50, Lifecycle::OnDemand)];
        let table = alloc_table(&[("comfy", 1, 26 * 1024)]); // self-overcommitting somehow
        let registry = with_handles(&["comfy"]);
        let action = resolve_contention(
            &SmolStr::new("comfy"),
            50,
            Lifecycle::OnDemand,
            &table,
            &[1],
            &registry,
            &services,
        );
        assert_eq!(action, ContentionAction::NoCandidate);
    }
}
