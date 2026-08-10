//! Shared lookup from service name to `SupervisorHandle`.
//!
//! The generic `ServiceRegistry<T>` lives in `ananke-placement`; this
//! module instantiates it with the daemon's `SupervisorHandle` and
//! re-exports the concrete alias so `crate::supervise::registry::…`
//! keeps resolving.

pub use ananke_placement::{DrainReason, KillHandle, ServiceRegistry, slot_to_key};

use crate::supervise::SupervisorHandle;

/// The daemon's registry: service name → supervisor handle.
pub type SupervisorRegistry = ServiceRegistry<SupervisorHandle>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smol_str::SmolStr;

    use super::*;
    use crate::{
        config::validate::{Lifecycle, test_fixtures::minimal_service},
        db::{Database, logs::spawn as spawn_batcher},
        devices::Allocation,
        supervise::spawn_supervisor,
    };

    #[tokio::test(flavor = "current_thread")]
    async fn insert_and_get() {
        let db = Database::open_in_memory().await.unwrap();
        let mut svc = minimal_service("demo");
        svc.lifecycle = Lifecycle::Persistent;
        let effective = crate::config::EffectiveConfig {
            daemon: crate::config::DaemonSettings::default(),
            services: vec![svc.clone()],
        };
        let events = ananke_events::EventBus::new();
        let config = crate::config::manager::ConfigManager::in_memory(effective, events.clone());
        let init = crate::supervise::SupervisorInit {
            identity: crate::supervise::ServiceIdentity::from_service(&svc),
            allocation: Allocation::from_override(&svc.placement_override),
            service_id: 1,
            last_activity: Arc::new(parking_lot::Mutex::new(tokio::time::Instant::now())),
            inflight: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        let deps = crate::supervise::SupervisorDeps {
            db: db.clone(),
            batcher: spawn_batcher(db),
            snapshot: crate::devices::snapshotter::new_shared(),
            allocations: Arc::new(parking_lot::Mutex::new(
                crate::allocator::AllocationTable::new(),
            )),
            rolling: crate::tracking::rolling::RollingTable::new(),
            observation: crate::tracking::observation::ObservationTable::new(),
            registry: SupervisorRegistry::new(),
            config,
            events,
            system: crate::system::SystemDeps::fake().0,
            inflight: crate::tracking::inflight::InflightTable::new(),
            activity: crate::tracking::activity::ActivityTable::new(),
            estimate_cache: crate::supervise::estimate_cache::EstimateCacheHandle::new(),
        };
        let handle = Arc::new(spawn_supervisor(init, svc.clone(), deps));

        let registry = SupervisorRegistry::new();
        registry.insert(SmolStr::new("demo"), handle.clone());
        assert!(registry.get("demo").is_some());
        assert!(registry.get("missing").is_none());
        assert_eq!(registry.names(), vec![SmolStr::new("demo")]);

        // Remove returns the evicted handle; subsequent lookups miss.
        let taken = registry.remove("demo").expect("registry had demo");
        assert!(registry.get("demo").is_none());
        assert_eq!(registry.names(), Vec::<SmolStr>::new());
        // Shutdown the evicted handle so the supervisor actor exits cleanly.
        taken.shutdown().await;

        handle.shutdown().await;
    }
}
