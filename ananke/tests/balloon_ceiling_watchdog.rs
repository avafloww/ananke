//! Scenario: the balloon resolver's over-ceiling watchdog fast-kills a
//! dynamic service that overruns `max_reserve_gb`, but only while the overrun
//! is *sustained*.
//!
//! These tests pin the current-reading semantics. A monotonic high-water
//! mark would make "spiked once" and "permanently overrunning" the same
//! observation: the first spike latches, the service is killed, respawns,
//! climbs back to the same spike, and is killed again — a loop nothing in
//! the service's own behaviour can break.
//!
//! Drives the resolver under tokio's paused clock so the
//! `SAMPLE_INTERVAL`-driven loop can be advanced deterministically.
#![cfg(feature = "test-fakes")]

mod common;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use ananke::{
    allocator::{
        AllocationTable,
        balloon::{BalloonConfig, ResolverDeps, spawn_resolver},
    },
    config::{DaemonSettings, DeviceSlot, EffectiveConfig, Lifecycle, manager::ConfigManager},
    daemon::events::EventBus,
    devices::snapshotter,
    supervise::{SupervisorCommand, SupervisorHandle, registry::ServiceRegistry},
    tracking::observation::ObservationTable,
};
use common::minimal_llama_service;
use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;

const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// Ticks that comfortably outlast the 30 s grace period at a 2 s cadence.
const TICKS_PAST_GRACE: usize = 20;

fn mb(n: u64) -> u64 {
    n * 1024 * 1024
}

/// One resolver under test, plus the mailbox of the supervisor handle it was
/// given. The mailbox is the point: `stub_for_test` swallows commands, so a
/// watchdog assertion made against it would pass whether or not the kill was
/// ever sent.
struct Harness {
    svc: SmolStr,
    allocations: Arc<Mutex<AllocationTable>>,
    observation: ObservationTable,
    mailbox: tokio::sync::mpsc::Receiver<SupervisorCommand>,
    shutdown: watch::Sender<bool>,
    _join: tokio::task::JoinHandle<()>,
}

impl Harness {
    fn new(service: &str, max_mb: u64, slot: DeviceSlot) -> Self {
        let cpu_intended = matches!(slot, DeviceSlot::Cpu);
        Self::with_intent(service, max_mb, slot, cpu_intended)
    }

    /// Build a harness whose declared placement intent may differ from the slot
    /// the reservation actually landed on — the NVML-fallback shape.
    fn with_intent(service: &str, max_mb: u64, slot: DeviceSlot, cpu_intended: bool) -> Self {
        let min_mb = 2 * 1024;
        let svc = SmolStr::new(service);
        let allocations = Arc::new(Mutex::new(AllocationTable::new()));
        insert_row(&allocations, &svc, slot, min_mb);

        let observation = ObservationTable::new();
        let registry = ServiceRegistry::new();
        let (handle, mailbox) = SupervisorHandle::stub_with_mailbox();
        registry.insert(svc.clone(), Arc::new(handle));
        let events = EventBus::new();
        let (shutdown, shutdown_rx) = watch::channel(false);

        // Empty snapshot and config: the contention path reads both and can
        // find no over-committed GPU without them, which keeps these tests
        // scoped to the ceiling watchdog.
        let join = spawn_resolver(
            svc.clone(),
            BalloonConfig {
                min_mb,
                max_mb,
                min_borrower_runtime: Duration::from_secs(60),
                margin_bytes: 512 * 1024 * 1024,
            },
            50, // priority
            Lifecycle::OnDemand,
            ResolverDeps {
                observation: observation.clone(),
                registry,
                allocations: allocations.clone(),
                events: events.clone(),
                snapshot: snapshotter::new_shared(),
                // Placement intent is read from the *live* config each tick,
                // so the service has to actually be in it with the placement
                // the test means to exercise.
                config: ConfigManager::in_memory(
                    EffectiveConfig {
                        daemon: DaemonSettings::default(),
                        services: vec![service_with_placement(service, cpu_intended)],
                    },
                    events,
                ),
                shutdown: shutdown_rx,
            },
        );
        Self {
            svc,
            allocations,
            observation,
            mailbox,
            shutdown,
            _join: join,
        }
    }

    /// Report the service's current usage. Unlike a high-water mark, a lower
    /// number here really is a lower reading.
    fn observe(&self, vram: u64, rss: u64) {
        self.observation.record_sample(
            &self.svc,
            vram,
            ananke::system::Rss {
                total: rss,
                owned: rss,
                file: 0,
            },
        );
    }

    async fn run_ticks(&self, n: usize) {
        for _ in 0..n {
            tokio::time::advance(SAMPLE_INTERVAL + Duration::from_millis(50)).await;
            tokio::task::yield_now().await;
        }
    }

    fn expect_fast_kill(&mut self, context: &str) {
        let cmd = self
            .mailbox
            .try_recv()
            .unwrap_or_else(|e| panic!("{context}: expected a command, got {e:?}"));
        assert!(
            matches!(cmd, SupervisorCommand::FastKill { .. }),
            "{context}: expected FastKill, got {cmd:?}"
        );
    }

    fn expect_no_command(&mut self, context: &str) {
        assert!(
            self.mailbox.try_recv().is_err(),
            "{context}: the resolver must not have sent a command"
        );
    }
}

fn insert_row(
    allocations: &Mutex<AllocationTable>,
    svc: &SmolStr,
    slot: DeviceSlot,
    pledge_mb: u64,
) {
    let mut row = BTreeMap::new();
    row.insert(slot, pledge_mb);
    allocations.lock().insert(svc.clone(), row);
}

/// The baseline the watchdog exists for: usage that stays over the ceiling
/// for longer than the grace period gets the service killed.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn sustained_breach_fast_kills() {
    let mut h = Harness::new("gpu-svc", 8 * 1024, DeviceSlot::Gpu(0));

    // 12 GB VRAM against an 8 GB ceiling — past the +10 % tolerance.
    h.observe(mb(12 * 1024), 0);
    h.run_ticks(TICKS_PAST_GRACE).await;

    h.expect_fast_kill("a breach held for well past the grace period");

    let _ = h.shutdown.send(true);
}

/// A spike that subsides inside the grace period must leave no trace. With a
/// high-water-mark input the spike latches and the kill lands 30 s later
/// regardless of what the service does next.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn transient_spike_above_the_ceiling_does_not_fast_kill() {
    let mut h = Harness::new("gpu-svc", 8 * 1024, DeviceSlot::Gpu(0));

    // ~10 s over the ceiling, comfortably inside the 30 s grace period.
    h.observe(mb(12 * 1024), 0);
    h.run_ticks(5).await;
    h.expect_no_command("mid-spike, still inside the grace period");

    // Then the service settles back well under the ceiling and stays there.
    h.observe(mb(3 * 1024), 0);
    h.run_ticks(30).await;

    h.expect_no_command("a spike that subsided");

    let _ = h.shutdown.send(true);
}

/// The kill/respawn loop, stated end to end. After the watchdog fires, the
/// supervisor drains the service (dropping its row and clearing the
/// observation) and re-ensures it; the fresh run sits under its ceiling and
/// must survive. A latched peak would kill it again every 30 s.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn a_service_back_under_its_ceiling_is_not_killed_again() {
    let mut h = Harness::new("gpu-svc", 8 * 1024, DeviceSlot::Gpu(0));

    h.observe(mb(12 * 1024), 0);
    h.run_ticks(TICKS_PAST_GRACE).await;
    h.expect_fast_kill("the first, justified kill");

    // What the drain does: the reservation goes away and the observation is
    // cleared.
    h.allocations.lock().remove(&h.svc);
    h.observation.clear(&h.svc);
    h.run_ticks(3).await;

    // The supervisor re-ensures the service, and the fresh run does exactly
    // what the first one did: a brief spike on load, then a working set well
    // inside the ceiling. This is the step that closes the loop under a
    // latching peak: the spike latches and buys another kill 30 s later,
    // every time, forever.
    insert_row(&h.allocations, &h.svc, DeviceSlot::Gpu(0), 2 * 1024);
    h.observe(mb(12 * 1024), 0);
    h.run_ticks(3).await;
    h.observe(mb(6 * 1024), 0);
    h.run_ticks(40).await;

    h.expect_no_command("the respawned run settled inside its ceiling");

    let _ = h.shutdown.send(true);
}

/// The over-ceiling watchdog has to fire for a CPU-pinned service too, which
/// means reading host RSS: sampling VRAM there reads zero however far the
/// service overruns.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cpu_pinned_service_trips_the_watchdog_on_rss() {
    let mut h = Harness::new("cpu-svc", 8 * 1024, DeviceSlot::Cpu);

    h.observe(0, mb(12 * 1024));
    h.run_ticks(TICKS_PAST_GRACE).await;

    h.expect_fast_kill("12 GB of RSS against an 8 GB ceiling");

    let _ = h.shutdown.send(true);
}

/// The watchdog must not fire while the service is inside its ceiling — the
/// same RSS reading that trips an 8 GB ceiling is fine under a 20 GB one.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cpu_pinned_service_under_its_ceiling_is_not_killed() {
    let mut h = Harness::new("cpu-svc", 20 * 1024, DeviceSlot::Cpu);

    h.observe(0, mb(12 * 1024));
    h.run_ticks(TICKS_PAST_GRACE).await;

    h.expect_no_command("RSS well inside a 20 GB ceiling");

    let _ = h.shutdown.send(true);
}

/// The mirror image: a GPU service's ceiling is a VRAM ceiling. An SDXL-shaped
/// workload holding 40 GB of host RSS against 3 GB of VRAM is nowhere near its
/// 8 GB VRAM ceiling, and counting the RSS would evict it constantly.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn gpu_service_ceiling_ignores_host_rss() {
    let mut h = Harness::new("gpu-svc", 8 * 1024, DeviceSlot::Gpu(0));

    h.observe(mb(3 * 1024), mb(40 * 1024));
    h.run_ticks(TICKS_PAST_GRACE).await;

    h.expect_no_command("a GPU service is held to its VRAM, not its RSS");

    let _ = h.shutdown.send(true);
}

/// A GPU service lands on `DeviceSlot::Cpu` when the snapshot reports no GPUs
/// at all — an NVML init failure, which CONTRIBUTING flags as a real NixOS
/// condition. Its `max_reserve_gb` is a VRAM budget; measuring it against the
/// python interpreter's RSS would fast-kill a service that is behaving
/// perfectly, and the drain-respawn cycle would repeat it forever.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn gpu_service_fallen_back_to_cpu_is_not_killed_on_rss() {
    // The reservation landed on the CPU by fallback, not by declaration.
    let mut h = Harness::with_intent("comfy", 12 * 1024, DeviceSlot::Cpu, false);

    // RSS far above the VRAM-denominated ceiling, sustained well past grace.
    h.observe(0, 40 * 1024 * 1024 * 1024);
    h.run_ticks(40).await;

    assert!(
        h.mailbox.try_recv().is_err(),
        "a VRAM ceiling must not be enforced against host RSS after an NVML \
         fallback; this was an unbreakable kill/respawn loop"
    );

    let _ = h.shutdown.send(true);
}

/// A minimal dynamic command service whose declared placement matches what the
/// test intends, so the resolver's live intent lookup resolves correctly.
fn service_with_placement(name: &str, cpu_only: bool) -> ananke::config::ServiceConfig {
    let mut svc = minimal_llama_service(name, 0);
    svc.placement_override.clear();
    svc.placement_policy = if cpu_only {
        ananke::config::PlacementPolicy::CpuOnly
    } else {
        ananke::config::PlacementPolicy::GpuOnly
    };
    svc
}
