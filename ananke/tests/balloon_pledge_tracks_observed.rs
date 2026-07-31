//! Scenario: a dynamic-allocation service's pledge in `AllocationTable` must
//! track its recent observed usage, not stay frozen at `min_mb`. Other
//! services' fit decisions depend on this — a peer that sees a 2 GB pledge
//! while ComfyUI is actually using 10 GB books the apparent headroom and
//! then OOMs at runtime.
//!
//! "Recent" is the resolver's rolling window over *current* readings, so the
//! pledge rises with a spike and falls again once it rolls out. The
//! over-ceiling watchdog that reads the same samples lives in
//! `balloon_ceiling_watchdog.rs`.
//!
//! Drives the resolver under tokio's paused clock so the
//! `SAMPLE_INTERVAL`-driven loop can be advanced deterministically.
#![cfg(feature = "test-fakes")]

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
use ananke_api::events::Event;
use parking_lot::Mutex;
use smol_str::SmolStr;
use tokio::sync::watch;

const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

fn mb(n: u64) -> u64 {
    n * 1024 * 1024
}

/// Wire-up for one resolver under test. Holds the collaborators a test
/// needs to mutate (observation, allocations) plus the receivers used to
/// observe outputs (event bus subscription) and stop the task.
struct Harness {
    svc: SmolStr,
    allocations: Arc<Mutex<AllocationTable>>,
    observation: ObservationTable,
    events_rx: tokio::sync::broadcast::Receiver<Event>,
    shutdown: watch::Sender<bool>,
    _join: tokio::task::JoinHandle<()>,
}

fn build_harness(service: &str, min_mb: u64, max_mb: u64) -> Harness {
    build_harness_on(service, min_mb, max_mb, DeviceSlot::Gpu(0)).0
}

/// Build a resolver harness whose service holds `slot` in the pledge book.
/// The slot decides which reading the resolver samples, so a CPU-pinned
/// service must be built with [`DeviceSlot::Cpu`] to exercise the RSS path.
/// Also returns the mailbox of the registered handle so a test can assert on
/// the commands the resolver sends.
fn build_harness_on(
    service: &str,
    min_mb: u64,
    max_mb: u64,
    slot: DeviceSlot,
) -> (Harness, tokio::sync::mpsc::Receiver<SupervisorCommand>) {
    let svc = SmolStr::new(service);
    let mut row = BTreeMap::new();
    row.insert(slot, min_mb); // MB
    let mut table = AllocationTable::new();
    table.insert(svc.clone(), row);
    let allocations = Arc::new(Mutex::new(table));

    let observation = ObservationTable::new();
    let registry = ServiceRegistry::new();
    let (handle, mailbox) = SupervisorHandle::stub_with_mailbox();
    registry.insert(svc.clone(), Arc::new(handle));
    let events = EventBus::new();
    let events_rx = events.subscribe();
    let (shutdown, shutdown_rx) = watch::channel(false);

    let cfg = BalloonConfig {
        min_mb,
        max_mb,
        min_borrower_runtime: Duration::from_millis(60_000),
        margin_bytes: 512 * 1024 * 1024,
    };
    // Empty config + snapshot — these tests exercise the pledge-reconcile
    // path only, which doesn't read either. The contention path (which
    // does) is covered by separate scenario tests.
    let snapshot = snapshotter::new_shared();
    let config = ConfigManager::in_memory(
        EffectiveConfig {
            daemon: DaemonSettings::default(),
            services: Vec::new(),
        },
        events.clone(),
    );
    let join = spawn_resolver(
        svc.clone(),
        cfg,
        50, // priority
        Lifecycle::OnDemand,
        ResolverDeps {
            observation: observation.clone(),
            registry,
            allocations: allocations.clone(),
            events,
            snapshot,
            config,
            shutdown: shutdown_rx,
        },
    );
    (
        Harness {
            svc,
            allocations,
            observation,
            events_rx,
            shutdown,
            _join: join,
        },
        mailbox,
    )
}

/// Read this service's pledge for `Gpu(0)` from the allocation table.
fn pledge_mb(table: &Mutex<AllocationTable>, service: &SmolStr) -> u64 {
    pledge_mb_on(table, service, &DeviceSlot::Gpu(0))
}

/// Read this service's pledge for an arbitrary slot.
fn pledge_mb_on(table: &Mutex<AllocationTable>, service: &SmolStr, slot: &DeviceSlot) -> u64 {
    table
        .lock()
        .get(service)
        .and_then(|row| row.get(slot).copied())
        .unwrap_or(0)
}

/// Drain any AllocationChanged events for `service` from `rx` and return
/// the latest pledge MB the bus reported (or `None` if no event was queued).
fn latest_event_pledge_mb(
    rx: &mut tokio::sync::broadcast::Receiver<Event>,
    service: &SmolStr,
) -> Option<u64> {
    let mut last: Option<u64> = None;
    while let Ok(ev) = rx.try_recv() {
        if let Event::AllocationChanged {
            service: s,
            reservations,
            ..
        } = ev
            && s.as_str() == service.as_str()
            && let Some(bytes) = reservations.get("gpu:0")
        {
            last = Some(bytes / (1024 * 1024));
        }
    }
    last
}

async fn step() {
    tokio::time::advance(SAMPLE_INTERVAL + Duration::from_millis(50)).await;
    tokio::task::yield_now().await;
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn pledge_grows_to_observed_peak() {
    let mut h = build_harness("comfy", 2 * 1024, 20 * 1024);

    // Initial state: pledge = min_mb (2 GB). No samples yet.
    assert_eq!(pledge_mb(&h.allocations, &h.svc), 2 * 1024);

    // Service grows to 10 GB observed peak. Advance the clock past one
    // sample interval so the resolver picks up the new value.
    h.observation.record_sample(
        &h.svc,
        mb(10 * 1024),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    step().await;

    let p = pledge_mb(&h.allocations, &h.svc);
    assert_eq!(
        p,
        10 * 1024,
        "pledge must track observed peak (10 GB), not stay at min_mb"
    );
    assert_eq!(
        latest_event_pledge_mb(&mut h.events_rx, &h.svc),
        Some(10 * 1024),
        "AllocationChanged must reflect the new pledge"
    );

    let _ = h.shutdown.send(true);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn pledge_clamps_to_max_on_overshoot() {
    let h = build_harness("comfy", 2 * 1024, 20 * 1024);

    // Observed peak above max_mb. The ceiling watchdog handles persistent
    // overshoots; the pledge book just clamps to max so peers see the
    // declared upper bound, not a runaway value.
    h.observation.record_sample(
        &h.svc,
        mb(28 * 1024),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    step().await;

    assert_eq!(pledge_mb(&h.allocations, &h.svc), 20 * 1024);

    let _ = h.shutdown.send(true);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn pledge_decays_as_spike_rolls_out_of_window() {
    let h = build_harness("comfy", 2 * 1024, 20 * 1024);

    // Tick 1: 12 GB spike. Pledge lifts.
    h.observation.record_sample(
        &h.svc,
        mb(12 * 1024),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    step().await;
    assert_eq!(pledge_mb(&h.allocations, &h.svc), 12 * 1024);

    // The service settles back to 4 GB. The resolver samples the *current*
    // reading, so simply reporting the lower number is enough — no clearing
    // the observation table to work around a latched high-water mark. Run
    // several ticks (≥ WINDOW_SIZE) so the 12 GB spike rolls out entirely.
    h.observation.record_sample(
        &h.svc,
        mb(4 * 1024),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    for _ in 0..7 {
        step().await;
    }

    let p = pledge_mb(&h.allocations, &h.svc);
    assert_eq!(
        p,
        4 * 1024,
        "pledge must decay back to the new peak once the spike rolls out; got {p} MB"
    );

    let _ = h.shutdown.send(true);
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn pledge_does_not_emit_for_sub_threshold_drift() {
    let mut h = build_harness("comfy", 2 * 1024, 20 * 1024);

    // Lift to 12 GB, then drift by 50 MB — well below the 5 % / 256 MB
    // rate-limit floor. We expect exactly one AllocationChanged event for
    // the initial 12 GB lift, none for the drift.
    h.observation.record_sample(
        &h.svc,
        mb(12 * 1024),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    step().await;

    let first = latest_event_pledge_mb(&mut h.events_rx, &h.svc);
    assert_eq!(first, Some(12 * 1024));
    assert_eq!(pledge_mb(&h.allocations, &h.svc), 12 * 1024);

    h.observation.record_sample(
        &h.svc,
        mb(12 * 1024) + mb(50),
        ananke::system::Rss {
            total: 0,
            owned: 0,
            file: 0,
        },
    );
    step().await;

    // No new event (rate-limited) and the pledge is unchanged.
    assert_eq!(
        latest_event_pledge_mb(&mut h.events_rx, &h.svc),
        None,
        "sub-threshold drift must not emit a new AllocationChanged"
    );
    assert_eq!(pledge_mb(&h.allocations, &h.svc), 12 * 1024);

    let _ = h.shutdown.send(true);
}

/// A CPU-pinned dynamic service reports no VRAM at all, so its pledge has to
/// come from RSS. Sampling the VRAM peak reads zero forever, leaving the
/// pledge frozen at `min_mb` however much host RAM the service takes — and
/// the packer books hybrid MoE expert spill against exactly that row.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cpu_pinned_service_pledges_from_rss() {
    let (h, _mailbox) = build_harness_on("cpu-svc", 2 * 1024, 20 * 1024, DeviceSlot::Cpu);

    // Zero VRAM, 10 GB RSS — the shape of any cpu-only workload.
    h.observation.record_sample(
        &h.svc,
        0,
        ananke::system::Rss {
            total: mb(10 * 1024),
            owned: mb(10 * 1024),
            file: 0,
        },
    );
    step().await;

    assert_eq!(
        pledge_mb_on(&h.allocations, &h.svc, &DeviceSlot::Cpu),
        10 * 1024,
        "a CPU-pinned service's pledge must track its RSS peak, not stay at min_mb"
    );

    let _ = h.shutdown.send(true);
}

/// The mirror image, and the reason the resolver samples VRAM alone: a GPU
/// service's pledge must not absorb the python interpreter's RSS. An SDXL
/// run at 6 GB VRAM + 30 GB RSS would otherwise pledge as 36 GB and trigger
/// an unjustified self-eviction.
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn gpu_service_pledges_from_vram_and_ignores_rss() {
    let (h, _mailbox) = build_harness_on("gpu-svc", 2 * 1024, 20 * 1024, DeviceSlot::Gpu(0));

    h.observation.record_sample(
        &h.svc,
        mb(6 * 1024),
        ananke::system::Rss {
            total: mb(30 * 1024),
            owned: mb(30 * 1024),
            file: 0,
        },
    );
    step().await;

    assert_eq!(
        pledge_mb(&h.allocations, &h.svc),
        6 * 1024,
        "a GPU service pledges its VRAM peak; RSS must not inflate it"
    );

    let _ = h.shutdown.send(true);
}
