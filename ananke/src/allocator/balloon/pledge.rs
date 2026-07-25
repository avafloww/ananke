//! Pledge-book reconciliation: turning a recent VRAM observation window
//! into the pledge a dynamic service should hold, and deciding when that
//! change is worth publishing.

use std::{collections::VecDeque, time::Duration};

use parking_lot::Mutex;
use smol_str::SmolStr;
use tracing::debug;

use crate::{
    allocator::AllocationTable, config::validate::DeviceSlot, daemon::events::EventBus,
    supervise::slot_to_key,
};

/// Pledge update rate-limit: ignore deltas smaller than this many MiB. A
/// dynamic service drifting by a few hundred MiB shouldn't churn the event
/// stream; a 5 % relative threshold catches the same pattern at higher pledges
/// (a 12 GiB pledge → 600 MiB sensitivity).
const PLEDGE_DELTA_FLOOR_MB: u64 = 256;
const PLEDGE_DELTA_PERMILLE: u64 = 50; // 5.0 %

#[derive(Debug, Clone)]
pub struct BalloonConfig {
    pub min_mb: u64,
    pub max_mb: u64,
    /// Minimum time a borrower must have been alive before we fast-kill it.
    pub min_borrower_runtime: Duration,
    /// Extra headroom added to the `min_mb` floor for growth detection.
    pub margin_bytes: u64,
}

/// Compute the pledge a dynamic service should hold given a recent
/// observation window. The window's max acts as a "recent peak" — transient
/// spikes lift the pledge while the corresponding sample is in the window
/// and decay back as samples roll out, so a one-time burst doesn't pin the
/// reservation forever. Always clamped to `[min_mb, max_mb]` per the
/// dynamic-allocation contract.
///
/// Returns `None` when the window is empty (no sample has been taken yet).
pub fn pledge_from_window(window: &VecDeque<u64>, min_mb: u64, max_mb: u64) -> Option<u64> {
    let peak_bytes = *window.iter().max()?;
    let peak_mb = peak_bytes / (1024 * 1024);
    Some(peak_mb.clamp(min_mb, max_mb))
}

/// Decide whether a pledge change is meaningful enough to push to the
/// allocation table (and out via `AllocationChanged`).
///
/// Rules:
/// - First-ever update (no current pledge) always emits.
/// - No change → don't emit.
/// - Crossing the `min_mb` boundary (baseline ↔ above-baseline) always
///   emits; peers should know immediately when slack appears or vanishes.
/// - Otherwise gate on a delta of at least 5 % of the current pledge or
///   `PLEDGE_DELTA_FLOOR_MB`, whichever is larger.
pub fn should_update_pledge(current_mb: Option<u64>, desired_mb: u64, min_mb: u64) -> bool {
    let Some(prev) = current_mb else {
        return true;
    };
    if prev == desired_mb {
        return false;
    }
    if (prev > min_mb) != (desired_mb > min_mb) {
        return true;
    }
    let delta = prev.abs_diff(desired_mb);
    let percent_threshold = (prev * PLEDGE_DELTA_PERMILLE) / 1000;
    delta >= percent_threshold.max(PLEDGE_DELTA_FLOOR_MB)
}

/// Update the pledge book row for a dynamic service so it reflects the
/// recent observed peak (clamped to `[min_mb, max_mb]`), and emit
/// `AllocationChanged` when the change clears the rate-limit gate.
///
/// The slot the service holds is read from its own row in
/// [`AllocationTable`]. Command-template services hold a single slot at a
/// time, so the first non-CPU entry (or the only entry) is the one to
/// update. No-op when the service has no row (idle, draining, or never
/// started).
pub(crate) fn reconcile_pledge(
    service_name: &SmolStr,
    window: &VecDeque<u64>,
    cfg: &BalloonConfig,
    allocations: &Mutex<AllocationTable>,
    events: &EventBus,
) {
    let Some(desired_mb) = pledge_from_window(window, cfg.min_mb, cfg.max_mb) else {
        return;
    };

    let mut guard = allocations.lock();
    let Some(row) = guard.get_mut(service_name) else {
        return;
    };
    // Pick the GPU slot the service is pinned to. Falls back to the only
    // entry (which may legitimately be Cpu in test setups) so a CPU-spilled
    // dynamic service still sees its pledge tracked.
    let target_slot = row
        .keys()
        .find(|s| matches!(s, DeviceSlot::Gpu(_)))
        .or_else(|| row.keys().next())
        .cloned();
    let Some(slot) = target_slot else {
        return;
    };
    let current_mb = row.get(&slot).copied();
    if !should_update_pledge(current_mb, desired_mb, cfg.min_mb) {
        return;
    }
    debug!(
        service = %service_name,
        slot = ?slot,
        previous_mb = ?current_mb,
        desired_mb,
        "balloon: reconciling pledge to observed peak"
    );
    row.insert(slot, desired_mb);

    let reservations: std::collections::BTreeMap<String, u64> = row
        .iter()
        .map(|(s, mb)| (slot_to_key(s), mb * 1024 * 1024))
        .collect();
    drop(guard);
    events.publish(ananke_api::events::Event::AllocationChanged {
        service: service_name.clone(),
        reservations,
        at_ms: crate::tracking::now_unix_ms(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::balloon::test_support::mk_window;

    fn mb(n: u64) -> u64 {
        n * 1024 * 1024
    }

    #[test]
    fn pledge_from_empty_window_is_none() {
        let w = VecDeque::new();
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), None);
    }

    #[test]
    fn pledge_clamps_to_min_when_observed_below() {
        // Observed peak < min_mb (e.g. process just started, hasn't allocated
        // anything yet). Pledge must not fall below the declared floor.
        let w = mk_window(&[mb(500), mb(800)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(2 * 1024));
    }

    #[test]
    fn pledge_clamps_to_max_when_observed_above() {
        // Observed peak > max_mb (transient overshoot — the ceiling watchdog
        // will fast-kill if it persists). Pledge stays at max_mb.
        let w = mk_window(&[mb(2 * 1024), mb(25 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(20 * 1024));
    }

    #[test]
    fn pledge_tracks_window_max_within_range() {
        // Mid-range observations: the pledge mirrors the window max so a
        // transient spike to 12 GiB lifts the pledge to 12 GiB until it rolls
        // out of the window.
        let w = mk_window(&[mb(3 * 1024), mb(12 * 1024), mb(7 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(12 * 1024));
    }

    #[test]
    fn pledge_decays_as_spikes_roll_out() {
        // Once the spike-sample is no longer in the window, the pledge falls
        // back to the new max — peers can reclaim the headroom they were
        // previously locked out of.
        let w = mk_window(&[mb(7 * 1024), mb(6 * 1024), mb(5 * 1024)]);
        assert_eq!(pledge_from_window(&w, 2 * 1024, 20 * 1024), Some(7 * 1024));
    }

    #[test]
    fn first_pledge_always_emits() {
        assert!(should_update_pledge(None, 2 * 1024, 2 * 1024));
    }

    #[test]
    fn no_change_does_not_emit() {
        assert!(!should_update_pledge(Some(8 * 1024), 8 * 1024, 2 * 1024));
    }

    #[test]
    fn baseline_to_above_baseline_always_emits() {
        // Service grew from min_mb (2 GiB floor) to 3 GiB. Even though 1 GiB
        // is below the absolute floor, the boundary transition matters: peers
        // need to know slack on the device just shrank.
        assert!(should_update_pledge(Some(2 * 1024), 3 * 1024, 2 * 1024));
    }

    #[test]
    fn return_to_baseline_always_emits() {
        // Service shrunk back to min_mb. Peers that were locked out should
        // see the headroom return immediately.
        assert!(should_update_pledge(Some(8 * 1024), 2 * 1024, 2 * 1024));
    }

    #[test]
    fn small_drift_does_not_emit() {
        // 50 MiB drift on a 12 GiB pledge: well below 5 % (614 MiB) and the
        // 256 MiB absolute floor. Don't churn the event stream.
        assert!(!should_update_pledge(
            Some(12 * 1024),
            12 * 1024 + 50,
            2 * 1024
        ));
    }

    #[test]
    fn meaningful_growth_emits() {
        // 700 MiB growth on a 12 GiB pledge: above both 5 % (614 MiB) and the
        // 256 MiB absolute floor.
        assert!(should_update_pledge(
            Some(12 * 1024),
            12 * 1024 + 700,
            2 * 1024
        ));
    }

    #[test]
    fn small_pledge_uses_absolute_floor() {
        // 200 MiB drift on a 4 GiB pledge: 5 % is only 204 MiB, but the
        // absolute floor is 256 MiB → don't emit.
        assert!(!should_update_pledge(
            Some(4 * 1024),
            4 * 1024 + 200,
            2 * 1024
        ));
        // 300 MiB drift on the same pledge: above the 256 MiB absolute floor.
        assert!(should_update_pledge(
            Some(4 * 1024),
            4 * 1024 + 300,
            2 * 1024
        ));
    }
}
