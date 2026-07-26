//! The over-ceiling watchdog's decision function.
//!
//! A dynamic service that overruns `max_reserve_gb` is fast-killed, but only
//! once the overrun has proven itself sustained: the input is the *current*
//! reading, and any sample back inside the ceiling disarms the timer. That
//! distinction is the whole watchdog. Fed a monotonic high-water mark
//! instead, a single spike is indistinguishable from a permanent overrun, so
//! the service is killed, respawns, climbs back to the same spike, and is
//! killed again — an unbreakable loop that nothing in the service's own
//! behaviour can end.

use std::time::Duration;

/// A service has to stay above its ceiling for this long before we kill it.
pub(crate) const OVER_CEILING_GRACE: Duration = Duration::from_secs(30);

/// What the watchdog should do with this tick's sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CeilingAction {
    /// Inside the ceiling — disarm the grace timer.
    Disarm,
    /// First breaching sample after a period inside — start the grace timer.
    Arm,
    /// Still breaching, still inside the grace period.
    Wait,
    /// Breaching continuously for longer than the grace period.
    Kill,
}

/// The byte ceiling a dynamic service is held to: its declared `max_mb` plus
/// the tolerated headroom.
pub(crate) fn ceiling_bytes(max_mb: u64) -> u64 {
    max_mb * 1024 * 1024 * OVER_CEILING_PERMILLE / 1000
}

/// Decide the watchdog's action from the current reading and how long the
/// breach has been running (`None` when the timer is disarmed).
pub(crate) fn ceiling_action(
    observed: u64,
    ceiling: u64,
    breaching_for: Option<Duration>,
) -> CeilingAction {
    if observed <= ceiling {
        return CeilingAction::Disarm;
    }
    match breaching_for {
        None => CeilingAction::Arm,
        Some(elapsed) if elapsed > OVER_CEILING_GRACE => CeilingAction::Kill,
        Some(_) => CeilingAction::Wait,
    }
}

/// Headroom above `max_mb` tolerated before `OVER_CEILING_GRACE` applies, as
/// permille (1100 ‰ = +10 %, i.e. 1.10 ×).
const OVER_CEILING_PERMILLE: u64 = 1100;

#[cfg(test)]
mod tests {
    use super::*;

    fn mb(n: u64) -> u64 {
        n * 1024 * 1024
    }

    #[test]
    fn ceiling_includes_ten_percent_headroom() {
        assert_eq!(ceiling_bytes(1000), mb(1100));
    }

    #[test]
    fn inside_the_ceiling_disarms() {
        let c = ceiling_bytes(8 * 1024);
        assert_eq!(ceiling_action(mb(8 * 1024), c, None), CeilingAction::Disarm);
        assert_eq!(
            ceiling_action(mb(8 * 1024), c, Some(Duration::from_secs(120))),
            CeilingAction::Disarm,
            "a sample back inside the ceiling disarms however long the \
             breach had been running"
        );
    }

    /// Exactly at the tolerated ceiling is not a breach; the daemon has
    /// always used a strict comparison and services are configured against
    /// it.
    #[test]
    fn exactly_at_the_ceiling_is_not_a_breach() {
        let c = ceiling_bytes(8 * 1024);
        assert_eq!(ceiling_action(c, c, None), CeilingAction::Disarm);
        assert_eq!(ceiling_action(c + 1, c, None), CeilingAction::Arm);
    }

    #[test]
    fn breach_arms_then_waits_then_kills() {
        let c = ceiling_bytes(8 * 1024);
        let over = mb(12 * 1024);
        assert_eq!(ceiling_action(over, c, None), CeilingAction::Arm);
        assert_eq!(
            ceiling_action(over, c, Some(Duration::from_secs(10))),
            CeilingAction::Wait
        );
        assert_eq!(
            ceiling_action(over, c, Some(OVER_CEILING_GRACE)),
            CeilingAction::Wait,
            "the grace period is exclusive — the kill lands on the tick after"
        );
        assert_eq!(
            ceiling_action(over, c, Some(OVER_CEILING_GRACE + Duration::from_secs(1))),
            CeilingAction::Kill
        );
    }

    /// The transient-spike case, stated at the level of the decision
    /// function: a breach that subsides re-arms from zero rather than
    /// resuming the earlier timer.
    #[test]
    fn a_subsided_breach_restarts_the_grace_period() {
        let c = ceiling_bytes(8 * 1024);
        assert_eq!(
            ceiling_action(mb(12 * 1024), c, Some(Duration::from_secs(29))),
            CeilingAction::Wait
        );
        assert_eq!(ceiling_action(mb(3 * 1024), c, None), CeilingAction::Disarm);
        assert_eq!(ceiling_action(mb(12 * 1024), c, None), CeilingAction::Arm);
    }
}
