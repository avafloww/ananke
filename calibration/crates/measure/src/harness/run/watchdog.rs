//! Stop a run before the box starts thrashing.
//!
//! A heavily expert-offloaded cell is measured with a margin of a gigabyte or
//! two, so overcommitting is a real possibility rather than a formality — and a
//! hybrid that overcommits does not fail cleanly, it pages. The measurement is
//! worthless either way once that starts, and the alternative to stopping is a
//! machine that stops responding. This tripped twice on GLM-5.2 during the
//! campaign, which is the whole argument for keeping it.
//!
//! The trip is reported and the caller stops the child it spawned, by pid.
//! Reaching for `pkill -f llama-server` instead would also match the shell driving
//! the campaign and any unrelated server the operator left running.

use crate::harness::sys::ProcFs;

#[derive(Debug)]
pub(crate) struct SwapWatchdog {
    limit_gib: f64,
    /// Swap already in use when the run started. Absolute swap says nothing —
    /// a box that has been up for weeks carries some — so the subject is growth.
    baseline_gib: f64,
    tripped: Option<f64>,
}

impl SwapWatchdog {
    pub(crate) fn start(procfs: &dyn ProcFs, limit_gib: f64) -> Self {
        Self {
            limit_gib,
            baseline_gib: procfs.swap_used_gib(),
            tripped: None,
        }
    }

    /// Read swap again, returning the growth once it has passed the limit.
    ///
    /// Latching: once tripped it stays tripped, because the caller's response is
    /// to stop the server, after which swap falls back and a fresh reading would
    /// say everything is fine.
    pub(crate) fn check(&mut self, procfs: &dyn ProcFs) -> Option<f64> {
        if self.tripped.is_some() {
            return self.tripped;
        }
        let grown = procfs.swap_used_gib() - self.baseline_gib;
        if grown > self.limit_gib {
            self.tripped = Some(grown);
        }
        self.tripped
    }

    pub(crate) fn tripped(&self) -> Option<f64> {
        self.tripped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sys::FakeProcFs;

    #[test]
    fn growth_past_the_limit_trips_and_stays_tripped() {
        // A GiB of swap on each read, against a two-GiB limit: the third check
        // is the one that crosses it.
        let procfs = FakeProcFs::new().with_swap_growth_gib(1.0);
        let mut watchdog = SwapWatchdog::start(&procfs, 2.0);
        assert_eq!(
            watchdog.baseline_gib, 1.0,
            "the baseline is read at the start"
        );
        assert_eq!(watchdog.check(&procfs), None);
        assert_eq!(watchdog.check(&procfs), None);
        assert_eq!(watchdog.check(&procfs), Some(3.0));
        // Latched: swap falling back after the kill must not un-trip it.
        assert_eq!(watchdog.tripped(), Some(3.0));
    }

    #[test]
    fn a_box_with_swap_already_in_use_does_not_trip_on_that_alone() {
        let procfs = FakeProcFs::new().with_swap_growth_gib(0.0);
        let mut watchdog = SwapWatchdog::start(&procfs, 4.0);
        for _ in 0..10 {
            assert_eq!(watchdog.check(&procfs), None);
        }
    }
}
