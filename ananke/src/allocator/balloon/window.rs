//! The resolver's rolling sample window, tagged with the device slot its
//! samples were read from.

use std::collections::VecDeque;

use crate::{allocator::balloon::WINDOW_SIZE, config::DeviceSlot};

/// A bounded window of the most recent memory samples for one service.
///
/// The samples are *current* readings, not high-water marks, so `max()` over
/// the window is a recent peak that decays as the spike rolls out — which is
/// what the pledge wants — and the sequence can trend downwards, which is what
/// growth detection wants.
///
/// The slot tag is load-bearing. The resolver reads VRAM for a GPU-pledged
/// service and host RSS for a cpu-pinned one, and a service can land on a
/// different slot from one run to the next. Mixing the two series in one
/// window would have the pledge and the growth trend both reading a
/// discontinuity as a real change in usage, so the window resets whenever the
/// slot changes — including across the `None` a service holds while it is
/// idle, draining, or between runs.
pub(crate) struct SampleWindow {
    slot: Option<DeviceSlot>,
    samples: VecDeque<u64>,
}

impl SampleWindow {
    pub(crate) fn new() -> Self {
        Self {
            slot: None,
            samples: VecDeque::with_capacity(WINDOW_SIZE),
        }
    }

    /// Record `sample`, taken from `slot`. Returns `true` when the slot
    /// differed from the previous sample's and the window was therefore
    /// reset, so the caller can reset the state it derives from the window
    /// too (the over-ceiling grace timer).
    pub(crate) fn push(&mut self, slot: Option<DeviceSlot>, sample: u64) -> bool {
        let reset = slot != self.slot;
        if reset {
            self.slot = slot;
            self.samples.clear();
        }
        if self.samples.len() == WINDOW_SIZE {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        reset
    }

    pub(crate) fn samples(&self) -> &VecDeque<u64> {
        &self.samples
    }

    /// Drop every retained sample, keeping the slot tag. Used after the
    /// resolver acts on the window (a fast-kill), so the action isn't
    /// immediately re-derived from the samples that triggered it.
    pub(crate) fn clear(&mut self) {
        self.samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu0() -> Option<DeviceSlot> {
        Some(DeviceSlot::Gpu(0))
    }

    #[test]
    fn samples_accumulate_and_are_bounded() {
        let mut w = SampleWindow::new();
        for i in 0..(WINDOW_SIZE as u64 + 3) {
            w.push(gpu0(), i);
        }
        assert_eq!(w.samples().len(), WINDOW_SIZE);
        // The three oldest samples rolled out of the front.
        assert_eq!(*w.samples().front().unwrap(), 3);
        assert_eq!(*w.samples().back().unwrap(), WINDOW_SIZE as u64 + 2);
    }

    /// The first push always "resets" (from the initial `None` tag) but has
    /// nothing to discard; what matters is that the sample survives it.
    #[test]
    fn first_push_retains_its_sample() {
        let mut w = SampleWindow::new();
        assert!(w.push(gpu0(), 42));
        assert_eq!(w.samples().iter().copied().collect::<Vec<_>>(), vec![42]);
    }

    /// A service that lands on the CPU after a run on a GPU must not have its
    /// RSS series fitted against the previous run's VRAM samples.
    #[test]
    fn slot_change_discards_the_previous_series() {
        let mut w = SampleWindow::new();
        w.push(gpu0(), 10);
        w.push(gpu0(), 11);
        assert!(w.push(Some(DeviceSlot::Cpu), 900));
        assert_eq!(w.samples().iter().copied().collect::<Vec<_>>(), vec![900]);
    }

    /// Draining takes the row away entirely, so the tag passes through
    /// `None`. That, too, has to break the series — the next run's samples
    /// are a fresh process.
    #[test]
    fn drain_and_respawn_breaks_the_series() {
        let mut w = SampleWindow::new();
        w.push(gpu0(), 10);
        w.push(gpu0(), 20);
        assert!(w.push(None, 0), "losing the row resets the window");
        assert!(w.push(gpu0(), 5), "regaining it resets again");
        assert_eq!(w.samples().iter().copied().collect::<Vec<_>>(), vec![5]);
    }

    #[test]
    fn steady_slot_does_not_reset() {
        let mut w = SampleWindow::new();
        w.push(gpu0(), 10);
        assert!(!w.push(gpu0(), 20));
        assert_eq!(
            w.samples().iter().copied().collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn clear_keeps_the_slot_tag() {
        let mut w = SampleWindow::new();
        w.push(gpu0(), 10);
        w.clear();
        assert!(w.samples().is_empty());
        assert!(
            !w.push(gpu0(), 20),
            "clearing must not make the next same-slot push look like a change"
        );
    }
}
