//! Shared window-builder fixture for the balloon resolver's unit tests.

use std::collections::VecDeque;

pub(crate) fn mk_window(samples: &[u64]) -> VecDeque<u64> {
    VecDeque::from(samples.to_vec())
}
