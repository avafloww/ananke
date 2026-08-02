//! Two readings that differ in exactly one flag.
//!
//! Several derivations measure a term as the difference between a cell with some
//! flag and its twin without it, accumulating both halves as the dataset presents
//! them. A `BTreeMap<bool, T>` would do that job and read as `pair[&true]` at the
//! far end: the indexing panics unless a length check the compiler cannot see has
//! run first, and nothing on the page says which boolean means which half.

/// Two readings that differ in a single flag, either half possibly unmeasured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pair<T> {
    pub with: Option<T>,
    pub without: Option<T>,
}

impl<T> Pair<T> {
    /// The half `flag` selects, to accumulate a reading into.
    pub fn half_mut(&mut self, flag: bool) -> &mut Option<T> {
        if flag {
            &mut self.with
        } else {
            &mut self.without
        }
    }

    /// Both halves, present only once the pair is complete.
    pub fn both(&self) -> Option<(&T, &T)> {
        Some((self.with.as_ref()?, self.without.as_ref()?))
    }
}

impl<T> Default for Pair<T> {
    fn default() -> Self {
        Self {
            with: None,
            without: None,
        }
    }
}
