//! An insertion-ordered map, because two evidence strings depend on one.
//!
//! Two derivers list their cells in the order the dataset presents them rather
//! than sorted — the per-device host cost and the MTP unaccounted residual. A
//! `BTreeMap` there would reorder the evidence text and fail `emit --check` while
//! every value matched, which is the most confusing possible failure. Everything
//! whose order is genuinely sorted uses `BTreeMap` instead.
//!
//! Linear lookup: the dataset is under a thousand rows and the groups are a
//! handful, so a hash is not worth the dependency.

/// A map that iterates in insertion order.
#[derive(Debug, Clone)]
pub struct OrderedMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K: PartialEq, V> OrderedMap<K, V> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The value for `key`, inserting `default()` first if it is absent.
    pub fn or_insert_with(&mut self, key: K, default: impl FnOnce() -> V) -> &mut V {
        match self.entries.iter().position(|(k, _)| *k == key) {
            Some(index) => &mut self.entries[index].1,
            None => {
                self.entries.push((key, default()));
                let last = self.entries.len() - 1;
                &mut self.entries[last].1
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<K: PartialEq, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
