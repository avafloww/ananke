//! RAII in-flight request guard.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// RAII guard that increments the counter on construction and decrements on drop.
pub struct InflightGuard {
    counter: Arc<AtomicU64>,
}

impl InflightGuard {
    pub fn new(counter: Arc<AtomicU64>) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_increments_and_decrements() {
        let c = Arc::new(AtomicU64::new(0));
        {
            let _g = InflightGuard::new(c.clone());
            assert_eq!(c.load(Ordering::Relaxed), 1);
        }
        assert_eq!(c.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn multiple_guards_stack() {
        let c = Arc::new(AtomicU64::new(0));
        let _g1 = InflightGuard::new(c.clone());
        let _g2 = InflightGuard::new(c.clone());
        let _g3 = InflightGuard::new(c.clone());
        assert_eq!(c.load(Ordering::Relaxed), 3);
    }
}
