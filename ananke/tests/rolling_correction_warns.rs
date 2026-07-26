//! Integration test: feeding repeated under-estimate samples to a `RollingTable`
//! converges that pool's rolling mean above the 1.2 warning threshold.
#![cfg(feature = "test-fakes")]

use ananke::tracking::rolling::{MemoryClass, RollingTable};
use smol_str::SmolStr;

#[test]
fn rolling_mean_converges_above_threshold_warns() {
    for class in [MemoryClass::Vram, MemoryClass::Host] {
        converges(class);
    }
}

fn converges(class: MemoryClass) {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    // Observed peak is 130, base estimate is 100 → ratio = 1.3.
    // After three samples the running mean should exceed 1.2.
    for _ in 0..3 {
        t.update(&svc, class, 130, 100);
    }
    let c = *t.get(&svc).class(class);
    assert!(
        c.mean > 1.2,
        "{} mean ({}) must exceed 1.2 after three 1.3× samples",
        class.as_str(),
        c.mean
    );
}
