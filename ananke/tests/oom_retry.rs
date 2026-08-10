//! Integration test: `bump_for_oom_retry` increases the rolling means so the
//! next spawn attempt reserves more memory.
#![cfg(feature = "test-fakes")]

use ananke_tracking::rolling::RollingTable;
use smol_str::SmolStr;

/// An OOM kill nudges both pools. The trigger is a SIGKILL shortly after spawn
/// — the kernel OOM killer or a cgroup limit, so a host verdict — but the
/// signal doesn't say which pool the daemon mis-modelled to get there, and an
/// under-reserved GPU side spills to the host and surfaces as exactly this
/// kill.
#[test]
fn oom_bump_increases_both_rolling_means() {
    let t = RollingTable::new();
    let svc = SmolStr::new("demo");
    let before = t.get(&svc);
    t.bump_for_oom_retry(&svc);
    let after = t.get(&svc);
    assert!(
        after.vram.mean > before.vram.mean,
        "vram mean after OOM bump ({}) must exceed initial value ({})",
        after.vram.mean,
        before.vram.mean
    );
    assert!(
        after.host.mean > before.host.mean,
        "host mean after OOM bump ({}) must exceed initial value ({})",
        after.host.mean,
        before.host.mean
    );
}
