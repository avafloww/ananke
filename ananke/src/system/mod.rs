//! System-boundary abstractions.
//!
//! Re-exported from `ananke-system` so `crate::system::…` paths inside
//! the daemon are unchanged by the split.

pub use ananke_system::{
    DynAsyncRead, Fs, InMemoryFs, LocalFs, LocalProcFs, LocalSpawner, ManagedChild, Meminfo,
    ProcFs, ProcessSpawner, Rss, SeekRead, SystemDeps, proc,
};
#[cfg(any(test, feature = "test-fakes"))]
pub use ananke_system::{
    FakeBag, FakeChildSnapshot, FakeProcessState, FakeSpawner, InMemoryProcFs,
};
