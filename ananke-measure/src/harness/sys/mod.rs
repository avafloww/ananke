//! Every outside-world capability the harness reaches for, behind a trait.
//!
//! Not for portability — the harness is Linux-only and says so per file — but so
//! the suite stays deterministic. A test substitutes an in-memory implementation
//! and the phases below run with no process spawned, no driver queried, no
//! `/proc` read, and no wall-clock waited on.
//!
//! The bundle is the same shape as ananke's `SystemDeps`: [`Deps::local`] in
//! production, [`Fakes`] in tests, which hands back the concrete fakes so an
//! assertion can inspect what the harness did to them.

use std::sync::Arc;

#[cfg(any(test, feature = "test-fakes"))]
pub use crate::harness::sys::{
    clock::FakeClock, gpu::FakeGpu, http::FakeHttp, procfs::FakeProcFs, spawn::FakeSpawner,
};
pub use crate::harness::sys::{
    clock::{Clock, SystemClock},
    gpu::{GpuSampler, NvidiaSmi},
    http::{Http, LoopbackHttp},
    procfs::{LocalProcFs, ProcFs},
    spawn::{Child, LocalSpawner, SpawnRequest, Spawner, Stop},
};

mod clock;
mod gpu;
mod http;
mod procfs;
mod spawn;

#[derive(Clone)]
pub struct Deps {
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
    pub procfs: Arc<dyn ProcFs>,
    pub gpu: Arc<dyn GpuSampler>,
    pub http: Arc<dyn Http>,
}

impl Deps {
    pub fn local() -> Self {
        Self {
            clock: Arc::new(SystemClock::new()),
            spawner: Arc::new(LocalSpawner),
            procfs: Arc::new(LocalProcFs),
            gpu: Arc::new(NvidiaSmi),
            http: Arc::new(LoopbackHttp),
        }
    }
}

/// The fakes, held concretely so a test can both configure them beforehand and
/// interrogate them afterwards.
#[cfg(any(test, feature = "test-fakes"))]
pub struct Fakes {
    pub clock: Arc<FakeClock>,
    pub spawner: Arc<FakeSpawner>,
    pub procfs: Arc<FakeProcFs>,
    pub gpu: Arc<FakeGpu>,
    pub http: Arc<FakeHttp>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl Fakes {
    pub fn new(spawner: FakeSpawner, procfs: FakeProcFs, gpu: FakeGpu, http: FakeHttp) -> Self {
        Self {
            clock: Arc::new(FakeClock::new()),
            spawner: Arc::new(spawner),
            procfs: Arc::new(procfs),
            gpu: Arc::new(gpu),
            http: Arc::new(http),
        }
    }

    pub fn deps(&self) -> Deps {
        Deps {
            clock: self.clock.clone(),
            spawner: self.spawner.clone(),
            procfs: self.procfs.clone(),
            gpu: self.gpu.clone(),
            http: self.http.clone(),
        }
    }
}
