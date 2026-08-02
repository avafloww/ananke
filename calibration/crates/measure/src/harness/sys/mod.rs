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
    clock::FakeClock, files::FakeFiles, gpu::FakeGpu, http::FakeHttp, procfs::FakeProcFs,
    spawn::FakeSpawner,
};
pub use crate::harness::sys::{
    clock::{Clock, SystemClock},
    files::{Files, LocalFiles},
    gpu::{GpuSampler, NvidiaSmi},
    http::{Http, LoopbackHttp, post_json},
    procfs::{LocalProcFs, ProcFs},
    spawn::{Child, LocalSpawner, SpawnRequest, Spawner, Stop},
};

mod clock;
mod files;
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
    pub files: Arc<dyn Files>,
}

impl Deps {
    pub fn local() -> Self {
        Self {
            clock: Arc::new(SystemClock::new()),
            spawner: Arc::new(LocalSpawner),
            procfs: Arc::new(LocalProcFs),
            gpu: Arc::new(NvidiaSmi),
            http: Arc::new(LoopbackHttp),
            files: Arc::new(LocalFiles),
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
    pub files: Arc<FakeFiles>,
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
            files: Arc::new(FakeFiles::new()),
        }
    }

    /// Start with a dataset, or any other file, already in place.
    pub fn with_files(mut self, files: FakeFiles) -> Self {
        self.files = Arc::new(files);
        self
    }

    pub fn deps(&self) -> Deps {
        Deps {
            clock: self.clock.clone(),
            spawner: self.spawner.clone(),
            procfs: self.procfs.clone(),
            gpu: self.gpu.clone(),
            http: self.http.clone(),
            files: self.files.clone(),
        }
    }
}
