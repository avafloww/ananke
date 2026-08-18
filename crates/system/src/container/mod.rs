//! Container-runtime boundary: typed two-phase create/start seam with Docker
//! and Podman CLI adapters, plus an in-memory fake for deterministic tests.

mod cli;
#[cfg(any(test, feature = "test-fakes"))]
mod fake;
mod render;
mod types;

pub use cli::{CliContainerEngine, DockerCli, PodmanCli};
#[cfg(any(test, feature = "test-fakes"))]
pub use fake::{FakeContainerEngine, FakeContainerSnapshot, FakeContainerState};
pub use render::{executable_for, render_create_argv};
pub use types::{
    ContainerEngine, ContainerInspect, ContainerSummary, ManagedContainer, PreparedContainer,
};
