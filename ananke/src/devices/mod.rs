//! Device types and allocation primitives.
//!
//! Re-exported from `ananke-devices` (and `ananke-observation` for the
//! shared snapshot) so `crate::devices::…` paths inside the daemon are
//! unchanged by the split.

pub mod snapshotter;

pub use ananke_devices::{
    Device, GpuInfo, GpuMemory, GpuProbe, GpuProcess, cpu, cuda_env, fake, nvml, probe,
};
pub use ananke_placement::devices::{
    Allocation, CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot,
};
