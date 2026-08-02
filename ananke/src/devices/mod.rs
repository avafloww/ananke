//! Device types and allocation primitives.

pub mod cpu;
pub mod cuda_env;
pub mod fake;
pub mod nvml;
pub mod probe;
pub mod snapshotter;

pub use probe::{GpuInfo, GpuMemory, GpuProbe, GpuProcess};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub id: DeviceId,
    pub total_bytes: u64,
}

pub use ananke_placement::devices::{
    Allocation, CpuSnapshot, DeviceId, DeviceSnapshot, GpuSnapshot,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::config::validate::DeviceSlot;

    #[test]
    fn allocation_from_override_converts_mb_to_bytes() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 1024);
        m.insert(DeviceSlot::Cpu, 2048);
        let a = Allocation::from_override(&m);
        assert_eq!(a.bytes[&DeviceId::Gpu(0)], 1024 * 1024 * 1024);
        assert_eq!(a.bytes[&DeviceId::Cpu], 2048 * 1024 * 1024);
    }

    #[test]
    fn gpu_ids_filters_cpu() {
        let mut m = BTreeMap::new();
        m.insert(DeviceSlot::Gpu(0), 10);
        m.insert(DeviceSlot::Gpu(1), 20);
        m.insert(DeviceSlot::Cpu, 30);
        let a = Allocation::from_override(&m);
        let mut ids = a.gpu_ids();
        ids.sort();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::Cpu.as_display(), "cpu");
        assert_eq!(DeviceId::Gpu(3).as_display(), "gpu:3");
    }
}
