//! What a device looks like to the packer.
//!
//! These are the placement crate's vocabulary rather than the daemon's: the
//! packer is a pure function over a snapshot, and the NVML probe that fills one
//! in lives in `ananke_devices`, which re-exports them.

use std::collections::BTreeMap;

use ananke_config::placement::DeviceSlot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceId {
    Cpu,
    Gpu(u32),
}

impl DeviceId {
    pub fn to_slot(self) -> DeviceSlot {
        match self {
            DeviceId::Cpu => DeviceSlot::Cpu,
            DeviceId::Gpu(n) => DeviceSlot::Gpu(n),
        }
    }

    pub fn from_slot(slot: &DeviceSlot) -> Self {
        match slot {
            DeviceSlot::Cpu => DeviceId::Cpu,
            DeviceSlot::Gpu(n) => DeviceId::Gpu(*n),
        }
    }

    pub fn as_display(self) -> String {
        match self {
            DeviceId::Cpu => "cpu".into(),
            DeviceId::Gpu(n) => format!("gpu:{n}"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Allocation {
    pub bytes: BTreeMap<DeviceId, u64>,
}

impl Allocation {
    pub fn from_override(map: &BTreeMap<DeviceSlot, u64>) -> Self {
        let mut bytes = BTreeMap::new();
        for (slot, b) in map {
            bytes.insert(DeviceId::from_slot(slot), b * 1024 * 1024); // MB → bytes
        }
        Self { bytes }
    }

    pub fn gpu_ids(&self) -> Vec<u32> {
        self.bytes
            .keys()
            .filter_map(|d| {
                if let DeviceId::Gpu(n) = d {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn total(&self) -> u64 {
        self.bytes.values().sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceSnapshot {
    pub gpus: Vec<GpuSnapshot>,
    pub cpu: Option<CpuSnapshot>,
    pub taken_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSnapshot {
    pub id: u32,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

impl DeviceSnapshot {
    pub fn free_bytes(&self, slot: &DeviceSlot) -> Option<u64> {
        match slot {
            DeviceSlot::Cpu => self.cpu.as_ref().map(|c| c.available_bytes),
            DeviceSlot::Gpu(id) => self.gpus.iter().find(|g| g.id == *id).map(|g| g.free_bytes),
        }
    }

    pub fn total_bytes(&self, slot: &DeviceSlot) -> Option<u64> {
        match slot {
            DeviceSlot::Cpu => self.cpu.as_ref().map(|c| c.total_bytes),
            DeviceSlot::Gpu(id) => self
                .gpus
                .iter()
                .find(|g| g.id == *id)
                .map(|g| g.total_bytes),
        }
    }
}
