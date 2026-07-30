//! Linux-only: shells out to `nvidia-smi` for per-process VRAM.
//!
//! Deliberately the *driver's* figure rather than llama.cpp's own breakdown
//! table. The breakdown attributes what llama.cpp allocated; the driver counts
//! the CUDA context and everything else besides, and the GPU compute-buffer
//! bases in ananke's estimator are defined against the driver's number. ik_llama
//! does not print the breakdown table at all, so for every ik cell this is the
//! only per-device source — summing the cards would leave the fork's placement
//! unmeasurable.
//!
//! `nvidia-smi` rather than NVML because the per-process query is one line of
//! CSV and this crate should not link a driver binding it would use once.

use std::collections::BTreeMap;

use crate::record::Gpu;

pub trait GpuSampler: Send + Sync {
    /// Per-process VRAM in MiB, split by card index. Empty when the driver
    /// reports nothing for the pid, which is also what a CPU-only cell looks
    /// like.
    fn per_process_mib(&self, pid: u32) -> BTreeMap<u32, u64>;
    /// The cards themselves, for the hardware block a constant is keyed on.
    fn devices(&self) -> Vec<Gpu>;
}

pub struct NvidiaSmi;

impl GpuSampler for NvidiaSmi {
    fn per_process_mib(&self, pid: u32) -> BTreeMap<u32, u64> {
        let apps = run(&[
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]);
        let index_of = index_by_uuid();
        let mut used = BTreeMap::new();
        for line in apps.lines() {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [reported, uuid, mib] = fields.as_slice() else {
                continue;
            };
            if reported.parse::<u32>() != Ok(pid) {
                continue;
            }
            // A card the index query did not name is skipped rather than
            // guessed: an index that is wrong attributes the memory to the wrong
            // device, which is worse than not attributing it.
            let (Some(index), Ok(mib)) = (index_of.get(*uuid), mib.parse::<u64>()) else {
                continue;
            };
            *used.entry(*index).or_default() += mib;
        }
        used
    }

    fn devices(&self) -> Vec<Gpu> {
        run(&[
            "--query-gpu=name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [name, total, capability, driver] = fields.as_slice() else {
                return None;
            };
            Some(Gpu {
                name: (*name).to_owned(),
                memory_total_mib: total.parse().unwrap_or_default(),
                compute_capability: (*capability).to_owned(),
                driver: (*driver).to_owned(),
            })
        })
        .collect()
    }
}

/// Map each card's UUID to its index; the per-process query reports only UUIDs.
fn index_by_uuid() -> BTreeMap<String, u32> {
    run(&["--query-gpu=index,uuid", "--format=csv,noheader"])
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [index, uuid] = fields.as_slice() else {
                return None;
            };
            index.parse().ok().map(|index| ((*uuid).to_owned(), index))
        })
        .collect()
}

/// A driver that is not there — no CUDA, or a wedged `nvidia-smi` — yields no
/// rows rather than an error: a CPU-only cell is a legitimate measurement.
fn run(arguments: &[&str]) -> String {
    std::process::Command::new("nvidia-smi")
        .args(arguments)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

/// A driver that reports whatever a test set, and can be made to report more on
/// each read so a peak has something to climb to.
#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
pub struct FakeGpu {
    inner: parking_lot::Mutex<FakeGpuState>,
}

#[cfg(any(test, feature = "test-fakes"))]
#[derive(Default)]
struct FakeGpuState {
    used_mib: BTreeMap<u32, u64>,
    devices: Vec<Gpu>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl FakeGpu {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_used_mib(self, used: &[(u32, u64)]) -> Self {
        self.inner.lock().used_mib = used.iter().copied().collect();
        self
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl GpuSampler for FakeGpu {
    fn per_process_mib(&self, _pid: u32) -> BTreeMap<u32, u64> {
        self.inner.lock().used_mib.clone()
    }

    fn devices(&self) -> Vec<Gpu> {
        self.inner.lock().devices.clone()
    }
}
