//! The four buffer-size figures the loaders print, each with its history.

use std::ops::Index;

use regex::Regex;
use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::parse::{patterns, repeats_key};

/// One buffer figure a load log reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BufferKind {
    /// The graph allocator's buffer, pinned (`CUDA_Host`) whenever a GPU is
    /// present and plain `CPU` otherwise.
    Arena,
    OutBuf,
    CpuKv,
    CpuModel,
}

impl BufferKind {
    pub(crate) const ALL: [BufferKind; 4] = [
        BufferKind::Arena,
        BufferKind::OutBuf,
        BufferKind::CpuKv,
        BufferKind::CpuModel,
    ];

    /// The record key this figure is written under.
    pub fn key(self) -> &'static str {
        match self {
            BufferKind::Arena => "arena_mib",
            BufferKind::OutBuf => "out_buf_mib",
            BufferKind::CpuKv => "cpu_kv_mib",
            BufferKind::CpuModel => "cpu_model_mib",
        }
    }

    fn pattern(self) -> &'static Regex {
        match self {
            BufferKind::Arena => &patterns::ARENA,
            BufferKind::OutBuf => &patterns::OUT_BUF,
            BufferKind::CpuKv => &patterns::CPU_KV,
            BufferKind::CpuModel => &patterns::CPU_MODEL,
        }
    }
}

/// One figure's value, plus every occurrence of it when there was more than
/// one.
///
/// A cell with `-md` or `--mmproj` loads two models, so a figure can appear
/// more than once with genuinely different values. Keeping every occurrence
/// lets a later reader tell the target's figure from the draft's rather than
/// inheriting whichever this pass happened to pick.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Series {
    /// The last occurrence, which is the figure to fit against: the loader logs
    /// a reserve pass first and then the real graph, with the same value.
    pub last: f64,
    /// Every occurrence, in log order, recorded only when there was more than
    /// one.
    pub all: Option<Vec<f64>>,
}

/// Every buffer figure in one log.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BufferSizes {
    values: [Series; 4],
}

impl BufferSizes {
    pub(crate) fn parse(text: &str) -> Self {
        let values = BufferKind::ALL.map(|kind| {
            let found: Vec<f64> = kind
                .pattern()
                .captures_iter(text)
                .filter_map(|caps| caps.get(1)?.as_str().parse().ok())
                .collect();
            Series {
                last: found.last().copied().unwrap_or(0.0),
                all: (found.len() > 1).then_some(found),
            }
        });
        Self { values }
    }
}

impl Index<BufferKind> for BufferSizes {
    type Output = Series;

    fn index(&self, kind: BufferKind) -> &Series {
        &self.values[kind as usize]
    }
}

impl Serialize for BufferSizes {
    /// Flattened into the record, with the repeat list under `<key>_all`.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for kind in BufferKind::ALL {
            let series = &self[kind];
            map.serialize_entry(kind.key(), &series.last)?;
            if let Some(all) = &series.all {
                map.serialize_entry(&repeats_key(kind.key()), all)?;
            }
        }
        map.end()
    }
}
