//! The resident-memory blocks: the per-cell summary, the time series, and the
//! per-turn checkpoints.
//!
//! All three mix **dynamic** keys — the driver's per-card readings are written
//! as `gpu{physical id}_used_mib` — in among fixed ones at the same nesting
//! level, hence the hand-written codecs: fixed keys stay named fields, card ids
//! are parsed out into a `BTreeMap<u32, u64>`, and anything else is an
//! unknown-field error. An open `BTreeMap<String, Value>` is where a
//! physical-versus-visible index mix-up hides.

use std::{collections::BTreeMap, fmt};

use serde::{
    Deserialize, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap,
};

/// The peak resident-memory summary for one cell, with the final reading and
/// the growth since startup alongside it. The `kb` figures are signed because
/// `growth_*` is a difference.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rss {
    /// `VmRSS`. Not what the host model compares against — see
    /// [`crate::record::Record::owned_bytes`].
    pub rss_total_kb: i64,
    pub rss_anon_kb: i64,
    /// `RssFile`: the mapped GGUF, which llama.cpp populates and then leaves
    /// resident as clean reclaimable pages.
    pub rss_file_kb: i64,
    /// `RssShmem`. `cudaMallocHost` is accounted here, not in anon.
    pub rss_shmem_kb: i64,
    /// The driver's total for the whole process.
    pub gpu_used_mib: Option<u64>,
    /// Keyed by *physical* GPU id, while the loader's breakdown rows are in
    /// visible order — see [`crate::record::Record::gpu_card_used_mib`].
    pub per_card: BTreeMap<u32, u64>,
    /// The same four counters at the end of the run.
    pub final_rss_total_kb: i64,
    pub final_rss_anon_kb: i64,
    pub final_rss_file_kb: i64,
    pub final_rss_shmem_kb: i64,
    /// The growth from load to steady state.
    pub growth_rss_total_kb: i64,
    pub growth_rss_anon_kb: i64,
    pub growth_rss_file_kb: i64,
    pub growth_rss_shmem_kb: i64,
    /// How many two-second samples the peak was taken over.
    pub samples: i64,
    /// How long the server took to answer `/health`.
    pub load_seconds: f64,
}

/// The `/proc/<pid>/status` resident-memory breakdown, flattened into
/// [`Sample`] and [`Checkpoint`] by their codecs rather than by
/// `serde(flatten)`, which cannot coexist with the dynamic GPU keys beside it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RssSnapshot {
    pub rss_total_kb: u64,
    pub rss_anon_kb: u64,
    pub rss_file_kb: u64,
    pub rss_shmem_kb: u64,
}

/// Per-process VRAM as the *driver* reports it, so it counts the CUDA context
/// and everything else llama.cpp's own breakdown cannot attribute. ik_llama
/// prints no breakdown table at all, so for every ik cell this is the only
/// per-device source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuUsage {
    pub total_mib: Option<u64>,
    /// Per card, keyed by physical id.
    pub used_mib: BTreeMap<u32, u64>,
}

/// One resident-memory sample, on the same two-second cadence ananke's
/// snapshotter uses — a single snapshot measures a different quantity than the
/// daemon does.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    pub t_seconds: f64,
    pub at_utc: String,
    pub rss: RssSnapshot,
    pub gpu: GpuUsage,
}

/// One turn's memory reading, against the tokens that produced it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Checkpoint {
    pub turn: u64,
    pub at_utc: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub generated_tokens_total: u64,
    /// The term that scales with context rather than with use.
    pub kv_depth_tokens: u64,
    pub rss: RssSnapshot,
    pub gpu: GpuUsage,
    /// Which of the alternating conversations this turn belongs to; absent
    /// outside a growth run.
    pub conversation: Option<u32>,
}

impl Serialize for Rss {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("rss_total_kb", &self.rss_total_kb)?;
        map.serialize_entry("rss_anon_kb", &self.rss_anon_kb)?;
        map.serialize_entry("rss_file_kb", &self.rss_file_kb)?;
        map.serialize_entry("rss_shmem_kb", &self.rss_shmem_kb)?;
        if let Some(total) = self.gpu_used_mib {
            map.serialize_entry(GPU_TOTAL_KEY, &total)?;
        }
        serialize_per_card(&mut map, &self.per_card)?;
        map.serialize_entry("final_rss_total_kb", &self.final_rss_total_kb)?;
        map.serialize_entry("final_rss_anon_kb", &self.final_rss_anon_kb)?;
        map.serialize_entry("final_rss_file_kb", &self.final_rss_file_kb)?;
        map.serialize_entry("final_rss_shmem_kb", &self.final_rss_shmem_kb)?;
        map.serialize_entry("growth_rss_total_kb", &self.growth_rss_total_kb)?;
        map.serialize_entry("growth_rss_anon_kb", &self.growth_rss_anon_kb)?;
        map.serialize_entry("growth_rss_file_kb", &self.growth_rss_file_kb)?;
        map.serialize_entry("growth_rss_shmem_kb", &self.growth_rss_shmem_kb)?;
        map.serialize_entry("samples", &self.samples)?;
        map.serialize_entry("load_seconds", &self.load_seconds)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Rss {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RssVisitor)
    }
}

impl Serialize for Sample {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("t_seconds", &self.t_seconds)?;
        map.serialize_entry("at_utc", &self.at_utc)?;
        serialize_snapshot(&mut map, &self.rss)?;
        serialize_gpu(&mut map, &self.gpu)?;
        map.end()
    }
}

impl<'de> Deserialize<'de> for Sample {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(SampleVisitor)
    }
}

impl Serialize for Checkpoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("turn", &self.turn)?;
        map.serialize_entry("at_utc", &self.at_utc)?;
        map.serialize_entry("prompt_tokens", &self.prompt_tokens)?;
        map.serialize_entry("completion_tokens", &self.completion_tokens)?;
        map.serialize_entry("generated_tokens_total", &self.generated_tokens_total)?;
        map.serialize_entry("kv_depth_tokens", &self.kv_depth_tokens)?;
        serialize_snapshot(&mut map, &self.rss)?;
        serialize_gpu(&mut map, &self.gpu)?;
        if let Some(conversation) = self.conversation {
            map.serialize_entry("conversation", &conversation)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Checkpoint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(CheckpointVisitor)
    }
}

/// The process total, whose "id" is empty — which is exactly why the per-card
/// scan below has to run after this key is matched, not before.
const GPU_TOTAL_KEY: &str = "gpu_used_mib";

const GPU_PREFIX: &str = "gpu";
const GPU_SUFFIX: &str = "_used_mib";

/// The fixed keys, named for the sake of serde's unknown-field message; the
/// dynamic `gpu{id}_used_mib` family cannot be listed and is described in the
/// module docs instead.
const RSS_FIELDS: &[&str] = &[
    "rss_total_kb",
    "rss_anon_kb",
    "rss_file_kb",
    "rss_shmem_kb",
    GPU_TOTAL_KEY,
    "final_rss_total_kb",
    "final_rss_anon_kb",
    "final_rss_file_kb",
    "final_rss_shmem_kb",
    "growth_rss_total_kb",
    "growth_rss_anon_kb",
    "growth_rss_file_kb",
    "growth_rss_shmem_kb",
    "samples",
    "load_seconds",
];

const SAMPLE_FIELDS: &[&str] = &[
    "t_seconds",
    "at_utc",
    "rss_total_kb",
    "rss_anon_kb",
    "rss_file_kb",
    "rss_shmem_kb",
    GPU_TOTAL_KEY,
];

const CHECKPOINT_FIELDS: &[&str] = &[
    "turn",
    "at_utc",
    "prompt_tokens",
    "completion_tokens",
    "generated_tokens_total",
    "kv_depth_tokens",
    "rss_total_kb",
    "rss_anon_kb",
    "rss_file_kb",
    "rss_shmem_kb",
    GPU_TOTAL_KEY,
    "conversation",
];

struct RssVisitor;

impl<'de> Visitor<'de> for RssVisitor {
    type Value = Rss;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a resident-memory summary")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Rss, A::Error> {
        let mut rss = Rss::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "rss_total_kb" => rss.rss_total_kb = map.next_value()?,
                "rss_anon_kb" => rss.rss_anon_kb = map.next_value()?,
                "rss_file_kb" => rss.rss_file_kb = map.next_value()?,
                "rss_shmem_kb" => rss.rss_shmem_kb = map.next_value()?,
                GPU_TOTAL_KEY => rss.gpu_used_mib = Some(map.next_value()?),
                "final_rss_total_kb" => rss.final_rss_total_kb = map.next_value()?,
                "final_rss_anon_kb" => rss.final_rss_anon_kb = map.next_value()?,
                "final_rss_file_kb" => rss.final_rss_file_kb = map.next_value()?,
                "final_rss_shmem_kb" => rss.final_rss_shmem_kb = map.next_value()?,
                "growth_rss_total_kb" => rss.growth_rss_total_kb = map.next_value()?,
                "growth_rss_anon_kb" => rss.growth_rss_anon_kb = map.next_value()?,
                "growth_rss_file_kb" => rss.growth_rss_file_kb = map.next_value()?,
                "growth_rss_shmem_kb" => rss.growth_rss_shmem_kb = map.next_value()?,
                "samples" => rss.samples = map.next_value()?,
                "load_seconds" => rss.load_seconds = map.next_value()?,
                other => match card_id(other) {
                    Some(id) => {
                        rss.per_card.insert(id, map.next_value()?);
                    }
                    None => return Err(de::Error::unknown_field(other, RSS_FIELDS)),
                },
            }
        }
        Ok(rss)
    }
}

struct SampleVisitor;

impl<'de> Visitor<'de> for SampleVisitor {
    type Value = Sample;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a resident-memory sample")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Sample, A::Error> {
        let mut sample = Sample::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "t_seconds" => sample.t_seconds = map.next_value()?,
                "at_utc" => sample.at_utc = map.next_value()?,
                _ => visit_shared(
                    &mut map,
                    &key,
                    &mut sample.rss,
                    &mut sample.gpu,
                    SAMPLE_FIELDS,
                )?,
            }
        }
        Ok(sample)
    }
}

struct CheckpointVisitor;

impl<'de> Visitor<'de> for CheckpointVisitor {
    type Value = Checkpoint;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a turn checkpoint")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Checkpoint, A::Error> {
        let mut checkpoint = Checkpoint::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "turn" => checkpoint.turn = map.next_value()?,
                "at_utc" => checkpoint.at_utc = map.next_value()?,
                "prompt_tokens" => checkpoint.prompt_tokens = map.next_value()?,
                "completion_tokens" => checkpoint.completion_tokens = map.next_value()?,
                "generated_tokens_total" => {
                    checkpoint.generated_tokens_total = map.next_value()?;
                }
                "kv_depth_tokens" => checkpoint.kv_depth_tokens = map.next_value()?,
                "conversation" => checkpoint.conversation = Some(map.next_value()?),
                _ => visit_shared(
                    &mut map,
                    &key,
                    &mut checkpoint.rss,
                    &mut checkpoint.gpu,
                    CHECKPOINT_FIELDS,
                )?,
            }
        }
        Ok(checkpoint)
    }
}

/// The `/proc` counters and the GPU readings, which [`Sample`] and
/// [`Checkpoint`] spell identically.
fn visit_shared<'de, A: MapAccess<'de>>(
    map: &mut A,
    key: &str,
    rss: &mut RssSnapshot,
    gpu: &mut GpuUsage,
    fields: &'static [&'static str],
) -> Result<(), A::Error> {
    match key {
        "rss_total_kb" => rss.rss_total_kb = map.next_value()?,
        "rss_anon_kb" => rss.rss_anon_kb = map.next_value()?,
        "rss_file_kb" => rss.rss_file_kb = map.next_value()?,
        "rss_shmem_kb" => rss.rss_shmem_kb = map.next_value()?,
        GPU_TOTAL_KEY => gpu.total_mib = Some(map.next_value()?),
        other => match card_id(other) {
            Some(id) => {
                gpu.used_mib.insert(id, map.next_value()?);
            }
            None => return Err(de::Error::unknown_field(other, fields)),
        },
    }
    Ok(())
}

/// The physical card id inside a `gpu{id}_used_mib` key, if the key is one.
fn card_id(key: &str) -> Option<u32> {
    key.strip_prefix(GPU_PREFIX)?
        .strip_suffix(GPU_SUFFIX)?
        .parse()
        .ok()
}

fn serialize_snapshot<S: SerializeMap>(map: &mut S, rss: &RssSnapshot) -> Result<(), S::Error> {
    map.serialize_entry("rss_total_kb", &rss.rss_total_kb)?;
    map.serialize_entry("rss_anon_kb", &rss.rss_anon_kb)?;
    map.serialize_entry("rss_file_kb", &rss.rss_file_kb)?;
    map.serialize_entry("rss_shmem_kb", &rss.rss_shmem_kb)
}

fn serialize_gpu<S: SerializeMap>(map: &mut S, gpu: &GpuUsage) -> Result<(), S::Error> {
    if let Some(total) = gpu.total_mib {
        map.serialize_entry(GPU_TOTAL_KEY, &total)?;
    }
    serialize_per_card(map, &gpu.used_mib)
}

fn serialize_per_card<S: SerializeMap>(
    map: &mut S,
    per_card: &BTreeMap<u32, u64>,
) -> Result<(), S::Error> {
    for (id, mib) in per_card {
        map.serialize_entry(&format!("{GPU_PREFIX}{id}{GPU_SUFFIX}"), mib)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::to_dataset_json;

    #[test]
    fn a_cards_id_survives_the_trip_through_its_key() {
        let line = r#"{"t_seconds": 2.0, "at_utc": "z", "rss_total_kb": 1, "rss_anon_kb": 2, "rss_file_kb": 3, "rss_shmem_kb": 4, "gpu_used_mib": 90, "gpu0_used_mib": 40, "gpu3_used_mib": 50}"#;
        let sample: Sample = serde_json::from_str(line).expect("a sample parses");
        // The total is not a card, and card three is not card one.
        assert_eq!(sample.gpu.total_mib, Some(90));
        assert_eq!(sample.gpu.used_mib, BTreeMap::from([(0, 40), (3, 50)]));
        assert_eq!(to_dataset_json(&sample), line);
    }

    #[test]
    fn an_unrecognised_key_is_an_error_rather_than_a_card() {
        let line = r#"{"t_seconds": 2.0, "gpu_temperature_c": 71}"#;
        let error = serde_json::from_str::<Sample>(line).expect_err("the key is not a card");
        assert!(error.to_string().contains("gpu_temperature_c"), "{error}");
    }
}
