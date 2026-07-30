//! Turn a captured llama-server log into the shape and the buffer sizes the
//! calibration campaign is fitted against.
//!
//! Nothing here fails: a figure the log does not carry is recorded as its zero
//! rather than as an error, because a run that logged less than expected is
//! still a measurement and the record has to say what was seen.

use std::collections::BTreeMap;

use serde::Serialize;

pub use crate::parse::{
    breakdown::{DeviceRow, GpuMirrors, HostBreakdown, MAX_GPUS},
    buffers::{BufferKind, BufferSizes, Series},
    contexts::{BufferRole, Context, KvPool, RsPool},
    meta::{MetaFields, MetaKey, MetaValue},
};

mod breakdown;
mod buffers;
mod contexts;
mod meta;
mod patterns;

/// The model's shape and its logged buffer sizes, as one record's `parsed`
/// block.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Parsed {
    #[serde(flatten)]
    pub buffers: BufferSizes,
    /// Whether the host-side weights are file-backed.
    ///
    /// Read from the loader's own naming (`CPU_Mapped model buffer size`
    /// against `CPU model buffer size`) rather than inferred from flags,
    /// because mainline and ik_llama disagree on it for identical
    /// configurations.
    pub cpu_model_mapped: Mapped,
    #[serde(flatten)]
    pub meta: MetaFields,
    /// Every numeric GGUF metadata key the loader echoed, first occurrence
    /// winning.
    pub gguf_kv: BTreeMap<String, i64>,
    pub contexts: Vec<Context>,
    #[serde(flatten)]
    pub mmproj: Option<Mmproj>,
    /// llama.cpp's own figure for an MTP context, which it reports per context.
    pub mtp_context_mib: f64,
    /// The embedded MTP head's depth.
    pub nextn_predict_layers: u64,
    /// Whether the model carries Gemma's per-layer token embeddings.
    pub per_layer_token_embd: bool,
    pub arch: String,
    pub devices: Vec<DeviceRow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_breakdown: Option<HostBreakdown>,
    #[serde(flatten)]
    pub gpu_mirrors: GpuMirrors,
}

/// Whether the host-side weights landed in `RssFile` or in anonymous memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mapped {
    Yes,
    #[default]
    No,
}

/// What a vision projector cost, as llama.cpp's own accounting states it.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Mmproj {
    /// Per device: the projector's weights *and* its CLIP graph buffer.
    #[serde(rename = "mmproj_reserved_mib")]
    pub reserved_mib: BTreeMap<String, f64>,
    /// The summed tensor sizes, which isolate the graph term from the figure
    /// above.
    #[serde(rename = "mmproj_tensor_bytes")]
    pub tensor_bytes: u64,
    #[serde(rename = "clip_image_size", skip_serializing_if = "Option::is_none")]
    pub image_size: Option<u64>,
    #[serde(rename = "clip_n_merge", skip_serializing_if = "Option::is_none")]
    pub n_merge: Option<u64>,
}

/// Pull the model's shape and its logged buffer sizes out of a load log.
pub fn parse_log(text: &str) -> Parsed {
    let devices = breakdown::parse_devices(text);
    Parsed {
        buffers: BufferSizes::parse(text),
        cpu_model_mapped: if text.contains("CPU_Mapped model buffer") {
            Mapped::Yes
        } else {
            Mapped::No
        },
        meta: MetaFields::parse(text),
        gguf_kv: parse_gguf_kv(text),
        contexts: contexts::parse_contexts(text),
        mmproj: parse_mmproj(text),
        mtp_context_mib: patterns::MTP
            .captures(text)
            .map(|caps| number(&caps, 1))
            .unwrap_or_default(),
        nextn_predict_layers: patterns::NEXTN
            .captures(text)
            .map(|caps| count(&caps, 1))
            .unwrap_or_default(),
        per_layer_token_embd: patterns::PER_LAYER_EMBD.is_match(text),
        arch: patterns::ARCH
            .captures(text)
            .map(|caps| text_at(&caps, 1))
            .unwrap_or_else(|| "?".to_owned()),
        gpu_mirrors: GpuMirrors::from_devices(&devices),
        host_breakdown: breakdown::parse_host(text),
        devices,
    }
}

fn parse_gguf_kv(text: &str) -> BTreeMap<String, i64> {
    let mut keys = BTreeMap::new();
    for caps in patterns::GGUF_KV.captures_iter(text) {
        let (Some(key), Some(value)) = (caps.get(1), caps.get(2)) else {
            continue;
        };
        if let Ok(value) = value.as_str().parse() {
            // First occurrence wins, matching the target-before-draft rule the
            // shape keys follow.
            keys.entry(key.as_str().to_owned()).or_insert(value);
        }
    }
    keys
}

fn parse_mmproj(text: &str) -> Option<Mmproj> {
    let mut reserved_mib = BTreeMap::new();
    for caps in patterns::MMPROJ_RESERVED.captures_iter(text) {
        if let Some(device) = caps.get(2) {
            reserved_mib.insert(device.as_str().to_owned(), number(&caps, 1));
        }
    }
    if reserved_mib.is_empty() {
        return None;
    }
    Some(Mmproj {
        reserved_mib,
        tensor_bytes: patterns::CLIP_TENSOR
            .captures_iter(text)
            .map(|caps| count(&caps, 1))
            .sum(),
        image_size: patterns::CLIP_IMAGE_SIZE
            .captures(text)
            .map(|caps| count(&caps, 1)),
        n_merge: patterns::CLIP_MERGE
            .captures(text)
            .map(|caps| count(&caps, 1)),
    })
}

/// The `<key>_all` spelling a repeated figure is recorded under.
fn repeats_key(key: &str) -> String {
    format!("{key}_all")
}

/// A capture group the pattern guarantees is present and numeric; anything else
/// is a mismatch between the pattern and this reader, not log variation.
fn number(caps: &regex::Captures<'_>, group: usize) -> f64 {
    caps.get(group)
        .and_then(|found| found.as_str().parse().ok())
        .unwrap_or_default()
}

fn count(caps: &regex::Captures<'_>, group: usize) -> u64 {
    caps.get(group)
        .and_then(|found| found.as_str().parse().ok())
        .unwrap_or_default()
}

fn text_at(caps: &regex::Captures<'_>, group: usize) -> String {
    caps.get(group)
        .map(|found| found.as_str().to_owned())
        .unwrap_or_default()
}
