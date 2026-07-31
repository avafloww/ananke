//! Turn a captured llama-server log into the shape and the buffer sizes the
//! calibration campaign is fitted against.
//!
//! Nothing here fails: a figure the log does not carry is recorded as its zero
//! rather than as an error, because a run that logged less than expected is
//! still a measurement and the record has to say what was seen.
//!
//! The result is [`ananke_dataset::Parsed`] — the dataset's own schema, not a
//! writer's private view of it. That block is flat, so each submodule below
//! fills the fields it produces rather than returning a group to be flattened
//! at serialisation time: a `serde(flatten)` group cannot coexist with the
//! `deny_unknown_fields` that makes a forgotten column a parse error.

use std::collections::BTreeMap;

pub use ananke_dataset::{
    BufferRole, Context, DeviceRow, HostBreakdown, KvPool, Mapped, Parsed, RsPool,
};

mod breakdown;
mod buffers;
mod contexts;
mod meta;
mod patterns;

/// Pull the model's shape and its logged buffer sizes out of a load log.
pub fn parse_log(text: &str) -> Parsed {
    let devices = breakdown::parse_devices(text);
    let mut parsed = Parsed {
        cpu_model_mapped: if text.contains("CPU_Mapped model buffer") {
            Mapped::Yes
        } else {
            Mapped::No
        },
        gguf_kv: parse_gguf_kv(text),
        contexts: contexts::parse_contexts(text),
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
        host_breakdown: breakdown::parse_host(text),
        ..Parsed::default()
    };
    buffers::fill(text, &mut parsed);
    meta::fill(text, &mut parsed);
    breakdown::fill_mirrors(&devices, &mut parsed);
    fill_mmproj(text, &mut parsed);
    parsed.devices = devices;
    parsed
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

/// What a vision projector cost, as llama.cpp's own accounting states it.
///
/// The four fields travel together — a log either loaded a projector or did not
/// — but the format nests none of them, so they are four independently optional
/// fields, left absent together.
fn fill_mmproj(text: &str, parsed: &mut Parsed) {
    let mut reserved_mib = BTreeMap::new();
    for caps in patterns::MMPROJ_RESERVED.captures_iter(text) {
        if let Some(device) = caps.get(2) {
            reserved_mib.insert(device.as_str().to_owned(), number(&caps, 1));
        }
    }
    if reserved_mib.is_empty() {
        return;
    }
    parsed.mmproj_reserved_mib = Some(reserved_mib);
    // The summed tensor sizes, which isolate the graph term out of the
    // reservation above.
    parsed.mmproj_tensor_bytes = Some(
        patterns::CLIP_TENSOR
            .captures_iter(text)
            .map(|caps| count(&caps, 1))
            .sum(),
    );
    parsed.clip_image_size = patterns::CLIP_IMAGE_SIZE
        .captures(text)
        .map(|caps| count(&caps, 1));
    parsed.clip_n_merge = patterns::CLIP_MERGE
        .captures(text)
        .map(|caps| count(&caps, 1));
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
