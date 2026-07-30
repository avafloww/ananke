//! One entry per context the server created, in creation order.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::parse::{count, number, patterns, text_at};

/// Everything the loader printed for one context, up to and including its
/// `graph nodes` line.
///
/// The first segment of a run belongs to llama.cpp's parameter-fitting dry run
/// — it reports the same shape with no weights loaded — so segments are kept
/// whole rather than merged, and a reader picks the one it wants by the pools
/// it holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Context {
    /// Per device, then per role, as the loader logged it. `Meta()` is the
    /// fused device a tensor split reports, and its figure is ONE card's share.
    pub buffers: BTreeMap<String, BTreeMap<BufferRole, f64>>,
    pub kv_pools: Vec<KvPool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rs_pool: Option<RsPool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_nodes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_splits: Option<u64>,
}

/// What a per-device buffer line was holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BufferRole {
    Model,
    Kv,
    Rs,
    Compute,
    Output,
}

/// The attention cache's summary line, term by term.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KvPool {
    /// The physical total across devices.
    pub total_mib: f64,
    /// Cells per sequence.
    pub cells: u64,
    /// The layer count that actually allocates, which excludes an MTP head.
    pub layers: u64,
    pub seqs: u64,
    pub seqs_max: u64,
    pub k_type: String,
    pub k_mib: f64,
    pub v_type: String,
    pub v_mib: f64,
}

/// The recurrent module's equivalent.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RsPool {
    pub total_mib: f64,
    pub cells: u64,
    pub layers: u64,
    pub seqs: u64,
    /// The speculative rollback depth: the state is replicated
    /// `seqs × (rs_seq + 1)` times, and this is non-zero only under
    /// speculative decoding.
    pub rs_seq: u64,
    pub r_mib: f64,
    pub s_mib: f64,
}

pub(crate) fn parse_contexts(text: &str) -> Vec<Context> {
    let mut contexts = Vec::new();
    let mut start = 0;
    for boundary in patterns::CONTEXT_END.find_iter(text) {
        let segment = &text[start..boundary.end()];
        start = boundary.end();
        contexts.push(parse_context(segment));
    }
    contexts
}

fn parse_context(segment: &str) -> Context {
    let mut context = Context::default();
    for caps in patterns::DEV_BUFFER.captures_iter(segment) {
        let (Some(stage), Some(device), Some(mib)) = (caps.get(1), caps.get(2), caps.get(4)) else {
            continue;
        };
        let Ok(mib) = mib.as_str().parse() else {
            continue;
        };
        let role = match caps.get(3).map(|kind| kind.as_str()) {
            Some("model") => BufferRole::Model,
            Some("KV") => BufferRole::Kv,
            Some("RS") => BufferRole::Rs,
            Some("compute") => BufferRole::Compute,
            Some("output") => BufferRole::Output,
            // ik_llama omits the kind entirely, so the stage names it instead.
            _ if stage.as_str().ends_with("load_tensors") => BufferRole::Model,
            _ => BufferRole::Compute,
        };
        context
            .buffers
            .entry(device.as_str().to_owned())
            .or_default()
            .insert(role, mib);
    }
    for caps in patterns::KV_POOL.captures_iter(segment) {
        context.kv_pools.push(KvPool {
            total_mib: number(&caps, 1),
            cells: count(&caps, 2),
            layers: count(&caps, 3),
            seqs: count(&caps, 4),
            seqs_max: count(&caps, 5),
            k_type: text_at(&caps, 6),
            k_mib: number(&caps, 7),
            v_type: text_at(&caps, 8),
            v_mib: number(&caps, 9),
        });
    }
    context.rs_pool = patterns::RS_POOL.captures(segment).map(|caps| RsPool {
        total_mib: number(&caps, 1),
        cells: count(&caps, 2),
        layers: count(&caps, 3),
        seqs: count(&caps, 4),
        rs_seq: count(&caps, 5),
        r_mib: number(&caps, 6),
        s_mib: number(&caps, 7),
    });
    for caps in patterns::GRAPH_SHAPE.captures_iter(segment) {
        let value = count(&caps, 2);
        match caps.get(1).map(|name| name.as_str()) {
            Some("nodes") => context.graph_nodes = Some(value),
            Some("splits") => context.graph_splits = Some(value),
            _ => {}
        }
    }
    context
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two contexts, as a server with a sliding-window sibling prints them.
    const LOG: &str = "\
I load_tensors:       Meta() model buffer size =  8616.47 MiB
I llama_kv_cache: size = 2560.00 MiB ( 32768 cells,  10 layers,  4/1 seqs), K (f16): 1280.00 MiB, V (f16): 1280.00 MiB
I sched_reserve:     Meta() compute buffer size =   215.52 MiB
I sched_reserve: graph nodes  = 2459
I sched_reserve: graph splits = 2
I llama_kv_cache: size = 3600.00 MiB (  4608 cells,  50 layers,  4/1 seqs), K (f16): 1800.00 MiB, V (f16): 1800.00 MiB
I sched_reserve:  CUDA_Host compute buffer size =    57.52 MiB
I sched_reserve: graph nodes  = 1211
";

    #[test]
    fn every_figure_is_attributed_to_its_own_context() {
        let contexts = parse_contexts(LOG);
        assert_eq!(contexts.len(), 2, "{contexts:?}");
        assert_eq!(contexts[0].kv_pools[0].cells, 32768);
        assert_eq!(contexts[0].buffers["Meta()"][&BufferRole::Model], 8616.47);
        assert_eq!(contexts[0].buffers["Meta()"][&BufferRole::Compute], 215.52);
        assert_eq!(contexts[1].kv_pools[0].layers, 50);
        assert_eq!(
            contexts[1].buffers["CUDA_Host"][&BufferRole::Compute],
            57.52
        );
        assert!(!contexts[1].buffers.contains_key("Meta()"), "{contexts:?}");
    }

    #[test]
    fn a_split_count_belongs_to_the_context_above_it() {
        let contexts = parse_contexts(LOG);
        assert_eq!(contexts[0].graph_nodes, Some(2459));
        assert_eq!(contexts[0].graph_splits, Some(2));
        assert_eq!(contexts[1].graph_nodes, Some(1211));
        assert_eq!(contexts[1].graph_splits, None);
    }

    #[test]
    fn ik_llamas_unnamed_buffer_takes_its_kind_from_the_stage() {
        let contexts = parse_contexts(
            "llm_load_tensors:      CUDA0 buffer size =  6992.89 MiB\n\
             llama_init_from_model: graph nodes  = 1
",
        );
        assert_eq!(contexts[0].buffers["CUDA0"][&BufferRole::Model], 6992.89);
    }
}
