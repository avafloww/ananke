//! Estimator output types.

use std::{collections::BTreeMap, path::Path};

use ananke_config::placement::DeviceSlot;
use ananke_gguf::Architecture;

/// Pure inputs the estimator reads. The daemon builds one of these from a
/// `ServiceConfig` on each spawn; standalone callers (calibration tools,
/// model-size inspection examples) construct one directly without having
/// to fabricate an entire `ServiceConfig`.
///
/// Lifetimes keep it borrow-only — a `&EstimatorInputs<'_>` never owns its
/// strings or path references.
#[derive(Debug, Clone)]
pub struct EstimatorInputs<'a> {
    /// Service name — used for log context only; the estimate itself is
    /// identical regardless of what this says.
    pub name: &'a str,
    /// Path to the first GGUF shard.
    pub model: &'a Path,
    /// Optional vision projector (adds its tensor bytes to the weights
    /// total and the first-GPU non-layer bucket).
    pub mmproj: Option<&'a Path>,
    /// Context window the child will be launched with. Absent means 4096.
    pub context: u32,
    /// Physical batch size (`--ubatch-size` / `-ub`) the child will launch
    /// with. Absent means llama.cpp's default of 512. Only the deepseek4
    /// NSA-indexer compute buffer scales with it (∝ `ubatch × context`);
    /// every other architecture's compute buffer is ~ubatch-independent, so
    /// the estimator ignores this outside that arch.
    pub ubatch: Option<u32>,
    /// How many CUDA devices the child will have visible.
    ///
    /// Each one costs host memory for its context — measured at ~20 MiB per
    /// device beyond the first, with placement pinned to the CPU so nothing
    /// but the context count varied. It has to be charged per device rather
    /// than folded into the process baseline, which would only be right at the
    /// card count the baseline was fitted at.
    pub visible_devices: u32,
    /// Whether the child will hold expert tensors on the host.
    ///
    /// A hybrid does not replicate the graph's attention masks across devices
    /// where a fully-resident model does — measured at 1.00 against 4.00, with
    /// a resident mixture of experts also at 4.00, so it follows from the
    /// placement rather than from the model being MoE.
    pub host_resident_experts: bool,
    /// How the child will split the model across those devices.
    ///
    /// mainline replicates the graph's attention masks when layers are split
    /// across more than one device, and does not under tensor split, where
    /// llama.cpp fuses the cards into a single device. The difference is a
    /// multiple of a term that scales with `n_kv x n_tokens`, so it is worth
    /// more than a gigabyte at production contexts.
    pub split_mode: ananke_config::placement::SplitMode,
    /// K-cache quantisation (f16, q8_0, etc.). Absent means f16.
    pub cache_type_k: Option<&'a str>,
    /// V-cache quantisation. Absent means f16.
    pub cache_type_v: Option<&'a str>,
    /// `override_tensor` regex rules to pin specific tensors to CPU / a GPU.
    pub override_tensor: &'a [String],
    /// Override for the compute-buffer reservation (MB per active device).
    /// Absent means the estimator's 400 MB default.
    pub compute_buffer_mb: Option<u32>,
    /// Speculative decoding, and where the draft head comes from.
    pub speculation: Speculation<'a>,
    /// Number of parallel slots (`-np`). Absent means llama.cpp's default
    /// of 1. With a non-unified cache this divides the KV budget per
    /// stream, which is what the host graph mask is sized against — see
    /// [`crate::host_buffer`].
    pub parallel: Option<u32>,
    /// Whether the child runs with flash attention (`-fa`). Absent means
    /// llama.cpp's default. Decides whether the graph's KQ mask is f16 or
    /// f32, i.e. halves or doubles the largest host-side allocation.
    pub flash_attn: Option<bool>,
    /// Whether the KV cache is unified across slots (`-kvu`). Absent means
    /// llama.cpp's default. Decides whether the mask is sized against the
    /// whole context or one slot's share.
    pub kv_unified: Option<bool>,
    /// Which llama.cpp serves the child, and the knobs only that fork has.
    pub fork: Fork,
    /// Host RAM cap for the server's prompt cache (`-cram`, MiB). Absent
    /// means llama.cpp's 8192 MiB default — which is charged either way,
    /// since the runtime will fill up to whatever the cap is.
    pub cache_ram_mb: Option<u32>,
}

impl<'a> EstimatorInputs<'a> {
    /// The inputs for `model` with every optional knob at llama.cpp's default.
    ///
    /// `Default` cannot express this — `model` has no meaningful default and
    /// the whole struct borrows — so this is the spreadable base a caller
    /// overrides the two or three fields it cares about on:
    ///
    /// ```ignore
    /// EstimatorInputs { context: 32768, ..EstimatorInputs::empty(path) }
    /// ```
    pub fn empty(model: &'a Path) -> Self {
        Self {
            name: "",
            model,
            mmproj: None,
            context: 0,
            ubatch: None,
            visible_devices: 1,
            host_resident_experts: false,
            split_mode: ananke_config::placement::SplitMode::Layer,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &[],
            compute_buffer_mb: None,
            speculation: Speculation::None,
            parallel: None,
            flash_attn: None,
            kv_unified: None,
            fork: Fork::Mainline,
            cache_ram_mb: None,
        }
    }

    /// How many separate KV caches the service runs.
    ///
    /// One per slot, unless the slots share a unified cache. Several terms are
    /// sized against one sequence's share of the context rather than the whole
    /// of it, so they divide by this.
    pub fn streams(&self) -> u32 {
        if self.kv_unified.unwrap_or(false) {
            1
        } else {
            self.parallel.unwrap_or(1).max(1)
        }
    }

    /// Tell the estimate how many devices the child will see.
    ///
    /// Callers that hold a device snapshot should use this; `from_service`
    /// alone cannot know, because a service config does not describe the
    /// machine it will run on.
    pub fn with_visible_devices(mut self, devices: u32) -> Self {
        self.visible_devices = devices.max(1);
        self
    }

    /// Stable hash of every field that would change the estimate's numbers.
    /// The daemon's `EstimateCache` compares this against the value stored
    /// with a cached entry to catch "the operator edited `context` /
    /// `override_tensor` / `cache_type_*` without changing the GGUF path".
    ///
    /// `model` and `mmproj` are excluded because the cache keys on them
    /// separately: a path change is a different model, not a different config
    /// of the same one. The draft GGUF is hashed here, because the cache does
    /// not key on it and swapping it does change the estimate. `name` is
    /// excluded because it only reaches the logs.
    pub fn config_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.context.hash(&mut hasher);
        self.ubatch.hash(&mut hasher);
        self.cache_type_k.hash(&mut hasher);
        self.cache_type_v.hash(&mut hasher);
        self.override_tensor.hash(&mut hasher);
        self.compute_buffer_mb.hash(&mut hasher);
        self.speculation.hash(&mut hasher);
        self.fork.hash(&mut hasher);
        self.parallel.hash(&mut hasher);
        self.flash_attn.hash(&mut hasher);
        self.kv_unified.hash(&mut hasher);
        self.cache_ram_mb.hash(&mut hasher);
        self.visible_devices.hash(&mut hasher);
        self.host_resident_experts.hash(&mut hasher);
        self.split_mode.hash(&mut hasher);
        hasher.finish()
    }
}

/// Which llama.cpp serves a child.
///
/// The two forks size the pinned graph arena by measurably different rules, so
/// this is not cosmetic — see [`crate::host_buffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Fork {
    #[default]
    Mainline,
    /// ik_llama.cpp. `dsa` is its sparse attention (`-dsa`), which builds two
    /// additional full-cache masks and which mainline has no equivalent of —
    /// hence a field on the variant rather than a flag beside it.
    Ik { dsa: bool },
}

impl Fork {
    pub fn is_ik(self) -> bool {
        matches!(self, Self::Ik { .. })
    }

    /// Whether ik's sparse attention is on. False for mainline, which cannot
    /// run it.
    pub fn dsa(self) -> bool {
        matches!(self, Self::Ik { dsa: true })
    }
}

/// Where a service's speculative-decoding draft head comes from.
///
/// A draft GGUF without `--spec-type draft-mtp` is meaningless, so the two are
/// one value rather than a flag and an `Option` that can disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Speculation<'a> {
    #[default]
    None,
    /// `--spec-type draft-mtp` against the target model's own head, present
    /// when it declares `nextn_predict_layers > 0`.
    EmbeddedMtp,
    /// The same, with the head shipped as a separate GGUF passed as `-md`.
    DraftMtp(&'a Path),
}

impl<'a> Speculation<'a> {
    /// Whether the child runs `--spec-type draft-mtp` at all.
    pub fn is_mtp(self) -> bool {
        !matches!(self, Self::None)
    }

    /// The separate draft GGUF, if the head is not the target model's own.
    pub fn draft_model(self) -> Option<&'a Path> {
        match self {
            Self::DraftMtp(path) => Some(path),
            _ => None,
        }
    }
}

/// Base estimate for a service's VRAM footprint, pre-safety-factor and
/// pre-rolling-correction.
#[derive(Debug, Clone)]
pub struct Estimate {
    /// Static weight bytes (including mmproj if present).
    pub weights_bytes: u64,
    /// KV cache bytes per context token (zero for architectures without KV).
    pub kv_per_token: u64,
    /// Compute buffer per device in MB (default 400).
    pub compute_buffer_mb: u32,
    /// Output logits buffer bytes — the `n_vocab × ubatch`-sized activation
    /// llama.cpp allocates only on the device that holds the output head
    /// (the first GPU). Every other GPU's real compute buffer is smaller by
    /// this amount, since it never materialises logits. The packer reserves
    /// the full [`Self::compute_buffer_mb`] on the head GPU but subtracts
    /// this term on the secondaries, so they fill with more expert weight
    /// instead of a phantom logits buffer. Deliberately a conservative
    /// under-estimate of the real logits allocation (see
    /// [`crate::compute_buffer::output_logits_bytes`]): subtracting less than
    /// the true value keeps the secondaries safe, subtracting more would
    /// under-reserve and OOM them.
    pub output_buffer_bytes: u64,
    /// Expert layers the MTP head accounted for, dropped from
    /// [`Self::expert_layers`] because ik does not load them. Non-zero only for
    /// an ik service on a model with an embedded head. See
    /// `Ncmoe` for why the count
    /// matters to `-ncmoe`.
    pub mtp_head_expert_layers: u32,
    /// Weights `--split-mode tensor` holds on *every* spanned card instead of
    /// dividing: the narrow gating and shared-expert paths every shard consumes.
    /// Zero for an architecture that ships none, which is every dense model
    /// measured. See [`crate::replicated`].
    pub tensor_split_replicated_bytes: u64,
    /// Extra VRAM (bytes) for the MTP / NextN draft context when the
    /// service runs `--spec-type draft-mtp`. Zero when MTP is off or the
    /// model carries no MTP head. Reserved as a single lump on the
    /// primary GPU by the packer. See [`crate::mtp`].
    pub mtp_bytes: u64,
    /// The share of [`Self::mtp_bytes`] that is model tensors read from a GGUF
    /// — non-zero only for a separate draft model. See
    /// [`crate::mtp::mtp_weight_bytes`].
    pub mtp_weight_bytes: u64,
    /// The vision projector's CLIP graph buffer, beyond its weights. llama.cpp
    /// puts it on one device and says so — `[mtmd] adding N MiB to
    /// fit_params_target for device CUDA0` — so the packer charges it to the
    /// main GPU as runtime, not as weight: it is a device allocation and never
    /// appears in the host RSS the rolling correction subtracts weights from.
    /// Zero without an mmproj.
    pub mmproj_graph_bytes: u64,
    /// Host bytes the runtime is predicted to hold that are neither weights
    /// nor KV: the pinned graph arena and the process baseline. Charged to the
    /// `Cpu` slot whatever the placement, because a fully GPU-offloaded model
    /// pays them too. See [`crate::host_buffer`].
    pub host_overhead_bytes: u64,
    /// The server prompt cache's host-RAM cap (`-cram`). Reserved but not
    /// predicted — it fills with use rather than at load — so the packer
    /// charges it as slop and the rolling correction never divides by it.
    pub host_cache_bytes: u64,
    /// Host RAM the slots beyond the first need when they are all busy.
    /// Reserved but not predicted, like [`Self::host_cache_bytes`]: an idle
    /// slot costs nothing, so the packer charges this as slop and the rolling
    /// correction never divides by it.
    pub host_slot_bytes: u64,
    /// Host RAM the server's context checkpoints need once prompts are long
    /// enough to be checkpointed. Reserved but not predicted, like
    /// [`Self::host_cache_bytes`]: a short-prompt service never allocates it.
    pub host_checkpoint_bytes: u64,
    /// Per-layer weight bytes for index-ordered packing. `None` for
    /// architectures where layer-aware placement isn't applicable
    /// (currently SSM/Mamba; in that case `placement` uses single-device
    /// best-fit on `total = weights + compute_buffer`).
    pub per_layer_bytes: Option<Vec<u64>>,
    /// Layer indices that are attention-bearing (used to scope KV
    /// cost to those layers). `None` = all layers carry KV.
    pub attention_layers: Option<Vec<u32>>,
    /// Non-layer tensors: output head, token embeddings, norms.
    pub non_layer: NonLayer,
    /// Tensor-level overrides (from `override_tensor` rules) already
    /// resolved to per-device byte attributions by the estimator.
    pub override_tensor_bytes: BTreeMap<DeviceSlot, u64>,
    /// Layer indices that carry expert (`_exps`) tensors — diagnostic only.
    /// Empty for non-MoE architectures.
    pub expert_layers: Vec<u32>,
    /// The offloadable expert tensors (fused `blk.N.ffn_{gate,up,down}_exps`),
    /// `Some` for MoE architectures. The packer chooses which of these to move
    /// off-GPU (to CPU or a secondary GPU) to make the model fit, and
    /// synthesises the matching `-ot` rules. These bytes are *also* counted in
    /// `per_layer_bytes[i]` (the full per-layer cost); when the packer offloads
    /// an expert it subtracts that tensor's bytes from the layer's GPU share.
    /// Keeping `per_layer_bytes` full means every non-expert-aware code path
    /// (the plain layer walk, the sharded/tensor-split path, `override_tensor`
    /// accounting) stays correct without special-casing MoE. `None` for non-MoE
    /// architectures.
    pub expert_tensors: Option<Vec<ExpertTensor>>,
    /// `context` that was used to compute `kv_per_token × context`.
    pub context: u32,
    /// The model's graph, carried through for diagnostics and for the
    /// packer's architecture-specific decisions.
    pub architecture: Architecture,
}

/// One offloadable fused expert tensor on a MoE layer. llama.cpp stacks every
/// expert of a given projection into a single tensor per layer
/// (`blk.N.ffn_gate_exps.weight`, …), so there are at most three per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertTensor {
    /// Block/layer index this expert tensor belongs to.
    pub layer: u32,
    /// Which projection (gate / up / down) this is.
    pub kind: ExpertKind,
    /// Tensor weight bytes — the amount freed from GPU when offloaded.
    pub bytes: u64,
}

/// The three expert projections a MoE layer can carry. Used to build precise
/// `-ot blk.N.ffn_<kind>_exps.=<device>` rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertKind {
    Gate,
    Up,
    Down,
}

impl ExpertKind {
    /// The `ffn_<…>_exps` token as it appears in the GGUF tensor name and the
    /// `-ot` regex.
    pub fn tensor_token(self) -> &'static str {
        match self {
            ExpertKind::Gate => "gate",
            ExpertKind::Up => "up",
            ExpertKind::Down => "down",
        }
    }
}

/// Non-layer tensor footprint (matches llama.cpp's behaviour).
#[derive(Debug, Clone, Default)]
pub struct NonLayer {
    /// Output head — attributed to GPU 0 if any GPU used, else CPU.
    pub output_head_bytes: u64,
    /// Token embeddings — always on CPU. Includes Gemma 4 E-variants'
    /// `per_layer_token_embd.weight` stack, which llama.cpp keeps there too.
    pub token_embd_bytes: u64,
    /// `token_embd.weight` alone, and only for a model that ties it to the
    /// output head — zero whenever [`Self::output_head_bytes`] is non-zero.
    ///
    /// A tied model has no separate head, so this table *is* the head. Under a
    /// tensor split spanning more than one device that matmul is sharded, and a
    /// CPU-resident weight cannot be, so llama.cpp keeps a second GPU-resident
    /// copy split across the cards while the CPU copy stays. Distinct from
    /// [`Self::token_embd_bytes`] because the E-variants' per-layer stack sits
    /// in that bucket and is *not* the head — copying it too over-reserves
    /// gemma-4-E4B by 3 GiB. See
    /// `Packer::distribute_sharded`.
    pub tied_head_bytes: u64,
    /// Small tensors (norms, rope tables) lumped together.
    pub other_bytes: u64,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ananke_config::placement::SplitMode;

    use super::EstimatorInputs;

    /// A baseline `EstimatorInputs` with every field populated. Tests clone
    /// this and flip a single field to prove the fingerprint is sensitive to
    /// it.
    fn baseline<'a>() -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "test",
            context: 4096,
            ubatch: Some(512),
            split_mode: SplitMode::Layer,
            ..EstimatorInputs::empty(Path::new("/fake/model.gguf"))
        }
    }

    #[test]
    fn fingerprint_distinguishes_split_mode() {
        let a = baseline();
        let b = EstimatorInputs {
            split_mode: SplitMode::Tensor,
            ..baseline()
        };
        assert_ne!(
            a.config_fingerprint(),
            b.config_fingerprint(),
            "split_mode must participate in the fingerprint"
        );
    }

    #[test]
    fn fingerprint_distinguishes_visible_devices() {
        let a = baseline();
        let b = EstimatorInputs {
            visible_devices: 2,
            ..baseline()
        };
        assert_ne!(
            a.config_fingerprint(),
            b.config_fingerprint(),
            "visible_devices must participate in the fingerprint"
        );
    }

    #[test]
    fn fingerprint_distinguishes_host_resident_experts() {
        let a = baseline();
        let b = EstimatorInputs {
            host_resident_experts: true,
            ..baseline()
        };
        assert_ne!(
            a.config_fingerprint(),
            b.config_fingerprint(),
            "host_resident_experts must participate in the fingerprint"
        );
    }

    #[test]
    fn fingerprint_stable_for_identical_inputs() {
        assert_eq!(
            baseline().config_fingerprint(),
            baseline().config_fingerprint(),
            "identical inputs must produce identical fingerprints"
        );
    }
}
