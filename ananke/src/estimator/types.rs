//! Estimator output types.

use std::{collections::BTreeMap, path::Path};

use smol_str::SmolStr;

use crate::config::{DeviceSlot, ServiceConfig};

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
    /// but the context count varied. The estimator ran on a two-card machine
    /// for its whole history, so that cost was folded into the process
    /// baseline; an operator with four or eight cards inherited it wrong.
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
    pub split_mode: crate::config::validate::SplitMode,
    /// K-cache quantisation (f16, q8_0, etc.). Absent means f16.
    pub cache_type_k: Option<&'a str>,
    /// V-cache quantisation. Absent means f16.
    pub cache_type_v: Option<&'a str>,
    /// `override_tensor` regex rules to pin specific tensors to CPU / a GPU.
    pub override_tensor: &'a [String],
    /// Override for the compute-buffer reservation (MB per active device).
    /// Absent means the estimator's 400 MB default.
    pub compute_buffer_mb: Option<u32>,
    /// Whether the operator has opted into the coarse fallback when the
    /// GGUF's architecture isn't recognised by any per-family estimator.
    /// `false` by default — unknown architectures return an error instead
    /// of silently producing a guess that may be badly wrong.
    pub allow_fallback: bool,
    /// Whether the service runs with `--spec-type draft-mtp`. When set and
    /// the model carries an MTP head (`nextn_predict_layers > 0`), the
    /// estimator adds the MTP draft context's KV + compute overhead. See
    /// [`crate::estimator::mtp`].
    pub mtp: bool,
    /// Optional separate draft-model GGUF (`-md`). When `mtp` is set and
    /// this is present, the estimator reads this file's resident weights
    /// plus a draft compute buffer rather than the target model's embedded
    /// MTP head. See [`crate::estimator::mtp`].
    pub draft_model: Option<&'a Path>,
    /// Number of parallel slots (`-np`). Absent means llama.cpp's default
    /// of 1. With a non-unified cache this divides the KV budget per
    /// stream, which is what the host graph mask is sized against — see
    /// [`crate::estimator::host_buffer`].
    pub parallel: Option<u32>,
    /// Whether the child runs with flash attention (`-fa`). Absent means
    /// llama.cpp's default. Decides whether the graph's KQ mask is f16 or
    /// f32, i.e. halves or doubles the largest host-side allocation.
    pub flash_attn: Option<bool>,
    /// Whether the KV cache is unified across slots (`-kvu`). Absent means
    /// llama.cpp's default. Decides whether the mask is sized against the
    /// whole context or one slot's share.
    pub kv_unified: Option<bool>,
    /// Whether the child is served by ik_llama.cpp rather than mainline.
    /// The two forks size the pinned graph arena by measurably different
    /// rules — see [`crate::estimator::host_buffer`].
    pub ik_llama: bool,
    /// Whether the child runs ik_llama's DSA sparse attention (`-dsa`), which
    /// builds two additional full-cache masks on top of the ordinary one.
    pub ik_dsa: bool,
    /// Host RAM cap for the server's prompt cache (`-cram`, MiB). Absent
    /// means llama.cpp's 8192 MiB default — which is charged either way,
    /// since the runtime will fill up to whatever the cap is.
    pub cache_ram_mb: Option<u32>,
}

impl<'a> EstimatorInputs<'a> {
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

    /// Distil the estimator-relevant fields out of a `ServiceConfig`.
    /// Returns `None` if `svc` is a command-template service — the
    /// estimator only applies to llama-cpp workloads.
    /// Tell the estimate how many devices the child will see.
    ///
    /// Callers that hold a device snapshot should use this; `from_service`
    /// alone cannot know, because a service config does not describe the
    /// machine it will run on.
    pub fn with_visible_devices(mut self, devices: u32) -> Self {
        self.visible_devices = devices.max(1);
        self
    }

    pub fn from_service(svc: &'a ServiceConfig) -> Option<Self> {
        let lc = svc.llama_cpp()?;
        Some(Self {
            name: svc.name.as_str(),
            model: lc.model.as_path(),
            mmproj: lc.mmproj.as_deref(),
            context: lc.context.unwrap_or(4096),
            ubatch: extra_arg_value(&svc.extra_args, &["-ub", "--ubatch-size", "--ubatch_size"])
                .and_then(|v| v.parse().ok())
                .or(lc.ubatch_size),
            // One device unless a caller that can see the machine says
            // otherwise: under-stating this under-reserves by ~20 MiB a card,
            // which the rolling correction can absorb, where over-stating it
            // on a single-card host could not be recovered.
            visible_devices: 1,
            // Any expert offload at all puts the model in the hybrid regime.
            host_resident_experts: !matches!(
                lc.expert_offload,
                crate::config::validate::OffloadMode::Off
            ),
            split_mode: svc.split_mode,
            cache_type_k: lc.cache_type_k.as_deref(),
            cache_type_v: lc.cache_type_v.as_deref(),
            override_tensor: &lc.override_tensor,
            compute_buffer_mb: lc.estimation.compute_buffer_mb,
            allow_fallback: lc.estimation.allow_fallback.unwrap_or(false),
            mtp: lc.spec_type.as_deref() == Some("draft-mtp"),
            draft_model: lc.draft_model.as_deref(),
            ik_llama: lc.runtime.ik().is_some(),
            ik_dsa: lc.runtime.ik().is_some_and(|ik| ik.dsa),
            parallel: extra_arg_value(&svc.extra_args, &["-np", "--parallel"])
                .and_then(|v| v.parse().ok())
                .or(lc.parallel),
            // Unset means llama.cpp's `-fa auto`, which resolves to *on* for
            // CUDA — the only backend ananke supports — and off when there is
            // no device to support it. Assuming off for a GPU service doubles
            // the modelled mask and puts the prediction outside the rolling
            // correction's reach; assuming on for a CPU-only one would halve
            // it. Resolve the same way the runtime will.
            flash_attn: flash_attn_from_extra_args(&svc.extra_args)
                .or(lc.flash_attn)
                .or(Some(!matches!(
                    svc.placement_policy,
                    crate::config::PlacementPolicy::CpuOnly
                ))),
            kv_unified: kv_unified_from_extra_args(&svc.extra_args).or(lc.kv_unified),
            // An operator who set the flag by hand in `extra_args` gets that
            // value in the estimate too, so the reservation matches the
            // runtime rather than the default the flag overrode.
            cache_ram_mb: lc
                .cache_ram_mb
                .or_else(|| cache_ram_from_extra_args(&svc.extra_args)),
        })
    }

    /// Stable hash of every field that would change the estimate's
    /// numbers. Cache layers (currently the daemon-side
    /// `EstimateCache`) compare this against the value stored
    /// alongside a cached entry to detect "the operator edited
    /// `context` / `override_tensor` / `cache_type_*` / … without
    /// changing the GGUF path" — the model on disk is the same but
    /// the estimate isn't.
    ///
    /// `model` and `mmproj` paths are deliberately excluded because
    /// the cache keys on them separately (any path change is a
    /// different model, not a different config of the same model).
    /// `draft_model` *is* hashed here because the cache does not key on
    /// it separately, yet swapping the draft GGUF changes the estimate.
    /// `name` is excluded because it's a log-context-only field.
    pub fn config_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.context.hash(&mut hasher);
        self.ubatch.hash(&mut hasher);
        self.cache_type_k.hash(&mut hasher);
        self.cache_type_v.hash(&mut hasher);
        self.override_tensor.hash(&mut hasher);
        self.compute_buffer_mb.hash(&mut hasher);
        self.allow_fallback.hash(&mut hasher);
        self.mtp.hash(&mut hasher);
        self.draft_model.hash(&mut hasher);
        self.ik_llama.hash(&mut hasher);
        self.ik_dsa.hash(&mut hasher);
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

/// The value an operator passed for one of `names` in `extra_args`.
///
/// Returns the **last** occurrence, because that is what llama.cpp uses: a
/// repeated argument logs "only last value will be used"
/// (`common/arg.cpp`). `extra_args` is appended after every flag the daemon
/// renders, so whatever is found here is what the child will actually run
/// with — and therefore what the estimate has to be built from.
///
/// The `--flag=value` form is deliberately not handled: llama.cpp requires an
/// exact whole-token match and exits with `invalid argument`, so it fails
/// loudly rather than diverging silently.
pub fn extra_arg_value<'a>(extra_args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut found = None;
    let mut it = extra_args.iter().peekable();
    while let Some(arg) = it.next() {
        if names.contains(&arg.as_str())
            && let Some(v) = it.peek()
        {
            found = Some(v.as_str());
        }
    }
    found
}

/// Names llama.cpp accepts for the prompt-cache cap. `--` arguments have
/// underscores normalised to dashes, so `--cache_ram` is valid too.
const CACHE_RAM_FLAGS: &[&str] = &["--cache-ram", "-cram", "--cache_ram"];

/// Find a `--cache-ram` / `-cram` value an operator passed through
/// `extra_args`. Configs predate the dedicated key and still set it this way;
/// without this the daemon would reserve the 8 GiB default for a service that
/// runs with the cache switched off.
pub fn cache_ram_from_extra_args(extra_args: &[String]) -> Option<u32> {
    extra_arg_value(extra_args, CACHE_RAM_FLAGS).and_then(|v| v.parse().ok())
}

/// `-kvu` / `--kv-unified` (a bare flag) or `-no-kvu` / `--no-kv-unified`
/// from `extra_args`.
fn kv_unified_from_extra_args(extra_args: &[String]) -> Option<bool> {
    extra_args.iter().rev().find_map(|a| match a.as_str() {
        "-kvu" | "--kv-unified" => Some(true),
        "-no-kvu" | "--no-kv-unified" => Some(false),
        _ => None,
    })
}

/// `-fa` / `--flash-attn`, which takes `on`/`off`/`auto`/`1`/`0`, or the bare
/// `-no-fa`. `auto` yields `None` so the caller falls through to resolving it
/// the way the runtime will.
fn flash_attn_from_extra_args(extra_args: &[String]) -> Option<bool> {
    if extra_args
        .iter()
        .any(|a| a == "-no-fa" || a == "--no-flash-attn")
    {
        return Some(false);
    }
    match extra_arg_value(extra_args, &["-fa", "--flash-attn"])? {
        "on" | "1" | "true" => Some(true),
        "off" | "0" | "false" => Some(false),
        _ => None,
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
    /// [`crate::estimator::compute_buffer::output_logits_bytes`]): subtracting less than
    /// the true value keeps the secondaries safe, subtracting more would
    /// under-reserve and OOM them.
    pub output_buffer_bytes: u64,
    /// Weights `--split-mode tensor` holds on *every* spanned card instead of
    /// dividing: the narrow gating and shared-expert paths every shard consumes.
    /// Zero for an architecture that ships none, which is every dense model
    /// measured. See [`crate::estimator::replicated`].
    pub tensor_split_replicated_bytes: u64,
    /// Extra VRAM (bytes) for the MTP / NextN draft context when the
    /// service runs `--spec-type draft-mtp`. Zero when MTP is off or the
    /// model carries no MTP head. Reserved as a single lump on the
    /// primary GPU by the packer. See [`crate::estimator::mtp`].
    pub mtp_bytes: u64,
    /// The share of [`Self::mtp_bytes`] that is model tensors read from a GGUF
    /// — non-zero only for a separate draft model. See
    /// [`crate::estimator::mtp::mtp_weight_bytes`].
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
    /// pays them too. See [`crate::estimator::host_buffer`].
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
    /// Architecture string for diagnostics.
    pub architecture: SmolStr,
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
    /// in that bucket and is *not* the head — copying it too over-reserved
    /// gemma-4-E4B by 3 GiB. See
    /// [`crate::allocator::placement::Packer::distribute_sharded`].
    pub tied_head_bytes: u64,
    /// Small tensors (norms, rope tables) lumped together.
    pub other_bytes: u64,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::EstimatorInputs;
    use crate::config::validate::SplitMode;

    /// A baseline `EstimatorInputs` with every field populated. Tests clone
    /// this and flip a single field to prove the fingerprint is sensitive to
    /// it.
    fn baseline<'a>() -> EstimatorInputs<'a> {
        EstimatorInputs {
            name: "test",
            model: Path::new("/fake/model.gguf"),
            mmproj: None,
            context: 4096,
            ubatch: Some(512),
            visible_devices: 1,
            host_resident_experts: false,
            split_mode: SplitMode::Layer,
            cache_type_k: None,
            cache_type_v: None,
            override_tensor: &[],
            compute_buffer_mb: None,
            allow_fallback: false,
            mtp: false,
            draft_model: None,
            parallel: None,
            flash_attn: None,
            kv_unified: None,
            ik_llama: false,
            ik_dsa: false,
            cache_ram_mb: None,
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
