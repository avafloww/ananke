//! The packer's public result and error types: `CommandArgs`, `PackError`,
//! `Packed`, and the internal `ShardedPlan` handed between the sharded-split
//! step and `finish`.

use crate::{config::SplitMode, devices::Allocation};

#[derive(Debug, Clone, Default)]
pub struct CommandArgs {
    /// `-ngl N` value. `None` means do not emit the flag (caller uses
    /// `placement_override` escape hatch or cpu-only).
    pub ngl: Option<u32>,
    /// `--tensor-split A,B,...`. In layer mode these are per-GPU layer
    /// counts; in a sharded (tensor/row) mode they are equal proportions
    /// (one `1` per spanned GPU).
    pub tensor_split: Option<Vec<u32>>,
    /// `-ot <regex>=<device>` rules, rendered verbatim from
    /// `service.raw.override_tensor`.
    pub override_tensor: Vec<String>,
    /// `--split-mode {row,tensor}` when the packer used a sharded
    /// (tensor-parallel) distribution. `None` keeps llama.cpp's default
    /// (`layer`), so layer-split services emit no `--split-mode` flag and
    /// their argv is unchanged.
    pub split_mode: Option<SplitMode>,
    /// `--main-gpu N` — the CUDA-visible index (after the
    /// `CUDA_VISIBLE_DEVICES` remap) that gathers intermediate results and
    /// KV in sharded modes. Always the lowest-id spanned GPU, which
    /// `cuda_env::render`'s ascending ordering places at visible index 0.
    pub main_gpu: Option<u32>,
    /// `--n-cpu-moe N` — offload the trailing `N` expert layers' experts to
    /// CPU as whole layers. Set on the coarse expert-offload path instead of
    /// synthesising per-tensor `-ot` rules: keeping whole layers together
    /// keeps the runtime's fused multi-threaded CPU MoE kernel engaged
    /// (~24× faster on ik_llama than scattered `-ot`) and stays under
    /// llama.cpp's `GGML_SCHED_MAX_SPLIT_INPUTS` graph-split limit. When
    /// set, the runtime distributes the GPU-resident experts across cards
    /// itself, so `-ngl 999` is emitted and no expert `-ot`/`--tensor-split`
    /// is synthesised. `None` on the non-expert paths.
    pub n_cpu_moe: Option<u32>,
}

/// Structured packer failure modes. Each variant carries the numbers the
/// operator needs to understand the overflow — no more string-matching
/// on the message to figure out what went wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// A specific block-layer's bytes didn't fit on any GPU the service
    /// was allowed to use, and CPU spill was disabled.
    LayerDoesNotFit { layer_index: u32, bytes: u64 },
    /// The estimator returned no per-layer breakdown (fallback path on
    /// an unknown architecture) and the weights can't fit on any
    /// allowed device.
    WeightsDoNotFit,
    /// A sharded (tensor/row) split's equal per-GPU share didn't fit on one
    /// of the spanned GPUs. Unlike layer split there is no CPU spill — every
    /// spanned GPU must hold its shard, so a single overflow fails the pack.
    ShardDoesNotFit { gpu_index: u32, bytes: u64 },
    /// Even after offloading every eligible expert tensor to the CPU, the
    /// bytes the packer wants to keep on the host exceed the available host
    /// RAM (minus the configured `[devices.cpu] reserved_gb`).
    CpuDoesNotFit { needed: u64, available: u64 },
    /// A manual `expert_offload = N` pins every non-offloaded layer's experts
    /// to its home GPU regardless of fit, and the experts kept on `gpu_index`
    /// overflow it. Unlike `Auto`, manual mode never spills the surplus for the
    /// operator — the fix is a larger offload count.
    ManualExpertsDoNotFit { gpu_index: u32, bytes: u64 },
    /// `tensor_split_weights` count doesn't match the number of spanned GPUs.
    /// This is a configuration error, not a capacity problem — eviction
    /// cannot fix it.
    InvalidTensorSplitWeights { expected: usize, got: usize },
}

impl std::fmt::Display for PackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerDoesNotFit { layer_index, bytes } => {
                write!(
                    f,
                    "layer {layer_index} ({bytes} bytes) does not fit on any allowed GPU"
                )
            }
            Self::WeightsDoNotFit => f.write_str("weights do not fit on any allowed device"),
            Self::ShardDoesNotFit { gpu_index, bytes } => {
                write!(
                    f,
                    "tensor-split shard ({bytes} bytes) does not fit on gpu:{gpu_index}"
                )
            }
            Self::CpuDoesNotFit { needed, available } => {
                write!(
                    f,
                    "host RAM offload ({needed} bytes) exceeds available CPU memory ({available} bytes)"
                )
            }
            Self::ManualExpertsDoNotFit { gpu_index, bytes } => {
                write!(
                    f,
                    "manual expert_offload keeps more expert weight on gpu:{gpu_index} than it can hold (overflow at a {bytes}-byte expert tensor); raise the expert_offload count"
                )
            }
            Self::InvalidTensorSplitWeights { expected, got } => {
                write!(
                    f,
                    "tensor_split_weights has {got} entries but {expected} GPUs are spanned"
                )
            }
        }
    }
}

impl std::error::Error for PackError {}

#[derive(Debug)]
pub struct Packed {
    pub allocation: Allocation,
    pub args: CommandArgs,
    /// Total expert-tensor bytes the packer moved onto the CPU (MoE
    /// auto/manual offload). Zero when no experts were offloaded. Surfaced in
    /// the placement preview so the UI can show "N layers · X GiB → CPU".
    pub expert_offload_bytes: u64,
    /// Number of distinct layers with at least one expert tensor offloaded to
    /// the CPU.
    pub expert_offload_layers: u32,
}

/// A tensor/row-split distribution decided by
/// [`crate::allocator::placement::sharded::distribute_sharded`].
/// [`crate::allocator::placement::finish`] turns it into `--split-mode`,
/// `--main-gpu`, and the `--tensor-split` ratio (equal `1`s by default, or
/// the weighted integers derived from `tensor_split_weights`).
#[derive(Debug)]
pub(crate) struct ShardedPlan {
    pub(crate) mode: SplitMode,
    /// Integer tensor-split values emitted for this sharded plan. Stored here
    /// so `finish` can render the same ratio that `distribute_sharded` used
    /// for the pledge book. The length is one entry per spanned GPU, in
    /// ascending GPU-id order.
    pub(crate) tensor_split: Vec<u32>,
}
