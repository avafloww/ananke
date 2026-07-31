//! The local model library, and the per-model settings every sweep must carry.
//!
//! A model is not just a path: an architecture that cannot be tensor-split, a
//! cache type the attention path rejects, or a native context below the sweep's
//! range all constrain which cells are worth planning at all. Planning cells that
//! cannot load wastes the slot and buries the failures that are real, so those
//! constraints live here beside the path rather than being rediscovered in each
//! sweep.

use std::path::{Path, PathBuf};

use ananke_config::placement::SplitMode;
use ananke_measure::record::{Factors, KvType, Runtime};

/// Where the model files live when `$LLM_DIR` says nothing.
pub const DEFAULT_LLM_DIR: &str = "/mnt/ssd0/ai/llm";

/// The root every model path is resolved against.
///
/// Read from `$LLM_DIR` rather than hardcoded so a plan is portable to another
/// machine with the same library. A plan with this box's absolute paths baked in
/// is a plan only this box can run.
pub struct Library {
    root: PathBuf,
}

impl Library {
    /// The root `$LLM_DIR` names, or the campaign machine's if it is unset.
    pub fn from_env() -> Self {
        Self {
            root: std::env::var_os("LLM_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_LLM_DIR)),
        }
    }

    /// A library rooted somewhere explicit, which is what makes the generated
    /// plans testable without touching the environment.
    pub fn rooted(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root itself, for a caller that joins its own paths onto it.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// One library-relative path, resolved.
    pub fn path_of(&self, relative: &str) -> String {
        self.root.join(relative).to_string_lossy().into_owned()
    }

    /// The same, for the paths a model may not have.
    pub fn path_opt(&self, relative: Option<&str>) -> Option<String> {
        relative.map(|r| self.path_of(r))
    }
}

/// A model in the local library, with what the estimator would need to know.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    pub key: &'static str,
    pub path: &'static str,
    pub runtimes: &'static [Runtime],
    pub mmproj: Option<&'static str>,
    pub draft: Option<&'static str>,
    /// Split modes the runtime can actually serve this architecture with.
    ///
    /// Not every architecture supports every mode. mainline's
    /// `llm_arch_supports_sm_tensor` (llama-arch.cpp) blocklists the mode, and of
    /// the models here that catches `deepseek4` and `lfm2`; `glm-dsa` is on the
    /// same list, and although ik does not gate on architecture at all, the
    /// operator serves that model hybrid rather than tensor-split, so measuring
    /// it would characterise a configuration nobody runs.
    pub splits: &'static [SplitMode],
    /// Flags the architecture needs to run the way production runs it.
    ///
    /// glm52 is served with `-mla 1 -dsa -amb 512`; without them ik takes a
    /// different attention path entirely, so a cell that omits them measures
    /// something the operator never runs — and `glm-dsa`'s curve is calibrated
    /// against the DSA path specifically.
    pub extra: &'static [&'static str],
    /// KV cache types the model can actually be served with.
    ///
    /// `-dsa` rejects a quantised cache, so sweeping q8_0 on glm52 plans eight
    /// cells that cannot load.
    pub kv_types: &'static [KvType],
    /// Card counts worth measuring. A 350M embedding model does not need two, and
    /// spreading it changes the very baseline the cell exists to isolate.
    pub gpus: &'static [&'static str],
    /// Native context, where it is below the sweep's range.
    ///
    /// talkie tops out at 2048. Requesting more does not fail — llama.cpp runs
    /// past `n_ctx_train` with a warning — it just makes every point on the curve
    /// an extrapolation from a regime the model was never trained for, which is
    /// not a calibration.
    pub max_ctx: Option<u32>,
    /// Whether the model is served with `--embeddings` and has no generation.
    pub embeddings: bool,
    pub threads: Option<u32>,
    pub no_mmap: bool,
    pub n_cpu_moe: Option<u32>,
    /// Expert layers to keep on the CPU when only one card is visible.
    ///
    /// `n_cpu_moe` is sized for both cards. A hybrid tuned that way does not fit
    /// on one — laguna keeps 18 layers on the GPU, which aborts a single-card
    /// load — so the single-GPU cells need their own figure or the `gpus` axis is
    /// simply missing for every large model.
    pub n_cpu_moe_1gpu: Option<u32>,
}

impl Model {
    /// Split modes both the architecture and the runtime will accept.
    pub fn splits_for(&self, runtime: Runtime) -> Vec<SplitMode> {
        let Some(allowed) = runtime_splits(runtime) else {
            return self.splits.to_vec();
        };
        let kept: Vec<_> = self
            .splits
            .iter()
            .copied()
            .filter(|s| allowed.contains(s))
            .collect();
        if kept.is_empty() {
            allowed.to_vec()
        } else {
            kept
        }
    }

    /// The per-model settings every sweep has to carry, in one place.
    ///
    /// Returned as a `Factors` so a sweep spreads it with struct-update syntax
    /// and cannot forget a field: scattering these across the sweep functions is
    /// how glm52 came to be planned without the DSA flags it is always served
    /// with.
    pub fn flags(&self, gpus: &str) -> Factors {
        Factors {
            extra: self.extra.iter().map(|s| (*s).to_owned()).collect(),
            embeddings: self.embeddings,
            threads: self.threads,
            no_mmap: self.no_mmap,
            n_cpu_moe: if gpus == "0" {
                self.n_cpu_moe_1gpu.or(self.n_cpu_moe)
            } else {
                self.n_cpu_moe
            },
            ..Factors::default()
        }
    }

    /// The card count a sweep should use when the model only needs one: a model
    /// that needs one card keeps one, since spreading a 350M embedding model
    /// across two changes the baseline the curve is fitted against.
    pub fn preferred_gpus(&self) -> &'static str {
        if self.gpus.contains(&"0,1") {
            "0,1"
        } else {
            "0"
        }
    }
}

/// The model this key names.
///
/// Panics on an unknown key: every caller is a sweep in this crate, so a typo is
/// a programming error rather than an operator's mistake.
pub fn model(key: &str) -> &'static Model {
    MODELS
        .iter()
        .find(|m| m.key == key)
        .unwrap_or_else(|| panic!("{key} is not a model in the library"))
}

/// How a runtime is spelled in a cell label.
pub fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Mainline => "mainline",
        Runtime::Ik => "ik",
    }
}

/// Every model the campaign draws on, smallest first only by coincidence — the
/// run order is decided in [`crate::plan::order_by_disturbance`], not here.
pub const MODELS: &[Model] = &[
    // Dense, 36L, n_embd 2560 — the fast factorial subject.
    Model::new(
        "qwen3-4b",
        "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-UD-Q5_K_XL.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik]),
    // Dense, 65L, n_embd 5120, embedded MTP head.
    Model::new(
        "qwen36-27b",
        "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik])
    .mmproj("unsloth/Qwen3.6-27B-GGUF/mmproj-F16.gguf"),
    // Dense + SWA 1024, separate draft GGUF.
    Model::new(
        "gemma4-31b-qat",
        "unsloth/gemma-4-31B-it-qat-GGUF/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
    )
    .mmproj("unsloth/gemma-4-31B-it-qat-GGUF/mmproj-F16.gguf")
    .draft("unsloth/gemma-4-31B-it-qat-GGUF/mtp-gemma-4-31B-it.gguf"),
    // Dense + SWA, no MoE — isolates SWA from experts.
    Model::new(
        "gemma3-27b",
        "mlabonne/gemma-3-27b-it-abliterated-GGUF/gemma-3-27b-it-abliterated.q4_k_m.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik]),
    // MoE 256/8, 41L.
    Model::new(
        "qwen36-35b-a3b",
        "unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik])
    .mmproj("unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf")
    .n_cpu_moe(40),
    // MoE 256/10 + SWA 512, 48L.
    Model::new(
        "laguna",
        "unsloth/Laguna-S-2.1-GGUF/UD-IQ4_NL/Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik])
    .n_cpu_moe(30)
    .n_cpu_moe_1gpu(39),
    // MoE + MLA + NSA indexer, 43L; mainline rejects tensor split.
    Model::new(
        "dsv4f",
        "unsloth/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf",
    )
    .n_cpu_moe(40)
    .splits(&[SplitMode::Layer]),
    // MoE + MLA + DSA, 79L — the production quant.
    Model::new(
        "glm52",
        "muzzy/GLM-5.2-GGUF/IQ2_KS/GLM-5.2-smol-IQ2_KS-00001-of-00033.gguf",
    )
    .runtimes(&[Runtime::Ik])
    .n_cpu_moe(92)
    .n_cpu_moe_1gpu(96)
    .extra(&["-mla", "1", "-dsa", "-amb", "512"])
    .kv_types(&[KvType::F16])
    .threads(24)
    .no_mmap()
    .splits(&[SplitMode::Layer]),
    // Every remaining llama.cpp service in the operator's config. These were
    // absent from the registry while being served in production daily, which
    // meant the campaign's "holdout" covered under half of what the daemon
    // actually runs.
    //
    // gemma4 MoE, 30L.
    Model::new(
        "gemma4-26b-a4b",
        "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    )
    .mmproj("unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-F16.gguf"),
    // gemma4 E-variant, 42L — the 1100/7 curve.
    Model::new(
        "gemma4-e4b",
        "unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q5_K_XL.gguf",
    )
    .mmproj("unsloth/gemma-4-E4B-it-GGUF/mmproj-F16.gguf"),
    // llama arch, 40L — the llama-family default curve.
    Model::new(
        "magidonia-24b",
        "bartowski/TheDrummer_Magidonia-24B-v4.3-GGUF/TheDrummer_Magidonia-24B-v4.3-Q5_K_M.gguf",
    )
    .runtimes(&[Runtime::Mainline, Runtime::Ik]),
    // talkie arch, 40L, full MHA; native context 2048.
    Model::new(
        "talkie-13b",
        "mradermacher/talkie-1930-13b-it-hf-GGUF/talkie-1930-13b-it-hf.Q6_K.gguf",
    )
    .max_ctx(2048),
    // Embedding modality; 128k native context. mainline's
    // `llm_arch_supports_sm_tensor` blocklists lfm2.
    Model::new(
        "lfm2-embed",
        "LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf",
    )
    .embeddings()
    .gpus(&["0"])
    .splits(&[SplitMode::Layer]),
];

/// ik_llama's `--split-mode` takes none/graph/layer — there is no `tensor`, and
/// passing one is a hard argument error rather than a fallback. Its analogue is
/// `graph`, which the operator does not run, so restricting to layer keeps the
/// fork's cells comparable to mainline's rather than measuring a mode nobody
/// uses.
fn runtime_splits(runtime: Runtime) -> Option<&'static [SplitMode]> {
    match runtime {
        Runtime::Mainline => None,
        Runtime::Ik => Some(&[SplitMode::Layer]),
    }
}

impl Model {
    /// A model with the defaults every entry shares, refined by the builders
    /// below. Written this way so an entry spells only what is unusual about it,
    /// which is what makes the unusual parts readable.
    const fn new(key: &'static str, path: &'static str) -> Self {
        Self {
            key,
            path,
            runtimes: &[Runtime::Mainline],
            mmproj: None,
            draft: None,
            splits: &[SplitMode::Layer, SplitMode::Tensor],
            extra: &[],
            kv_types: &[KvType::F16, KvType::Q80],
            gpus: &["0", "0,1"],
            max_ctx: None,
            embeddings: false,
            threads: None,
            no_mmap: false,
            n_cpu_moe: None,
            n_cpu_moe_1gpu: None,
        }
    }

    const fn runtimes(mut self, runtimes: &'static [Runtime]) -> Self {
        self.runtimes = runtimes;
        self
    }

    const fn mmproj(mut self, path: &'static str) -> Self {
        self.mmproj = Some(path);
        self
    }

    const fn draft(mut self, path: &'static str) -> Self {
        self.draft = Some(path);
        self
    }

    const fn splits(mut self, splits: &'static [SplitMode]) -> Self {
        self.splits = splits;
        self
    }

    const fn extra(mut self, extra: &'static [&'static str]) -> Self {
        self.extra = extra;
        self
    }

    const fn kv_types(mut self, kv_types: &'static [KvType]) -> Self {
        self.kv_types = kv_types;
        self
    }

    const fn gpus(mut self, gpus: &'static [&'static str]) -> Self {
        self.gpus = gpus;
        self
    }

    const fn max_ctx(mut self, max_ctx: u32) -> Self {
        self.max_ctx = Some(max_ctx);
        self
    }

    const fn embeddings(mut self) -> Self {
        self.embeddings = true;
        self
    }

    const fn threads(mut self, threads: u32) -> Self {
        self.threads = Some(threads);
        self
    }

    const fn no_mmap(mut self) -> Self {
        self.no_mmap = true;
        self
    }

    const fn n_cpu_moe(mut self, layers: u32) -> Self {
        self.n_cpu_moe = Some(layers);
        self
    }

    const fn n_cpu_moe_1gpu(mut self, layers: u32) -> Self {
        self.n_cpu_moe_1gpu = Some(layers);
        self
    }
}
