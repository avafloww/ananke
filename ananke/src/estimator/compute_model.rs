//! The unified per-device compute model.
//!
//! One design of named columns, fitted per (runtime, split, architecture,
//! variant), replaces what used to be three separate mechanisms: a fitted
//! per-architecture curve for a mainline layer split, a hyperparameter model for
//! a tensor split plus companion tables for its intermediates, quantised-cache
//! rate, and per-device shadow, and four rate tables for ik. They were three
//! because each was derived against a different target — llama.cpp's `compute`
//! column plus `unaccounted`, the fused `Meta()` row's `compute` column alone,
//! and for ik the driver total less a modelled remainder, ik printing no
//! breakdown table at all. Three targets meant three shapes, and a cell that
//! fell between them was covered by whichever mechanism claimed it rather than
//! by the one that described it.
//!
//! Here every cell is measured the same way, as what one card holds beyond its
//! own weights and context, so all of the data constrains one model. The
//! columns and their coefficients live in `tuning.json`; what each column counts
//! is documented on [`Columns`], and `scripts/calibration/compute_model.py`
//! holds the fit that produces them.
//!
//! Two quantities come out of it, because a layer split is not symmetric:
//! [`per_device_mib`] is what *every* spanned card holds, and
//! [`head_extra_mib`] is what the primary card holds on top. Charging the pair
//! that way also reproduces a tensor split, whose fused row reports a per-card
//! average: the average times the card count is the same total either way.

use crate::{
    config::validate::SplitMode,
    estimator::{
        tuning::{
            COMPUTE_MODEL, COMPUTE_MODEL_DEFAULT, ComputeCoefficients, DEFAULT_UBATCH,
            IK_ATTENTION_CHUNK, MAINLINE_LAYER_SPLIT_MASK_COPIES,
        },
        types::EstimatorInputs,
    },
    gguf::GgufSummary,
};

/// The design row for one device, in the units the fitted coefficients expect.
///
/// `flat`, `head_flat`, and `doubling` are dimensionless or MiB-valued, so their
/// coefficients carry MiB. The rest are element counts divided by 2^20, so their
/// coefficients carry *bytes per element* and the dot product lands in MiB
/// throughout. That normalisation is the reason architectures of different width
/// can share one set of coefficients rather than each needing an absolute flat
/// term of its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Columns {
    /// Always one: the CUDA context, workspaces, and graph metadata.
    flat: f64,
    /// The primary card's share — 1 on the head card of a layer split, 0 on the
    /// others, and `1 / cards` for a tensor split whose figure is an average.
    head_flat: f64,
    /// `n_embd × ubatch`. Graph intermediates are some count of hidden-width f32
    /// buffers per batch token, so the coefficient is that count times four.
    hidden: f64,
    /// `log2(ubatch / chunk)` above ik's attention chunk, else zero. The fork
    /// grows its attention scratch by a constant per doubling of the batch
    /// rather than proportionally, which no affine term can express.
    doubling: f64,
    /// `copies × ubatch × n_kv` — the KQ mask, with the replication a split
    /// costs. The coefficient lands on 2 for nearly every group, an f16 entry
    /// per (batch token, cache token), which is a check on the model rather
    /// than a parameter of it.
    mask: f64,
    /// `ubatch × context` on a quantised cache, else zero. It follows the full
    /// context rather than one stream's share.
    quant: f64,
    /// `head_share × n_vocab × ubatch` — the logits the head card materialises.
    logits: f64,
    /// `head_share` when experts are host-resident. Gathering and scattering
    /// CPU-resident expert activations stages through the primary device, and
    /// that — not the output head — is what makes a hybrid layer split
    /// asymmetric.
    offload_head: f64,
}

const BYTES_PER_MIB: f64 = (1024 * 1024) as f64;

/// What the columns are computed from, once the GGUF and the placement have been
/// read. Every field is a quantity the fitter has too.
struct Scalars {
    ubatch: f64,
    n_kv: f64,
    ctx: f64,
    quantised: bool,
    head_share: f64,
    n_vocab: f64,
    n_embd: f64,
    offloading: bool,
    mask_copies: f64,
}

impl Columns {
    /// The row for a card holding `head_share` of the primary card's duties.
    pub(crate) fn new(
        summary: &GgufSummary,
        inputs: &EstimatorInputs<'_>,
        head_share: f64,
    ) -> Self {
        let arch = summary.architecture.as_str();
        let ubatch = f64::from(inputs.ubatch.unwrap_or(DEFAULT_UBATCH).max(1));
        let n_embd = f64::from(
            summary
                .metadata
                .get(&smol_str::SmolStr::new(format!("{arch}.embedding_length")))
                .and_then(|v| v.as_u32())
                .unwrap_or(0),
        );
        let n_vocab = summary
            .tensors
            .get("output.weight")
            .or_else(|| summary.tensors.get("token_embd.weight"))
            .and_then(|t| t.shape.iter().max().copied())
            .unwrap_or(0) as f64;
        Self::from_scalars(Scalars {
            ubatch,
            n_kv: f64::from(inputs.context / inputs.streams().max(1)),
            ctx: f64::from(inputs.context),
            quantised: crate::estimator::host_buffer::quantised_kv(inputs),
            head_share,
            n_vocab,
            n_embd,
            offloading: inputs.host_resident_experts,
            mask_copies: f64::from(mask_copies(inputs)),
        })
    }

    /// The column arithmetic itself, over plain scalars.
    ///
    /// Separated from [`Self::new`] because this half — and only this half — has
    /// to agree with the fitter's `columns()`. Reading the scalars out of a GGUF
    /// summary is the estimator's own business; computing the columns from them is
    /// a contract shared with `scripts/calibration/compute_model.py`, and the test
    /// below holds the two to fixtures the fitter generates.
    fn from_scalars(s: Scalars) -> Self {
        let chunk = IK_ATTENTION_CHUNK.max(1) as f64;
        Self {
            flat: 1.0,
            head_flat: s.head_share,
            hidden: s.n_embd * s.ubatch / BYTES_PER_MIB,
            doubling: (s.ubatch.max(chunk) / chunk).log2(),
            mask: s.mask_copies * s.ubatch * s.n_kv / BYTES_PER_MIB,
            quant: if s.quantised {
                s.ubatch * s.ctx / BYTES_PER_MIB
            } else {
                0.0
            },
            logits: s.head_share * s.n_vocab * s.ubatch / BYTES_PER_MIB,
            offload_head: if s.offloading { s.head_share } else { 0.0 },
        }
    }

    /// This row's columns by name, in the order `tuning.json` declares them.
    ///
    /// Exists for the contract test below, which asserts these against fixtures
    /// generated by the fitter. Keeping it in terms of the same field names means
    /// a column added to the struct and not to this list fails to compile.
    #[cfg(test)]
    fn by_name(&self) -> [(&'static str, f64); 8] {
        [
            ("flat", self.flat),
            ("head_flat", self.head_flat),
            ("hidden", self.hidden),
            ("doubling", self.doubling),
            ("mask", self.mask),
            ("quant", self.quant),
            ("logits", self.logits),
            ("offload_head", self.offload_head),
        ]
    }

    /// This row against a group's coefficients, in MiB.
    fn dot(&self, coefficients: &ComputeCoefficients) -> f64 {
        self.flat * coefficients.flat
            + self.head_flat * coefficients.head_flat
            + self.hidden * coefficients.hidden
            + self.doubling * coefficients.doubling
            + self.mask * coefficients.mask
            + self.quant * coefficients.quant
            + self.logits * coefficients.logits
            + self.offload_head * coefficients.offload_head
    }
}

/// What every card spanned by this service holds, in MiB.
pub(crate) fn per_device_mib(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u32 {
    let coefficients = coefficients_for(summary, inputs);
    round_up(Columns::new(summary, inputs, 0.0).dot(coefficients))
}

/// What the primary card holds *on top of* [`per_device_mib`], in MiB.
///
/// A tensor split's figure is an average over the cards it spans, so the head
/// terms were fitted at `1 / cards` there. Reconstructing them at a full share
/// and charging them once to the primary card yields the same total across the
/// cards, and is what the packer can actually attribute.
pub(crate) fn head_extra_mib(summary: &GgufSummary, inputs: &EstimatorInputs<'_>) -> u32 {
    let coefficients = coefficients_for(summary, inputs);
    let with_head = Columns::new(summary, inputs, 1.0).dot(coefficients);
    let without = Columns::new(summary, inputs, 0.0).dot(coefficients);
    round_up(with_head - without)
}

/// The coefficients for this model, runtime, and split, from the generated
/// table.
///
/// A runtime-specific entry wins over one fitted across both forks, because the
/// two build different graphs for the same architecture. Falling all the way
/// through means nobody has measured this combination, and the pooled default
/// applies.
fn coefficients_for(
    summary: &GgufSummary,
    inputs: &EstimatorInputs<'_>,
) -> &'static ComputeCoefficients {
    let arch = summary.architecture.as_str();
    let variant = variant_of(summary);
    let split = match inputs.split_mode {
        SplitMode::Tensor | SplitMode::Row => "tensor",
        SplitMode::Layer => "layer",
    };
    let runtime = inputs.ik_llama.then_some("ik");
    let matches = |entry: &&'static crate::estimator::tuning::ComputeEntry| {
        entry.archs.contains(&arch) && entry.variant == variant && entry.split == split
    };
    COMPUTE_MODEL
        .iter()
        .find(|entry| matches(entry) && entry.runtime == runtime)
        .or_else(|| {
            COMPUTE_MODEL
                .iter()
                .find(|entry| matches(entry) && entry.runtime.is_none())
        })
        .map(|entry| &entry.coefficients)
        .unwrap_or(&COMPUTE_MODEL_DEFAULT)
}

/// How many copies of the KQ mask this placement holds.
///
/// A tensor split reports one card's share, so its mask is unreplicated whatever
/// the card count. A hybrid does not replicate it either — measured at 1.00
/// against a fully-resident model's 4.00 — so the replication follows from the
/// placement rather than from the model.
fn mask_copies(inputs: &EstimatorInputs<'_>) -> u32 {
    let layer_split_across_cards =
        inputs.split_mode == SplitMode::Layer && inputs.visible_devices > 1;
    if layer_split_across_cards && !inputs.host_resident_experts {
        MAINLINE_LAYER_SPLIT_MASK_COPIES as u32
    } else {
        1
    }
}

/// The variant discriminator, where one architecture string covers models whose
/// graphs differ.
fn variant_of(summary: &GgufSummary) -> Option<&'static str> {
    crate::estimator::compute_buffer::is_gemma_e_variant(summary).then_some("gemma_e")
}

/// MiB, rounded away from zero: a reservation short by a byte is still short.
fn round_up(mib: f64) -> u32 {
    if mib <= 0.0 {
        return 0;
    }
    mib.ceil().min(f64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The design columns must match the fitter's, value for value.
    ///
    /// `build.rs` already asserts that the column *names* in `tuning.json` match
    /// the generated struct, which catches a column added on one side and not the
    /// other. It cannot catch the two sides computing a column *differently* — an
    /// off-by-one in a stream divisor, or one side forgetting that the mask
    /// carries its replication count — and a coefficient fitted against one
    /// definition then multiplies the other.
    ///
    /// The fixtures are regenerated by `analyse.py emit` beside `tuning.json`, so
    /// changing the fitter's `columns()` fails this test until the evaluator is
    /// changed to match. That is the single-source-of-truth property the two
    /// implementations cannot otherwise have.
    #[test]
    fn the_columns_match_the_fitter() {
        #[derive(serde::Deserialize)]
        struct Fixtures {
            columns: Vec<String>,
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            inputs: Inputs,
            columns: std::collections::BTreeMap<String, f64>,
        }
        #[derive(serde::Deserialize)]
        struct Inputs {
            ubatch: u32,
            n_kv: u32,
            ctx: u32,
            quantised: bool,
            head_share: f64,
            n_vocab: u64,
            n_embd: u32,
            offloaded: u32,
            mask_copies: u32,
        }

        let raw = include_str!("../../tests/fixtures/compute_columns.json");
        let fixtures: Fixtures = serde_json::from_str(raw).expect("fixture parses");

        // The declared order is part of the contract: a coefficient vector is
        // read positionally in `tuning.json`.
        let empty = Columns::from_scalars(Scalars {
            ubatch: 0.0,
            n_kv: 0.0,
            ctx: 0.0,
            quantised: false,
            head_share: 0.0,
            n_vocab: 0.0,
            n_embd: 0.0,
            offloading: false,
            mask_copies: 0.0,
        });
        let names: Vec<&str> = empty.by_name().iter().map(|(n, _)| *n).collect();
        assert_eq!(
            fixtures.columns, names,
            "the fitter's column list has diverged from the evaluator's"
        );

        for (index, case) in fixtures.cases.iter().enumerate() {
            let built = Columns::from_scalars(Scalars {
                ubatch: f64::from(case.inputs.ubatch),
                n_kv: f64::from(case.inputs.n_kv),
                ctx: f64::from(case.inputs.ctx),
                quantised: case.inputs.quantised,
                head_share: case.inputs.head_share,
                n_vocab: case.inputs.n_vocab as f64,
                n_embd: f64::from(case.inputs.n_embd),
                offloading: case.inputs.offloaded > 0,
                mask_copies: f64::from(case.inputs.mask_copies),
            });
            for (name, value) in built.by_name() {
                let expected = case.columns[name];
                assert!(
                    (value - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "case {index}, column `{name}`: evaluator {value} against fitter {expected}"
                );
            }
        }
    }
}
