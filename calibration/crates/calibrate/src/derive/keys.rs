//! The vocabularies a rate table can be keyed by, one type each.
//!
//! Four of them are in the committed document, and nothing in the JSON tells them
//! apart: `quantised_cache_rates` is keyed by architecture, `no_flash_attn_rates`
//! by architecture-and-variant, `baseline_offset` by variant-and-environment, and
//! `ik_moe_rates` by architecture-and-card-count. A lookup at the wrong vocabulary
//! never errors — it misses and takes the table's worst rate, which is a plausible
//! number — so the vocabulary is a type here rather than a spelling convention.
//! [`crate::derive::Table`] is generic over it, and a mismatched `get` is a
//! compile error.
//!
//! Each key renders exactly the string the committed document spells, and wraps
//! that rendering rather than its parts, so a table's ordering is the document's
//! ordering.

use std::fmt;

use serde::{Serialize, Serializer};

use crate::record::Record;

/// The architecture the loader named: `qwen35`, `gemma4`, `laguna`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArchKey(String);

impl ArchKey {
    /// The architecture the log names, absent where it named none.
    ///
    /// Every derivation keys on the architecture, so the empty string a
    /// failed-to-load run leaves behind has to read as absent rather than pool
    /// unrelated models under one key.
    pub fn named(record: &Record) -> Option<Self> {
        record
            .parsed
            .architecture()
            .map(|arch| Self(arch.to_owned()))
    }

    /// The architecture field verbatim, the empty string included.
    ///
    /// For the derivers that pair cells by model and read the architecture off the
    /// pair: the pairing has already excluded a cell that never loaded, so absence
    /// cannot arise and treating it as a distinct key would be inventing a case.
    pub fn recorded(record: &Record) -> Self {
        Self(record.parsed.arch.clone())
    }
}

/// The architecture, plus the distinctions that split one arch string.
///
/// `gemma4` covers three models whose host terms differ by more than the rolling
/// correction can travel: a mixture of experts, a dense model, and an E-variant.
/// Both discriminators are ones `host_buffer` already applies — `has_experts` and
/// `compute_buffer::is_gemma_e_variant` — so a key built from them is one the
/// estimator can construct at lookup time.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariantKey(String);

impl VariantKey {
    pub fn of(record: &Record) -> Self {
        Self(variant(record))
    }
}

/// A [`VariantKey`] plus the environment it ran in: `@ik`, `@nofa`.
///
/// Asked for by the baseline offset alone. It differs by runtime (ik sits 24 to
/// 192 MiB above mainline on the same architecture) and by flash attention, which
/// shifts it by +21 to +33 MiB on most architectures and +131 on lfm2 — on top of
/// the per-token arena rate, which is a separate term.
///
/// The flash-attention *rates* must not be keyed this way, which is what the two
/// types buy: ik is excluded from that derivation, so an ik-suffixed key would
/// have no row and would inherit the table's worst rate as its default.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VariantEnvironmentKey(String);

impl VariantEnvironmentKey {
    pub fn of(record: &Record) -> Self {
        let mut key = variant(record);
        if record.factors.runtime_is_ik() {
            key.push_str(IK_SUFFIX);
        }
        if record.factors.flash_attn.charged_unfused() {
            key.push_str(NO_FLASH_ATTN_SUFFIX);
        }
        Self(key)
    }
}

/// The architecture and the number of cards it ran on: `glm-dsa@2`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ArchCardsKey(String);

impl ArchCardsKey {
    pub fn new(arch: &ArchKey, cards: usize) -> Self {
        Self(format!("{arch}{CARDS_SEPARATOR}{cards}"))
    }

    /// An architecture spelled as though the card count were part of it, which no
    /// row of such a table can equal.
    ///
    /// The escape hatch for one caller,
    /// [`crate::derive::tuning::Tuning::ik_moe_rate_frozen_arch_miss`]: the arena
    /// model has always looked ik's MoE rate up by architecture alone against the
    /// `{arch}@{cards}` table, and so has always taken the table's `default`.
    /// Correcting it would move the arena model out from under every constant
    /// fitted on it, so the miss is reproduced rather than repaired — and named
    /// here so that it cannot be written by accident.
    pub fn without_cards_frozen_miss(arch: &ArchKey) -> Self {
        Self(arch.as_str().to_owned())
    }
}

/// The architecture and the two suffixes [`VariantKey`] splits it by.
fn variant(record: &Record) -> String {
    let parsed = &record.parsed;
    // The literal `"None"` is a real key, and appears in no table only because
    // every cell reaching a keyed deriver has an architecture. It is spelled this
    // way because the committed tables are keyed this way.
    let mut key = parsed.architecture().unwrap_or(UNNAMED).to_string();
    if parsed.n_expert != 0 {
        key.push_str(MOE_SUFFIX);
    }
    // The same discriminator `compute_buffer::is_gemma_e_variant` uses, read
    // from the load log rather than guessed from the filename. A filename proxy
    // would disagree with the estimator the moment an E-variant shipped under
    // another name: the analysis would fit one curve while the estimator
    // selected a different one.
    if parsed.per_layer_token_embd {
        key.push_str(E_VARIANT_SUFFIX);
    }
    key
}

/// What an architecture-less cell is keyed as, spelled the way the committed
/// tables spell it.
const UNNAMED: &str = "None";
const MOE_SUFFIX: &str = "+moe";
const E_VARIANT_SUFFIX: &str = "+e";
const IK_SUFFIX: &str = "@ik";
const NO_FLASH_ATTN_SUFFIX: &str = "@nofa";
const CARDS_SEPARATOR: &str = "@";

/// The rendering, the borrow, and the serialization — one spelling per key type,
/// which is what makes a table's order the document's order.
macro_rules! rendered_key {
    ($name:ident) => {
        impl $name {
            /// The key as the committed document spells it.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.0)
            }
        }
    };
}

rendered_key!(ArchKey);
rendered_key!(VariantKey);
rendered_key!(VariantEnvironmentKey);
rendered_key!(ArchCardsKey);
