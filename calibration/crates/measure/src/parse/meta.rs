//! The model's shape, as llama.cpp's own `print_info:` block states it.

use std::collections::{BTreeMap, BTreeSet};

use ananke_dataset::Parsed;

use crate::parse::patterns;

/// Read every hyperparameter out of a log into the record's flat fields.
///
/// The assignments are spelled out one per field rather than driven off a key
/// table, because the record names each hyperparameter and its disagreement
/// list as its own typed field: a table would only be a second place for the
/// spelling to drift from.
pub(crate) fn fill(text: &str, parsed: &mut Parsed) {
    let mut found = parse(text);
    let mut take = |key: MetaKey| found.remove(&key).unwrap_or_default();

    (parsed.n_layer, parsed.n_layer_all) = take(MetaKey::NLayer);
    (parsed.n_embd, parsed.n_embd_all) = take(MetaKey::NEmbd);
    (parsed.n_expert, parsed.n_expert_all) = take(MetaKey::NExpert);
    (parsed.n_expert_used, parsed.n_expert_used_all) = take(MetaKey::NExpertUsed);
    (parsed.n_swa, parsed.n_swa_all) = take(MetaKey::NSwa);
    (parsed.n_vocab, parsed.n_vocab_all) = take(MetaKey::NVocab);
    (parsed.n_head, parsed.n_head_all) = take(MetaKey::NHead);
    (parsed.n_head_kv, parsed.n_head_kv_all) = take(MetaKey::NHeadKv);
    (parsed.n_embd_head_k, parsed.n_embd_head_k_all) = take(MetaKey::NEmbdHeadK);
    (parsed.n_embd_head_v, parsed.n_embd_head_v_all) = take(MetaKey::NEmbdHeadV);
    (parsed.n_ctx_train, parsed.n_ctx_train_all) = take(MetaKey::NCtxTrain);
    (parsed.n_ff, parsed.n_ff_all) = take(MetaKey::NFf);
    (parsed.ssm_d_conv, parsed.ssm_d_conv_all) = take(MetaKey::SsmDConv);
    (parsed.ssm_d_inner, parsed.ssm_d_inner_all) = take(MetaKey::SsmDInner);
    (parsed.ssm_d_state, parsed.ssm_d_state_all) = take(MetaKey::SsmDState);
    (parsed.ssm_n_group, parsed.ssm_n_group_all) = take(MetaKey::SsmNGroup);
    (parsed.ssm_dt_rank, parsed.ssm_dt_rank_all) = take(MetaKey::SsmDtRank);
    (parsed.n_group_used, parsed.n_group_used_all) = take(MetaKey::NGroupUsed);
}

/// One hyperparameter's value, plus every occurrence when they disagreed.
///
/// The value is the *first* occurrence, not the last: the target model loads
/// before any draft or projector, so last-wins would record the draft's shape
/// for exactly the MTP cells whose target shape the constants are fitted
/// against.
///
/// A pair rather than a struct because [`fill`] destructures it straight into
/// the record's two fields, which name both halves at every use.
type MetaValue = (u64, Option<Vec<u64>>);

/// One hyperparameter the loader echoes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum MetaKey {
    NLayer,
    NEmbd,
    NExpert,
    NExpertUsed,
    NSwa,
    NVocab,
    NHead,
    NHeadKv,
    NEmbdHeadK,
    NEmbdHeadV,
    NCtxTrain,
    NFf,
    SsmDConv,
    SsmDInner,
    SsmDState,
    SsmNGroup,
    SsmDtRank,
    NGroupUsed,
}

impl MetaKey {
    const ALL: [MetaKey; 18] = [
        MetaKey::NLayer,
        MetaKey::NEmbd,
        MetaKey::NExpert,
        MetaKey::NExpertUsed,
        MetaKey::NSwa,
        MetaKey::NVocab,
        MetaKey::NHead,
        MetaKey::NHeadKv,
        MetaKey::NEmbdHeadK,
        MetaKey::NEmbdHeadV,
        MetaKey::NCtxTrain,
        MetaKey::NFf,
        MetaKey::SsmDConv,
        MetaKey::SsmDInner,
        MetaKey::SsmDState,
        MetaKey::SsmNGroup,
        MetaKey::SsmDtRank,
        MetaKey::NGroupUsed,
    ];

    /// The spelling llama.cpp prints, which is also the record's field name.
    fn key(self) -> &'static str {
        match self {
            MetaKey::NLayer => "n_layer",
            MetaKey::NEmbd => "n_embd",
            MetaKey::NExpert => "n_expert",
            MetaKey::NExpertUsed => "n_expert_used",
            MetaKey::NSwa => "n_swa",
            MetaKey::NVocab => "n_vocab",
            MetaKey::NHead => "n_head",
            MetaKey::NHeadKv => "n_head_kv",
            MetaKey::NEmbdHeadK => "n_embd_head_k",
            MetaKey::NEmbdHeadV => "n_embd_head_v",
            MetaKey::NCtxTrain => "n_ctx_train",
            MetaKey::NFf => "n_ff",
            MetaKey::SsmDConv => "ssm_d_conv",
            MetaKey::SsmDInner => "ssm_d_inner",
            MetaKey::SsmDState => "ssm_d_state",
            MetaKey::SsmNGroup => "ssm_n_group",
            MetaKey::SsmDtRank => "ssm_dt_rank",
            MetaKey::NGroupUsed => "n_group_used",
        }
    }

    fn from_key(key: &str) -> Option<Self> {
        MetaKey::ALL.into_iter().find(|meta| meta.key() == key)
    }
}

fn parse(text: &str) -> BTreeMap<MetaKey, MetaValue> {
    let mut occurrences: BTreeMap<MetaKey, Vec<u64>> = BTreeMap::new();
    for caps in patterns::META.captures_iter(text) {
        let (Some(key), Some(value)) = (caps.get(1), caps.get(2)) else {
            continue;
        };
        let (Some(key), Ok(value)) = (MetaKey::from_key(key.as_str()), value.as_str().parse())
        else {
            continue;
        };
        occurrences.entry(key).or_default().push(value);
    }
    occurrences
        .into_iter()
        .map(|(key, seen)| {
            let distinct = seen.iter().collect::<BTreeSet<_>>().len();
            let first = seen.first().copied().unwrap_or(0);
            (key, (first, (distinct > 1).then_some(seen)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(text: &str) -> Parsed {
        let mut parsed = Parsed::default();
        fill(text, &mut parsed);
        parsed
    }

    #[test]
    fn the_targets_shape_wins_over_a_drafts() {
        let parsed = shape(
            "print_info: n_layer           = 62\n\
             print_info: n_embd            = 5120\n\
             print_info: n_layer           = 4\n",
        );
        assert_eq!(parsed.n_layer, 62);
        assert_eq!(parsed.n_layer_all, Some(vec![62, 4]));
        // A key seen once, or seen twice with the same value, carries no list.
        assert_eq!(parsed.n_embd, 5120);
        assert_eq!(parsed.n_embd_all, None);
        assert_eq!(parsed.n_swa, 0);
    }

    #[test]
    fn a_longer_key_is_not_shadowed_by_its_prefix() {
        let parsed = shape(
            "print_info: n_expert          = 256\n\
             print_info: n_expert_used     = 8\n\
             print_info: n_embd_head_k     = 128\n",
        );
        assert_eq!(parsed.n_expert, 256);
        assert_eq!(parsed.n_expert_used, 8);
        assert_eq!(parsed.n_embd_head_k, 128);
        assert_eq!(parsed.n_embd, 0);
    }
}
