//! The model's shape, as llama.cpp's own `print_info:` block states it.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Index,
};

use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::parse::{patterns, repeats_key};

/// One hyperparameter the loader echoes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetaKey {
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
    /// In the record's own order, which is the order the campaign's rows carry.
    pub(crate) const ALL: [MetaKey; 18] = [
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

    /// The spelling llama.cpp prints, which is also the record key.
    pub fn key(self) -> &'static str {
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

/// One hyperparameter's value, plus every occurrence when they disagreed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaValue {
    /// The first occurrence, not the last: the target model loads before any
    /// draft or projector, and last-wins recorded the *draft's* shape for
    /// exactly the MTP cells whose target shape the constants are fitted
    /// against.
    pub first: u64,
    /// Every occurrence, in log order, recorded only when they were not all
    /// the same value.
    pub all: Option<Vec<u64>>,
}

/// Every hyperparameter in one log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaFields {
    values: [MetaValue; 18],
}

impl MetaFields {
    pub(crate) fn parse(text: &str) -> Self {
        let mut found: BTreeMap<MetaKey, Vec<u64>> = BTreeMap::new();
        for caps in patterns::META.captures_iter(text) {
            let (Some(key), Some(value)) = (caps.get(1), caps.get(2)) else {
                continue;
            };
            let (Some(key), Ok(value)) = (MetaKey::from_key(key.as_str()), value.as_str().parse())
            else {
                continue;
            };
            found.entry(key).or_default().push(value);
        }
        let values = MetaKey::ALL.map(|key| {
            let occurrences = found.remove(&key).unwrap_or_default();
            let distinct = occurrences.iter().collect::<BTreeSet<_>>().len();
            MetaValue {
                first: occurrences.first().copied().unwrap_or(0),
                all: (distinct > 1).then_some(occurrences),
            }
        });
        Self { values }
    }
}

impl Index<MetaKey> for MetaFields {
    type Output = MetaValue;

    fn index(&self, key: MetaKey) -> &MetaValue {
        &self.values[key as usize]
    }
}

impl Serialize for MetaFields {
    /// Flattened into the record, with the disagreement list under
    /// `<key>_all`.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for key in MetaKey::ALL {
            let value = &self[key];
            map.serialize_entry(key.key(), &value.first)?;
            if let Some(all) = &value.all {
                map.serialize_entry(&repeats_key(key.key()), all)?;
            }
        }
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_targets_shape_wins_over_a_drafts() {
        let meta = MetaFields::parse(
            "print_info: n_layer           = 62\n\
             print_info: n_embd            = 5120\n\
             print_info: n_layer           = 4\n",
        );
        assert_eq!(meta[MetaKey::NLayer].first, 62);
        assert_eq!(meta[MetaKey::NLayer].all, Some(vec![62, 4]));
        // A key seen once, or seen twice with the same value, carries no list.
        assert_eq!(meta[MetaKey::NEmbd].first, 5120);
        assert_eq!(meta[MetaKey::NEmbd].all, None);
        assert_eq!(meta[MetaKey::NSwa].first, 0);
    }

    #[test]
    fn a_longer_key_is_not_shadowed_by_its_prefix() {
        let meta = MetaFields::parse(
            "print_info: n_expert          = 256\n\
             print_info: n_expert_used     = 8\n\
             print_info: n_embd_head_k     = 128\n",
        );
        assert_eq!(meta[MetaKey::NExpert].first, 256);
        assert_eq!(meta[MetaKey::NExpertUsed].first, 8);
        assert_eq!(meta[MetaKey::NEmbdHeadK].first, 128);
        assert_eq!(meta[MetaKey::NEmbd].first, 0);
    }
}
