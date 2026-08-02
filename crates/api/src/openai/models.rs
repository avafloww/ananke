//! `GET /v1/models` — list available models.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::shared::{metadata::AnankeMetadata, modality::Modality};

/// One entry in the `GET /v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListing {
    /// Model id (the service name).
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Creation timestamp (always 0 for ananke-managed models).
    pub created: u64,
    /// Always `"ananke"`.
    pub owned_by: String,
    /// What kind of OpenAI endpoint this model serves. Non-standard
    /// OpenAI field, elided when [`Modality::Chat`] (the default), so
    /// strict OpenAI clients never see it; embedding clients can filter
    /// on it.
    #[serde(default, skip_serializing_if = "Modality::is_chat")]
    pub modality: Modality,
    /// Passthrough entries from `[[service]] metadata.*`. Non-standard
    /// OpenAI field, elided when empty; strict OpenAI clients ignore it.
    #[serde(default, skip_serializing_if = "AnankeMetadata::is_empty")]
    #[schema(value_type = Object)]
    pub ananke_metadata: AnankeMetadata,
}

impl ModelListing {
    /// One listing for a service ananke serves. The constructor exists so
    /// `object` and `owned_by` — constants OpenAI's schema fixes — are
    /// spelled in exactly one place.
    pub fn new(id: String, modality: Modality, ananke_metadata: AnankeMetadata) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: 0,
            owned_by: "ananke".to_string(),
            modality,
            ananke_metadata,
        }
    }
}

/// `GET /v1/models` response body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// Always `"list"`.
    pub object: String,
    /// Available models.
    pub data: Vec<ModelListing>,
}

impl ModelsResponse {
    /// Wrap listings in the response envelope.
    pub fn new(data: Vec<ModelListing>) -> Self {
        Self {
            object: "list".to_string(),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The response has to survive the round trip a client actually makes:
    /// serialised daemon-side, then deserialised out of an owned body.
    /// `anankectl`'s model picker is that client.
    #[test]
    fn a_listing_round_trips_through_an_owned_body() {
        let listing = ModelListing::new(
            "qwen".to_string(),
            Modality::default(),
            AnankeMetadata::new(),
        );
        let text = serde_json::to_string(&ModelsResponse::new(vec![listing])).unwrap();
        let back: ModelsResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(back.object, "list");
        assert_eq!(back.data[0].id, "qwen");
        assert_eq!(back.data[0].object, "model");
        assert_eq!(back.data[0].owned_by, "ananke");
    }
}
