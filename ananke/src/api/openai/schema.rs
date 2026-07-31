//! Request and response envelopes for the unified OpenAI listener.
//!
//! The schema types live in `ananke-api` (`ananke_api::openai`), part of the
//! shared wire-types crate alongside all other endpoint types. This module
//! re-exports them under `crate::api::openai::schema`.

pub use ananke_api::openai::{
    ChatCompletionEnvelope, CompletionEnvelope, EmbeddingEnvelope, ModelListing, ModelsResponse,
};
