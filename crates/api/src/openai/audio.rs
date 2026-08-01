//! `POST /v1/audio/transcriptions` — request envelope.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// `POST /v1/audio/transcriptions` request envelope (multipart/form-data).
///
/// The daemon only interprets the `model` field; the whole multipart body
/// (file included) is forwarded byte-for-byte to the upstream ASR service,
/// which ignores `model` and reads the remaining fields itself.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TranscriptionEnvelope {
    /// Model name (maps to an ananke service name with
    /// `modality = "transcription"`).
    pub model: String,
    /// The audio file to transcribe. Accepted formats depend on the
    /// upstream server (parakeet-server: WAV only; whisper-server with
    /// `--convert`: anything ffmpeg reads).
    #[schema(value_type = String, format = Binary)]
    pub file: String,
    /// Upstream-interpreted response format: `json` (default), `text`,
    /// or `verbose_json` (upstream-dependent; whisper-server also
    /// supports `srt` and `vtt`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}
