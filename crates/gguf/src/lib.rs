//! GGUF reader — single-file and sharded.
//!
//! All filesystem interaction flows through a [`ananke_fs::Fs`] handle,
//! so tests can substitute [`ananke_fs::InMemoryFs`] preloaded with
//! synthetic bytes. Production calls pass [`ananke_fs::LocalFs`].

pub mod keys;
pub mod reader;
pub mod shards;
pub mod types;

pub use reader::{ReadError, read_single};
pub use shards::read;
pub use types::{GgufSummary, GgufTensor, GgufType, GgufValue};
