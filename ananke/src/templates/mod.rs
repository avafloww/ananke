//! Template dispatch and rendering.
//!
//! Re-exported from `ananke-templates` so `crate::templates::…` paths
//! inside the daemon are unchanged by the split.

pub use ananke_templates::{
    PlaceholderContext, SubstituteError, substitute, substitute_argv, substitute_launcher_argv,
};
