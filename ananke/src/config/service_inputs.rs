//! Config → estimator/packer input distillation.
//!
//! The builders now live beside their consumers: `estimator_inputs` and
//! the llama.cpp CLI readers in `ananke-estimate`, `placement_inputs` in
//! `ananke-placement`. This module re-exports them so
//! `crate::config::service_inputs::…` paths inside the daemon are
//! unchanged by the split.

pub use ananke_estimate::service_inputs::{
    cache_ram_from_extra_args, estimator_inputs, extra_arg_value,
};
pub use ananke_placement::service_inputs::placement_inputs;
