//! Unified OpenAI listener — Axum router factory.

pub mod errors;
pub mod filters;
pub mod handlers;
pub mod llamacpp;
pub mod metrics;
pub mod stall;
pub mod unimplemented;

use axum::Router;

use crate::daemon::app_state::AppState;

pub fn router(state: AppState) -> Router {
    handlers::register(Router::new(), state.clone()).merge(llamacpp::register(Router::new(), state))
}
