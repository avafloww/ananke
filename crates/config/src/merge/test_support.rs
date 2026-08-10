//! Shared TOML-parsing fixtures for this folder module's unit tests.
//!
//! Centralised so `field_merge`, `migrations`, and `resolve` don't each carry
//! their own copy of the same parse-and-lookup helpers.

use std::path::Path;

use crate::parse::{RawConfig, RawService, parse_toml};

pub fn parse(src: &str) -> RawConfig {
    parse_toml(src, Path::new("/t")).unwrap()
}

pub fn find_llama<'a>(cfg: &'a RawConfig, name: &str) -> &'a crate::parse::RawLlamaCppService {
    let svc = cfg
        .services
        .iter()
        .find(|s| s.common().name.as_deref() == Some(name))
        .unwrap();
    match svc {
        RawService::LlamaCpp(lc) => lc,
        _ => panic!("expected llama-cpp service"),
    }
}
