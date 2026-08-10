//! Resolve `extends` inheritance and `*_append` concatenation, and
//! `migrate_from` rename chains, before validation.

mod field_merge;
mod migrations;
mod resolve;
#[cfg(test)]
mod test_support;

pub use migrations::{Migration, resolve_migrations};
pub use resolve::resolve_inheritance;
