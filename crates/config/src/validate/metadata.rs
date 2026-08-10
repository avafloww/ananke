//! Coerce the raw `[[service]] metadata.*` TOML table into the JSON-valued
//! map the OpenAI and management APIs serve.

use std::collections::BTreeMap;

use ananke_api::shared::metadata::AnankeMetadata;

/// Convert the raw `[[service]] metadata.*` table into the JSON-valued
/// map that [`ServiceConfig`] carries through to the OpenAI and
/// management API responses. Keeps the TOML→JSON coercion in one place
/// so it matches `filters.set_params` and doesn't drift.
pub(crate) fn build_ananke_metadata(
    raw: Option<&BTreeMap<String, toml::Value>>,
) -> Result<AnankeMetadata, String> {
    let Some(m) = raw else {
        return Ok(AnankeMetadata::new());
    };
    m.iter()
        .map(|(k, v)| toml_value_to_json(v.clone()).map(|j| (k.clone(), j)))
        .collect()
}

/// Coerce one TOML value to its JSON counterpart. Untyped on both
/// sides: the operator writes these, and ananke never interprets them.
pub(crate) fn toml_value_to_json(v: toml::Value) -> Result<serde_json::Value, String> {
    Ok(match v {
        toml::Value::String(s) => serde_json::Value::String(s),
        toml::Value::Integer(i) => serde_json::Value::Number(i.into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| "non-finite float".to_string())?,
        toml::Value::Boolean(b) => serde_json::Value::Bool(b),
        toml::Value::Array(a) => serde_json::Value::Array(
            a.into_iter()
                .map(toml_value_to_json)
                .collect::<Result<_, _>>()?,
        ),
        toml::Value::Table(t) => {
            let mut m = serde_json::Map::new();
            for (k, v) in t {
                m.insert(k, toml_value_to_json(v)?);
            }
            serde_json::Value::Object(m)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    })
}
