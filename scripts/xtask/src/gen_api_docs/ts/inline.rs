//! Compact single-line TypeScript type generation, used for `oneOf` union
//! members where the multi-line form of `ts_type` would be too verbose.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::gen_api_docs::ts::{get_types, is_null_type};

/// Generate a compact single-line TypeScript type (for `oneOf` variants and
/// other inline contexts). Never expands `$ref`s — always uses the type
/// name (to avoid duplicating large schemas in union members).
pub fn ts_inline(schema: &Value, spec: &Value, visited: &mut BTreeSet<String>) -> String {
    // $ref → type name only.
    if let Some(ref_str) = schema.get("$ref").and_then(|r| r.as_str()) {
        return ref_str
            .strip_prefix("#/components/schemas/")
            .unwrap_or(ref_str)
            .to_string();
    }

    // allOf → merge and format inline.
    if let Some(all_of) = schema.get("allOf").and_then(|a| a.as_array()) {
        let mut merged: BTreeMap<String, &Value> = BTreeMap::new();
        let mut required: BTreeSet<String> = BTreeSet::new();
        let mut has_passthrough = false;
        for sub in all_of {
            if let Some(props) = sub.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in props {
                    merged.insert(k.clone(), v);
                }
            }
            if let Some(req) = sub.get("required").and_then(|r| r.as_array()) {
                for r in req {
                    if let Some(s) = r.as_str() {
                        required.insert(s.to_string());
                    }
                }
            }
            if sub.get("additionalProperties").and_then(|v| v.as_bool()) == Some(true) {
                has_passthrough = true;
            }
        }
        if merged.is_empty() {
            return "Record<string, any>".to_string();
        }
        let mut obj = format_ts_inline_object(&merged, &required, spec, visited);
        if has_passthrough {
            obj = obj.replace(" }", ", ...any }");
        }
        return obj;
    }

    // oneOf / anyOf.
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(|o| o.as_array()) {
            let non_null: Vec<&Value> = variants.iter().filter(|v| !is_null_type(v)).collect();
            let has_null = non_null.len() < variants.len();
            if non_null.is_empty() {
                return "null".to_string();
            }
            if non_null.len() == 1 && has_null {
                let inner = ts_inline(non_null[0], spec, visited);
                return format!("{} | null", inner);
            }
            let parts: Vec<String> = non_null
                .iter()
                .map(|v| ts_inline(v, spec, visited))
                .collect();
            return parts.join(" | ");
        }
    }

    let (types, has_null) = get_types(schema);
    let non_null_types: Vec<&str> = types
        .iter()
        .map(|s| s.as_str())
        .filter(|s| *s != "null")
        .collect();
    let primary = non_null_types.first().copied();

    let base = match primary {
        Some("object") => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                if !props.is_empty() {
                    let required: BTreeSet<String> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    let props_map: BTreeMap<String, &Value> =
                        props.iter().map(|(k, v)| (k.clone(), v)).collect();
                    format_ts_inline_object(&props_map, &required, spec, visited)
                } else {
                    "Record<string, any>".to_string()
                }
            } else {
                "Record<string, any>".to_string()
            }
        }
        Some("array") => {
            if let Some(items) = schema.get("items") {
                let item_type = ts_inline(items, spec, visited);
                format!("{}[]", item_type)
            } else {
                "any[]".to_string()
            }
        }
        Some("string") => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "string".to_string()
            }
        }
        Some("integer") | Some("number") => "number".to_string(),
        Some("boolean") => "boolean".to_string(),
        Some(other) => other.to_string(),
        None => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else {
                "any".to_string()
            }
        }
    };

    if has_null {
        format!("{} | null", base)
    } else {
        base
    }
}

/// Format an object as a single-line TypeScript literal: `{ a: T, b: U }`.
fn format_ts_inline_object(
    props: &BTreeMap<String, &Value>,
    required: &BTreeSet<String>,
    spec: &Value,
    visited: &mut BTreeSet<String>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut sorted: Vec<_> = props.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    for (field, field_schema) in sorted {
        let field_type = ts_inline(field_schema, spec, visited);
        let is_required = required.contains(field.as_str());
        let field_name = if is_required {
            field.to_string()
        } else {
            format!("{}?", field)
        };
        parts.push(format!("{}: {}", field_name, field_type));
    }
    format!("{{ {} }}", parts.join(", "))
}
