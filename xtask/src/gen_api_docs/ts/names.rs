//! Human-readable TypeScript type names for a schema, never expanding
//! `$ref`s. Used in field tables and response columns where a full type
//! expansion would be too noisy.

use serde_json::Value;

/// Extract a human-readable, TS-style type name from a schema. Used in
/// field tables and response columns. Never expands `$ref`s.
pub fn schema_ref_name(schema: &Value) -> String {
    // $ref
    if let Some(r#ref) = schema.get("$ref").and_then(|r| r.as_str()) {
        return r#ref
            .strip_prefix("#/components/schemas/")
            .unwrap_or(r#ref)
            .to_string();
    }

    // oneOf / anyOf — nullable ref or tagged union.
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(|o| o.as_array()) {
            let refs: Vec<_> = variants
                .iter()
                .filter_map(|v| v.get("$ref").and_then(|r| r.as_str()))
                .map(|r| {
                    r.strip_prefix("#/components/schemas/")
                        .unwrap_or(r)
                        .to_string()
                })
                .collect();
            let has_null = variants
                .iter()
                .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("null"));
            if refs.len() == 1 && has_null {
                return format!("{} | null", refs[0]);
            }
            if !refs.is_empty() {
                return refs.join(" | ");
            }
            let non_null: Vec<_> = variants
                .iter()
                .filter(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"))
                .collect();
            if non_null.len() == 1 {
                return format!("{} | null", schema_ref_name(non_null[0]));
            }
            return "oneOf".to_string();
        }
    }

    // allOf
    if schema.get("allOf").is_some() {
        return "object".to_string();
    }

    // type as string
    if let Some(ty) = schema.get("type").and_then(|t| t.as_str()) {
        return ts_primitive_name(ty, schema);
    }

    // type as array (nullable) — e.g. ["integer", "null"]
    if let Some(types) = schema.get("type").and_then(|t| t.as_array()) {
        let non_null: Vec<_> = types
            .iter()
            .filter_map(|v| v.as_str())
            .filter(|s| *s != "null")
            .collect();
        let has_null = types.iter().any(|t| t.as_str() == Some("null"));
        if let Some(first) = non_null.first() {
            let base = ts_primitive_name(first, schema);
            return if has_null {
                format!("{} | null", base)
            } else {
                base
            };
        }
        return "null".to_string();
    }

    // No type — maybe an enum.
    if schema.get("enum").is_some() {
        return "string".to_string();
    }

    "any".to_string()
}

/// Map an OpenAPI primitive type to a TypeScript type name.
fn ts_primitive_name(ty: &str, schema: &Value) -> String {
    match ty {
        "array" => {
            if let Some(items) = schema.get("items") {
                format!("{}[]", schema_ref_name(items))
            } else {
                "any[]".to_string()
            }
        }
        "object" => "object".to_string(),
        "string" => {
            if let Some(enums) = schema.get("enum").and_then(|e| e.as_array()) {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|v| format!("\"{}\"", v.as_str().unwrap_or("")))
                    .collect();
                vals.join(" | ")
            } else if let Some(format) = schema.get("format").and_then(|f| f.as_str()) {
                format.to_string()
            } else {
                "string".to_string()
            }
        }
        "integer" | "number" => "number".to_string(),
        "boolean" => "boolean".to_string(),
        other => other.to_string(),
    }
}
