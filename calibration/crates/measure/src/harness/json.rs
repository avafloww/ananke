//! JSON text mechanics: finding one member of a record without disturbing the
//! rest.
//!
//! The dataset is the campaign's oracle and it is checked in, so a maintenance
//! pass over it must change only what it means to change — a status, or a
//! `parsed` block that the parser now reads differently. Re-serialising a whole
//! record would rewrite every line's key order and spacing, turning a one-field
//! edit into a five-megabyte diff and destroying the only way to see what a pass
//! actually did.
//!
//! So a rewrite splices the new value into the original bytes ([`member_span`]),
//! and anything newly written goes through [`to_dataset_json`], which lives with
//! the schema because it *is* part of the format.

use std::ops::Range;

pub(crate) use ananke_dataset::to_dataset_json;

/// The byte range of one top-level member's *value* in a JSON object.
///
/// A textual search for `"parsed":` would also find the string inside a retained
/// log tail, so the object is walked properly: keys are read as JSON strings and
/// values are skipped with brace, bracket, and quote nesting respected.
pub(crate) fn member_span(text: &str, key: &str) -> Option<Range<usize>> {
    let bytes = text.as_bytes();
    let mut at = skip_space(bytes, 0);
    if bytes.get(at) != Some(&b'{') {
        return None;
    }
    at += 1;
    loop {
        at = skip_space(bytes, at);
        if bytes.get(at) != Some(&b'"') {
            return None;
        }
        let name = string_span(bytes, at)?;
        at = skip_space(bytes, name.end + 1);
        if bytes.get(at) != Some(&b':') {
            return None;
        }
        at = skip_space(bytes, at + 1);
        let end = skip_value(bytes, at)?;
        // Keys in these records are plain, so a literal comparison is enough and
        // avoids unescaping a name only to throw it away.
        if &text[name.clone()] == key {
            return Some(at..end);
        }
        at = skip_space(bytes, end);
        match bytes.get(at) {
            Some(b',') => at += 1,
            _ => return None,
        }
    }
}

/// Replace one top-level member's value, leaving every other byte alone.
pub(crate) fn splice_member(text: &str, key: &str, value: &str) -> Option<String> {
    let span = member_span(text, key)?;
    let mut out = String::with_capacity(text.len() + value.len());
    out.push_str(&text[..span.start]);
    out.push_str(value);
    out.push_str(&text[span.end..]);
    Some(out)
}

fn skip_space(bytes: &[u8], mut at: usize) -> usize {
    while matches!(bytes.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        at += 1;
    }
    at
}

/// The span *inside* the quotes of the string starting at `at`.
fn string_span(bytes: &[u8], at: usize) -> Option<Range<usize>> {
    let end = string_end(bytes, at)?;
    Some(at + 1..end - 1)
}

/// One past the closing quote of the string starting at `at`.
fn string_end(bytes: &[u8], at: usize) -> Option<usize> {
    let mut index = at + 1;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'\\' => index += 2,
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

/// One past the end of the JSON value starting at `at`.
fn skip_value(bytes: &[u8], at: usize) -> Option<usize> {
    match bytes.get(at)? {
        b'"' => string_end(bytes, at),
        open @ (b'{' | b'[') => {
            let close = if *open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            let mut index = at;
            while let Some(&byte) = bytes.get(index) {
                match byte {
                    b'"' => {
                        index = string_end(bytes, index)?;
                        continue;
                    }
                    byte if byte == *open => depth += 1,
                    byte if byte == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(index + 1);
                        }
                    }
                    _ => {}
                }
                index += 1;
            }
            None
        }
        // A number, or one of the three literals: everything up to the next
        // structural byte.
        _ => {
            let mut index = at;
            while let Some(&byte) = bytes.get(index) {
                if matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }
                index += 1;
            }
            (index > at).then_some(index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_member_is_found_past_a_decoy_in_a_log_tail() {
        // The hazard this scanner exists for: the key spelled inside a string
        // value earlier in the record.
        let line = r#"{"log_tail": "warn: \"parsed\": {\"a\": 1} was odd", "parsed": {"x": [1, {"y": 2}]}, "z": 3}"#;
        let span = member_span(line, "parsed").expect("the member is there");
        assert_eq!(&line[span], r#"{"x": [1, {"y": 2}]}"#);
    }

    #[test]
    fn splicing_leaves_every_other_byte_alone() {
        let line = r#"{"status": "ok", "cell": "abc", "n": 1}"#;
        assert_eq!(
            splice_member(line, "status", "\"stale-runtime\"").expect("status is a member"),
            r#"{"status": "stale-runtime", "cell": "abc", "n": 1}"#
        );
        assert_eq!(
            splice_member(line, "n", "2").expect("a bare number is a value too"),
            r#"{"status": "ok", "cell": "abc", "n": 2}"#
        );
        assert_eq!(splice_member(line, "absent", "1"), None);
    }
}
