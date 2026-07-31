//! Writing JSON the way the dataset is written.
//!
//! This lives with the schema rather than with the harness because it *is* part
//! of the format. Every line already committed uses `", "` and `": "`
//! separators and escapes non-ASCII as `\uXXXX`, and the same writer produces
//! the payload a cell's identity hashes — where matching byte for byte is not
//! cosmetic but the difference between recognising the dataset and re-measuring
//! all of it.

use std::io;

use serde::Serialize;

/// Serialize the way every line already in the dataset was written: `", "` and
/// `": "` separators, and non-ASCII escaped rather than emitted raw.
///
/// Floats go through serde_json's own shortest-round-trip writer, which agrees
/// with the committed lines on every magnitude this dataset holds. The two
/// diverge only in exponent notation (`1e+22` against `1e22`), which no field
/// here reaches.
pub fn to_dataset_json<T: Serialize>(value: &T) -> String {
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, DatasetFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing to a Vec cannot fail");
    String::from_utf8(out).expect("the formatter emits ASCII only")
}

/// The dataset's spacing and escaping, which is all that separates it from
/// serde_json's compact form.
struct DatasetFormatter;

impl serde_json::ser::Formatter for DatasetFormatter {
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    /// `ensure_ascii=True` is a default it would be easy to overlook, and it
    /// changes the bytes a hash is taken over the moment a model path is not
    /// ASCII.
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if fragment.is_ascii() {
            return writer.write_all(fragment.as_bytes());
        }
        for character in fragment.chars() {
            if character.is_ascii() {
                writer.write_all(&[character as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_separators_and_ascii_escaping() {
        let value = serde_json::json!({"a": 1, "b": [2.0, "x"], "c": "caf\u{e9}"});
        // The accented byte comes back escaped, exactly as `ensure_ascii` writes
        // it, because a cell's identity is hashed over these bytes.
        assert_eq!(
            to_dataset_json(&value),
            r#"{"a": 1, "b": [2.0, "x"], "c": "caf\u00e9"}"#
        );
    }
}
