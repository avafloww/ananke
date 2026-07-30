//! llama.cpp's own memory-breakdown table, per device and for the host.

use serde::{Serialize, Serializer, ser::SerializeMap};

use crate::parse::{count, patterns};

/// How many device rows the flat mirrors below cover.
pub const MAX_GPUS: usize = 4;

/// One device's row, with every column.
///
/// `unaccounted_mib` is the difference between what the driver reports for the
/// process and what llama.cpp can attribute — the term the GPU compute-buffer
/// bases carry as a margin.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DeviceRow {
    pub device: String,
    pub total_mib: u64,
    pub free_mib: u64,
    pub self_mib: u64,
    pub model_mib: u64,
    pub kv_mib: u64,
    pub compute_mib: u64,
    pub unaccounted_mib: u64,
}

/// The host row, which has no total/free and no unaccounted column.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HostBreakdown {
    pub self_mib: u64,
    pub model_mib: u64,
    pub kv_mib: u64,
    pub compute_mib: u64,
}

/// Every device row of the table, in the order the loader printed them.
///
/// Only the *last* table is read. Recent builds print one at the
/// parameter-fitting stage, before anything is allocated, and another once the
/// context exists; the first is a projection and its rows would misalign
/// `devices[index]` against the cards. Relying on the fit-stage rows' negative
/// `unaccounted` to exclude them worked by accident and stopped working for 16
/// cells whose projection happened to come out positive.
pub(crate) fn parse_devices(text: &str) -> Vec<DeviceRow> {
    patterns::BREAKDOWN
        .captures_iter(final_table(text))
        .map(|caps| DeviceRow {
            device: caps
                .get(1)
                .map(|found| found.as_str().trim().to_owned())
                .unwrap_or_default(),
            total_mib: count(&caps, 2),
            free_mib: count(&caps, 3),
            self_mib: count(&caps, 4),
            model_mib: count(&caps, 5),
            kv_mib: count(&caps, 6),
            compute_mib: count(&caps, 7),
            unaccounted_mib: count(&caps, 8),
        })
        .collect()
}

pub(crate) fn parse_host(text: &str) -> Option<HostBreakdown> {
    let caps = patterns::BREAKDOWN_HOST.captures(final_table(text))?;
    Some(HostBreakdown {
        self_mib: count(&caps, 1),
        model_mib: count(&caps, 2),
        kv_mib: count(&caps, 3),
        compute_mib: count(&caps, 4),
    })
}

/// Flat mirrors of the first `MAX_GPUS` device rows.
///
/// Kept because they are convenient to fit against; `Parsed::devices` is the
/// authoritative list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuMirrors {
    rows: [DeviceRow; MAX_GPUS],
}

impl GpuMirrors {
    pub(crate) fn from_devices(devices: &[DeviceRow]) -> Self {
        let mut rows: [DeviceRow; MAX_GPUS] = Default::default();
        for (mirror, device) in rows.iter_mut().zip(devices) {
            *mirror = device.clone();
        }
        Self { rows }
    }
}

impl Serialize for GpuMirrors {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for (index, row) in self.rows.iter().enumerate() {
            for (column, value) in [
                ("model", row.model_mib),
                ("kv", row.kv_mib),
                ("compute", row.compute_mib),
                ("unaccounted", row.unaccounted_mib),
                ("self", row.self_mib),
            ] {
                map.serialize_entry(&format!("gpu{index}_{column}_mib"), &value)?;
            }
        }
        map.end()
    }
}

/// The table the context's teardown printed, which is the only one whose rows
/// describe allocations that happened.
fn final_table(text: &str) -> &str {
    text.rsplit_once(BREAKDOWN_HEADING)
        .map(|(_, last)| last)
        .unwrap_or(text)
}

const BREAKDOWN_HEADING: &str = "memory breakdown [MiB]";

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-aligned table, with the second card's free figure one digit
    /// narrower than the first's.
    const TABLE: &str = "\
| memory breakdown [MiB] | total    free    self   model   context   compute    unaccounted |
|   - CUDA0 (RTX 3090)   | 24576 = 11121 + (13424 = 11876 +     516 +    1032) +          30 |
|   - CUDA1 (RTX 3090)   | 24576 =  9877 + (14668 = 13120 +     516 +    1032) +          24 |
|   - Host               |                   316 =   304 +       0 +      12                |
";

    #[test]
    fn padding_does_not_drop_a_card() {
        let devices = parse_devices(TABLE);
        assert_eq!(devices.len(), 2, "{devices:?}");
        assert_eq!(devices[0].device, "CUDA0 (RTX 3090)");
        assert_eq!(devices[1].free_mib, 9877);
        assert_eq!(devices[1].compute_mib, 1032);
        assert_eq!(devices[1].unaccounted_mib, 24);
    }

    #[test]
    fn only_the_final_table_is_read() {
        // The parameter-fitting stage prints a projection before anything is
        // allocated, and its rows would misalign `devices[index]` against the
        // cards.
        let projection = TABLE.replace("1032", "2042");
        let devices = parse_devices(&format!("{projection}some other output\n{TABLE}"));
        assert_eq!(devices.len(), 2, "{devices:?}");
        assert!(
            devices.iter().all(|row| row.compute_mib == 1032),
            "{devices:?}"
        );
    }

    #[test]
    fn the_host_row_carries_no_total() {
        assert_eq!(
            parse_host(TABLE),
            Some(HostBreakdown {
                self_mib: 316,
                model_mib: 304,
                kv_mib: 0,
                compute_mib: 12,
            })
        );
    }

    #[test]
    fn mirrors_cover_the_first_cards_and_zero_the_rest() {
        let mirrors = GpuMirrors::from_devices(&parse_devices(TABLE));
        let json = serde_json::to_value(&mirrors).expect("mirrors serialize as a map");
        assert_eq!(json["gpu1_model_mib"], 13120);
        assert_eq!(json["gpu3_compute_mib"], 0);
    }
}
