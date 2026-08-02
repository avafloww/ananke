//! llama.cpp's own memory-breakdown table, per device and for the host.

use ananke_dataset::{DeviceRow, HostBreakdown, Parsed};

use crate::parse::{count, patterns};

/// Every device row of the table, in the order the loader printed them.
///
/// Only the *last* table is read. Recent builds print one at the
/// parameter-fitting stage, before anything is allocated, and another once the
/// context exists; the first is a projection and its rows would misalign
/// `devices[index]` against the cards. The fit-stage rows cannot be excluded by
/// a negative `unaccounted` instead: 16 cells' projections come out positive.
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

/// Mirror the first four device rows into the record's flat `gpu{n}_*` fields.
///
/// Redundant with `Parsed::devices`, which is authoritative, and kept only
/// because the flat columns are convenient to fit against. Closed at four
/// cards, because the schema names exactly that many; a fifth card is a schema
/// change rather than something to absorb here.
pub(crate) fn fill_mirrors(devices: &[DeviceRow], parsed: &mut Parsed) {
    let row = |index: usize| devices.get(index).cloned().unwrap_or_default();
    let (first, second, third, fourth) = (row(0), row(1), row(2), row(3));

    parsed.gpu0_model_mib = first.model_mib;
    parsed.gpu0_kv_mib = first.kv_mib;
    parsed.gpu0_compute_mib = first.compute_mib;
    parsed.gpu0_unaccounted_mib = first.unaccounted_mib;
    parsed.gpu0_self_mib = first.self_mib;

    parsed.gpu1_model_mib = second.model_mib;
    parsed.gpu1_kv_mib = second.kv_mib;
    parsed.gpu1_compute_mib = second.compute_mib;
    parsed.gpu1_unaccounted_mib = second.unaccounted_mib;
    parsed.gpu1_self_mib = second.self_mib;

    parsed.gpu2_model_mib = third.model_mib;
    parsed.gpu2_kv_mib = third.kv_mib;
    parsed.gpu2_compute_mib = third.compute_mib;
    parsed.gpu2_unaccounted_mib = third.unaccounted_mib;
    parsed.gpu2_self_mib = third.self_mib;

    parsed.gpu3_model_mib = fourth.model_mib;
    parsed.gpu3_kv_mib = fourth.kv_mib;
    parsed.gpu3_compute_mib = fourth.compute_mib;
    parsed.gpu3_unaccounted_mib = fourth.unaccounted_mib;
    parsed.gpu3_self_mib = fourth.self_mib;
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
        let mut parsed = Parsed::default();
        fill_mirrors(&parse_devices(TABLE), &mut parsed);
        assert_eq!(parsed.gpu1_model_mib, 13120);
        assert_eq!(parsed.gpu3_compute_mib, 0);
    }
}
