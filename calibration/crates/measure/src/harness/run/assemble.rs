//! Turn what a run yielded into the record that is written.
//!
//! Pure, and separated from the run for that reason: the summary is where a
//! measurement becomes a number an analysis will fit against, and getting the
//! wrong quantity in `rss_anon_kb` is not something a later reader can detect.
//!
//! The peak is reported rather than the final reading, because the peak is the
//! figure ananke's snapshotter keeps and the rolling correction divides by. The
//! final reading and the growth since startup ride alongside it, under their own
//! prefixes, so a reader that wants "where did it end up" is not left inferring
//! it from a peak.

use crate::{
    harness::run::sampler::Sampler,
    parse::Parsed,
    record::{
        Checkpoint, Factors, Hardware, Provenance, Record, Rss, RssSnapshot, SCHEMA, Sample, Status,
    },
};

/// Everything one run produced, ready to be recorded.
pub(crate) struct Outcome {
    pub(crate) status: Status,
    pub(crate) parsed: Parsed,
    pub(crate) rss: Rss,
    /// The end of a failed run's log, so a bad record says why it is bad.
    pub(crate) log_tail: String,
    /// The archived log's file name.
    pub(crate) log: String,
    pub(crate) trace: Vec<Sample>,
    pub(crate) checkpoints: Vec<Checkpoint>,
}

impl Outcome {
    /// A run that produced no measurement: a skip, a load failure, a timeout.
    /// The factors, the provenance, and the hardware are still recorded, because
    /// a cell that could not be measured on this box is itself a finding.
    pub(crate) fn failed(status: Status) -> Self {
        Self {
            status,
            parsed: Parsed::default(),
            rss: Rss::default(),
            log_tail: String::new(),
            log: String::new(),
            trace: Vec::new(),
            checkpoints: Vec::new(),
        }
    }
}

pub(crate) fn record(
    cell: String,
    factors: Factors,
    provenance: Provenance,
    hardware: Hardware,
    outcome: Outcome,
) -> Record {
    Record {
        schema: SCHEMA,
        cell,
        status: outcome.status,
        provenance,
        hardware,
        factors,
        parsed: outcome.parsed,
        rss: outcome.rss,
        log_tail: outcome.log_tail,
        log: outcome.log,
        trace: outcome.trace,
        checkpoints: outcome.checkpoints,
        // Only a `--reparse` pass sets this; a fresh row's `parsed` came from the
        // parser that is running.
        reparsed: None,
    }
}

/// The `rss` block: peaks, the final reading, the growth, and how the run went.
///
/// `final_reading` stands in for the peaks when the sampler never got a sample —
/// a load that failed before `/proc` could be read twice — so the block is never
/// silently empty when something was in fact measured.
pub(crate) fn rss_summary(sampler: &Sampler, final_reading: RssSnapshot, load_seconds: f64) -> Rss {
    let mut rss = if sampler.is_empty() {
        Rss {
            rss_total_kb: whole(final_reading.rss_total_kb),
            rss_anon_kb: whole(final_reading.rss_anon_kb),
            rss_file_kb: whole(final_reading.rss_file_kb),
            rss_shmem_kb: whole(final_reading.rss_shmem_kb),
            ..Rss::default()
        }
    } else {
        sampler.summary()
    };
    rss.final_rss_total_kb = whole(final_reading.rss_total_kb);
    rss.final_rss_anon_kb = whole(final_reading.rss_anon_kb);
    rss.final_rss_file_kb = whole(final_reading.rss_file_kb);
    rss.final_rss_shmem_kb = whole(final_reading.rss_shmem_kb);
    rss.samples = i64::try_from(sampler.samples()).unwrap_or(i64::MAX);
    // A tenth of a second, as every recorded row spells it.
    rss.load_seconds = (load_seconds * 10.0).round() / 10.0;
    rss
}

/// A `/proc` counter as the record spells it: signed, because the growth beside
/// it is a difference.
fn whole(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// The end of a log, for a record that has to say why it is bad.
pub(crate) fn tail(log: &str, lines: usize) -> String {
    let all: Vec<&str> = log.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn snapshot(anon: u64, shmem: u64, file: u64) -> RssSnapshot {
        RssSnapshot {
            rss_total_kb: anon + shmem + file,
            rss_anon_kb: anon,
            rss_file_kb: file,
            rss_shmem_kb: shmem,
        }
    }

    #[test]
    fn the_summary_reports_the_peak_with_the_final_reading_beside_it() {
        let mut sampler = Sampler::default();
        sampler.observe(0.0, "t0".to_owned(), snapshot(1_000, 0, 0), BTreeMap::new());
        sampler.observe(
            2.0,
            "t1".to_owned(),
            snapshot(9_000, 400, 0),
            BTreeMap::new(),
        );
        let rss = rss_summary(&sampler, snapshot(5_000, 400, 0), 41.44);

        assert_eq!(rss.rss_anon_kb, 9_000, "the peak, not the end");
        assert_eq!(rss.final_rss_anon_kb, 5_000);
        assert_eq!(rss.growth_rss_anon_kb, 8_000);
        assert_eq!(rss.samples, 2);
        assert_eq!(rss.load_seconds, 41.4);
    }

    /// A run whose process vanished before a second sample still has one reading
    /// worth recording, and an empty `rss` block would read as "not measured".
    #[test]
    fn a_run_with_no_samples_falls_back_to_the_final_reading() {
        let rss = rss_summary(&Sampler::default(), snapshot(2_000, 8, 100), 3.0);
        assert_eq!(rss.rss_anon_kb, 2_000);
        assert_eq!(rss.final_rss_anon_kb, 2_000);
        assert_eq!(rss.samples, 0);
        // Nothing was sampled, so nothing grew.
        assert_eq!(rss.growth_rss_anon_kb, 0);
    }

    #[test]
    fn a_failed_outcome_still_records_the_factors_and_the_status() {
        let record = record(
            "abc123456789".to_owned(),
            Factors {
                label: "glm-2card".to_owned(),
                ..Factors::default()
            },
            Provenance {
                host: "redline".to_owned(),
                ..Provenance::default()
            },
            Hardware::default(),
            Outcome::failed(Status::Timeout),
        );
        assert_eq!(record.status, Status::Timeout);
        assert_eq!(record.factors.label, "glm-2card");
        assert_eq!(record.schema, SCHEMA);
        assert!(record.reparsed.is_none());
        // Serialisable as one line, which is the only thing the writer asks of it.
        let line = serde_json::to_string(&record).expect("a record serializes");
        assert!(line.contains(r#""status":"timeout""#), "{line}");
    }

    #[test]
    fn the_tail_is_the_end_of_the_log_and_survives_a_short_one() {
        let log = (1..=100)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(tail(&log, 40).starts_with("61\n62"));
        assert_eq!(tail("only one line", 40), "only one line");
        assert_eq!(tail("", 40), "");
    }
}
