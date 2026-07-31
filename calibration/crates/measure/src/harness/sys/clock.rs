//! Monotonic time and wall-clock timestamps.
//!
//! Behind a trait because every phase of a measurement is bounded by a deadline
//! and a poll cadence, and a test that waited those out would take the half-hour
//! a real load is allowed. The fake advances virtual time on each sleep, so a
//! load-timeout test costs microseconds and still exercises the same arithmetic.

use std::time::Duration;

/// The one wall-clock format the records use: ISO 8601 to the second, with an
/// explicit offset, as every row already in the dataset carries.
const STAMP: &str = "%Y-%m-%dT%H:%M:%S%:z";

pub trait Clock: Send + Sync {
    /// Monotonic time since the clock was created. A measurement only ever
    /// wants differences, so there is no absolute monotonic instant to leak.
    fn elapsed(&self) -> Duration;
    /// UTC, for the field every analysis sorts on.
    fn now_utc(&self) -> String;
    /// The same instant in the box's own zone, so a row can be placed against
    /// whatever else the operator remembers about that evening.
    fn now_local(&self) -> String;
    fn sleep(&self, duration: Duration);
}

pub struct SystemClock {
    started: std::time::Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            started: std::time::Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    fn now_utc(&self) -> String {
        jiff::Timestamp::now()
            .to_zoned(jiff::tz::TimeZone::UTC)
            .strftime(STAMP)
            .to_string()
    }

    fn now_local(&self) -> String {
        jiff::Zoned::now().strftime(STAMP).to_string()
    }

    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// Virtual time: `sleep` advances it instead of waiting.
///
/// The wall-clock stamps advance with it too, so a record assembled under the
/// fake carries timestamps that move the way real ones would rather than a
/// single frozen instant.
#[cfg(any(test, feature = "test-fakes"))]
pub struct FakeClock {
    elapsed: parking_lot::Mutex<Duration>,
    base: jiff::Timestamp,
}

#[cfg(any(test, feature = "test-fakes"))]
impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl FakeClock {
    /// An arbitrary but fixed epoch, so an assertion can spell the timestamp it
    /// expects.
    pub fn new() -> Self {
        Self {
            elapsed: parking_lot::Mutex::new(Duration::ZERO),
            base: "2026-01-02T03:04:05Z"
                .parse()
                .expect("the fake clock's epoch is a literal"),
        }
    }

    pub fn advance(&self, by: Duration) {
        *self.elapsed.lock() += by;
    }

    fn instant(&self) -> jiff::Timestamp {
        let seconds = i64::try_from(self.elapsed.lock().as_secs()).unwrap_or(i64::MAX);
        self.base + jiff::SignedDuration::from_secs(seconds)
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl Clock for FakeClock {
    fn elapsed(&self) -> Duration {
        *self.elapsed.lock()
    }

    fn now_utc(&self) -> String {
        self.instant()
            .to_zoned(jiff::tz::TimeZone::UTC)
            .strftime(STAMP)
            .to_string()
    }

    fn now_local(&self) -> String {
        // Deliberately UTC as well: a fake that read the box's zone would make
        // an assertion on this field depend on where the suite runs.
        self.now_utc()
    }

    fn sleep(&self, duration: Duration) {
        self.advance(duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sleeping_advances_virtual_time_and_the_stamps_with_it() {
        let clock = FakeClock::new();
        assert_eq!(clock.now_utc(), "2026-01-02T03:04:05+00:00");
        clock.sleep(Duration::from_secs(90));
        assert_eq!(clock.elapsed(), Duration::from_secs(90));
        assert_eq!(clock.now_utc(), "2026-01-02T03:05:35+00:00");
    }
}
