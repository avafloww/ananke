//! Persistence for oneshot request submissions: idempotent insert on
//! submission, stamped end time on completion.

use ananke_errors::ExpectedError;
use rusqlite::params;

use crate::Database;

impl Database {
    /// Record a oneshot's submission. Idempotent on the id — repeated
    /// calls with the same id are a no-op.
    pub async fn insert_oneshot(
        &self,
        id: &str,
        service_id: i64,
        submitted_at: i64,
        ttl_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR IGNORE INTO oneshots(id, service_id, submitted_at, ttl_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![id, service_id, submitted_at, ttl_ms],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Stamp a oneshot as finished. The row stays in place so status
    /// polls keep seeing the terminal state until it falls out of
    /// retention.
    pub async fn mark_oneshot_ended(
        &self,
        id: &str,
        ended_at_ms: i64,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE oneshots SET ended_at = ?1 WHERE id = ?2",
            params![ended_at_ms, id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }
}
