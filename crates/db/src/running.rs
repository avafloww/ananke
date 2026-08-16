//! Persistence for currently-running supervised children
//! (`running_services` table): recorded on spawn, removed on drain or
//! orphan cleanup, and replayed at startup for orphan recovery.

use ananke_errors::ExpectedError;
use rusqlite::{OptionalExtension, params};

use crate::{Database, models::RunningService};

impl Database {
    /// Insert (or upsert) a running_services row for a freshly-spawned
    /// supervisor child.
    pub async fn insert_running(&self, row: &RunningService) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO running_services
                 (service_id, run_id, pid, spawned_at, command_line, allocation, state,
                  workload_kind, runtime, container_name, container_id, runtime_executable)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                row.service_id,
                row.run_id,
                row.pid,
                row.spawned_at,
                row.command_line,
                row.allocation,
                row.state,
                row.workload_kind,
                row.runtime,
                row.container_name,
                row.container_id,
                row.runtime_executable
            ],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Delete a running_services row on drain / orphan cleanup.
    pub async fn delete_running(&self, service_id: i64, run_id: i64) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM running_services WHERE service_id = ?1 AND run_id = ?2",
            params![service_id, run_id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Every running_services row, used by orphan recovery at startup.
    pub async fn list_running(&self) -> Result<Vec<RunningService>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!("SELECT {} FROM running_services", RunningService::COLUMNS);
        let mut stmt = conn.prepare(&sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], RunningService::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// The most recently spawned running_services row for a service, if any.
    pub async fn latest_running(
        &self,
        service_id: i64,
    ) -> Result<Option<RunningService>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM running_services WHERE service_id = ?1 \
             ORDER BY spawned_at DESC LIMIT 1",
            RunningService::COLUMNS
        );
        conn.query_row(&sql, params![service_id], RunningService::from_row)
            .optional()
            .map_err(|e| self.db_err(e))
    }
}
