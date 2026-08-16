//! Installation identity and container launch-intent persistence.
//!
//! The owner UUID isolates this ananke instance from others sharing a
//! container runtime. It is created once (transactionally) before any
//! runtime scan or launch, and every container object is labeled with it.
//! Launch intents are the durable record written *before* invoking the
//! runtime so a crash in any window leaves enough information to discover
//! and clean up the container object at the next startup.

use ananke_errors::ExpectedError;
use rusqlite::{OptionalExtension, params};

use crate::{Database, models::ContainerLaunchIntent};

/// The single mandatory `installation_metadata` row version.
const INSTALLATION_VERSION: i64 = 1;

impl Database {
    /// Return the stable owner UUID, creating it once if absent.
    ///
    /// The create-if-missing path is transactional so two concurrent
    /// bootstraps cannot mint two owners. Returns an error (and no owner)
    /// when the stored value is unreadable or corrupt — reconciliation and
    /// launch must be disabled, never guessed.
    pub async fn ensure_owner_uuid(&self) -> Result<String, ExpectedError> {
        let conn = self.conn.lock();
        let existing: Option<String> = conn
            .query_row(
                "SELECT owner_uuid FROM installation_metadata WHERE version = ?1",
                params![INSTALLATION_VERSION],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| self.db_err(e))?;
        if let Some(uuid) = existing {
            validate_owner_uuid(&uuid)?;
            return Ok(uuid);
        }
        let uuid = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO installation_metadata (version, owner_uuid) VALUES (?1, ?2)",
            params![INSTALLATION_VERSION, uuid],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(uuid)
    }

    /// Load the stored owner UUID without creating it. `None` means the row
    /// has not been created yet; an error means the stored value is corrupt.
    pub async fn owner_uuid(&self) -> Result<Option<String>, ExpectedError> {
        let conn = self.conn.lock();
        let value: Option<String> = conn
            .query_row(
                "SELECT owner_uuid FROM installation_metadata WHERE version = ?1",
                params![INSTALLATION_VERSION],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| self.db_err(e))?;
        match value {
            Some(uuid) => {
                validate_owner_uuid(&uuid)?;
                Ok(Some(uuid))
            }
            None => Ok(None),
        }
    }

    /// Insert a durable launch intent *before* any runtime invocation.
    pub async fn insert_launch_intent(
        &self,
        row: &ContainerLaunchIntent,
    ) -> Result<i64, ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO container_launch_intents
                 (service_id, run_id, owner_uuid, workload_kind, runtime,
                  runtime_executable, container_name, labels_json, spec_json,
                  container_id, state, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                row.service_id,
                row.run_id,
                row.owner_uuid,
                row.workload_kind,
                row.runtime,
                row.runtime_executable,
                row.container_name,
                row.labels_json,
                row.spec_json,
                row.container_id,
                row.state,
                row.created_at
            ],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(conn.last_insert_rowid())
    }

    /// Attach a container ID to an existing intent after a successful create.
    pub async fn attach_container_id(
        &self,
        intent_id: i64,
        container_id: &str,
    ) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE container_launch_intents SET container_id = ?1, state = 'attached' WHERE intent_id = ?2",
            params![container_id, intent_id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Mark an intent as a reconciliation block (compensating cleanup
    /// failed, so startup reconciliation must resolve it before reprovision).
    pub async fn mark_intent_blocked(&self, intent_id: i64) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE container_launch_intents SET state = 'blocked' WHERE intent_id = ?1",
            params![intent_id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Delete an intent after its container object has been confirmed
    /// removed.
    pub async fn delete_launch_intent(&self, intent_id: i64) -> Result<(), ExpectedError> {
        let conn = self.conn.lock();
        conn.execute(
            "DELETE FROM container_launch_intents WHERE intent_id = ?1",
            params![intent_id],
        )
        .map_err(|e| self.db_err(e))?;
        Ok(())
    }

    /// Every unresolved launch intent for this owner, ordered by id.
    pub async fn list_launch_intents(&self) -> Result<Vec<ContainerLaunchIntent>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = "SELECT intent_id, service_id, run_id, owner_uuid, workload_kind, \
                   runtime, runtime_executable, container_name, labels_json, spec_json, \
                   container_id, state, created_at \
                   FROM container_launch_intents ORDER BY intent_id";
        let mut stmt = conn.prepare(sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map([], ContainerLaunchIntent::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }

    /// Every unresolved launch intent matching `service_id`, used to detect
    /// a blocked service before reprovision.
    pub async fn launch_intents_for_service(
        &self,
        service_id: i64,
    ) -> Result<Vec<ContainerLaunchIntent>, ExpectedError> {
        let conn = self.conn.lock();
        let sql = "SELECT intent_id, service_id, run_id, owner_uuid, workload_kind, \
                   runtime, runtime_executable, container_name, labels_json, spec_json, \
                   container_id, state, created_at \
                   FROM container_launch_intents WHERE service_id = ?1 ORDER BY intent_id";
        let mut stmt = conn.prepare(sql).map_err(|e| self.db_err(e))?;
        let rows = stmt
            .query_map(params![service_id], ContainerLaunchIntent::from_row)
            .map_err(|e| self.db_err(e))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| self.db_err(e))
    }
}

fn validate_owner_uuid(uuid: &str) -> Result<(), ExpectedError> {
    uuid::Uuid::parse_str(uuid).map_err(|_| {
        ExpectedError::config_unparseable(
            std::path::PathBuf::from("<installation_metadata>"),
            format!("stored owner_uuid `{uuid}` is not a valid UUID"),
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::Database;

    #[tokio::test]
    async fn owner_identity_created_once() {
        let db = Database::open_in_memory().await.unwrap();
        assert!(db.owner_uuid().await.unwrap().is_none());

        let first = db.ensure_owner_uuid().await.unwrap();
        // Minting a second owner would make every container this
        // installation already labelled look foreign to it.
        for _ in 0..3 {
            assert_eq!(db.ensure_owner_uuid().await.unwrap(), first);
        }
        assert_eq!(db.owner_uuid().await.unwrap().as_deref(), Some(&*first));
    }

    #[tokio::test]
    async fn owner_identity_survives_database_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ananke.db");

        let first = {
            let db = Database::open(&path).await.unwrap();
            db.ensure_owner_uuid().await.unwrap()
        };
        let db = Database::open(&path).await.unwrap();
        assert_eq!(db.owner_uuid().await.unwrap().as_deref(), Some(&*first));
        assert_eq!(db.ensure_owner_uuid().await.unwrap(), first);
    }

    #[tokio::test]
    async fn corrupt_owner_identity_is_an_error_not_a_new_owner() {
        let db = Database::open_in_memory().await.unwrap();
        db.ensure_owner_uuid().await.unwrap();
        {
            let conn = db.conn.lock();
            conn.execute(
                "UPDATE installation_metadata SET owner_uuid = 'not-a-uuid'",
                [],
            )
            .unwrap();
        }
        // Both paths must refuse: minting a replacement would orphan every
        // container already labelled with the real owner.
        assert!(db.owner_uuid().await.is_err());
        assert!(db.ensure_owner_uuid().await.is_err());
    }
}
