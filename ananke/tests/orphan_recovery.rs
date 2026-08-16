//! Integration test: orphan recovery cleans stale rows for non-existent PIDs.
#![cfg(feature = "test-fakes")]

mod common;

use ananke::supervise::{OrphanDisposition, reconcile};
use ananke_db::{Database, models::RunningService};
use ananke_system::{InMemoryProcFs, container::FakeContainerEngine};

#[tokio::test]
async fn cleans_row_for_dead_pid() {
    let db = Database::open_in_memory().await.expect("open db");

    // Register a service so the foreign key is satisfied.
    let service_id = db
        .upsert_service("test-svc", 0)
        .await
        .expect("upsert_service");

    // Insert a running_services row pointing at PID 99999. The empty
    // InMemoryProcFs reports that pid as exited, so reconcile must
    // clean the row.
    db.insert_running(&RunningService {
        service_id,
        run_id: 1,
        pid: Some(99999),
        spawned_at: 0,
        command_line: "fake-server".to_string(),
        allocation: "{}".to_string(),
        state: "running".to_string(),
        workload_kind: Some("process".to_string()),
        runtime: None,
        container_name: None,
        container_id: None,
    })
    .await
    .expect("insert row");

    let proc = InMemoryProcFs::new();
    let engine = FakeContainerEngine::new();
    let dispositions = reconcile(&proc, &engine, None, &db).await;

    assert_eq!(dispositions.len(), 1);
    assert!(
        matches!(
            dispositions[0],
            OrphanDisposition::Cleaned { pid: 99999, .. }
        ),
        "expected Cleaned, got {:?}",
        dispositions[0]
    );

    let remaining = db.list_running().await.expect("query running_services");
    assert!(remaining.is_empty(), "stale row should have been deleted");
}
