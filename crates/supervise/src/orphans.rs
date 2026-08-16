//! Linux-only: startup orphan recovery. Reads
//! `/proc/{pid}/cmdline` (through [`ananke_system::ProcFs`] — tests can
//! substitute [`ananke_system::InMemoryProcFs`] with preloaded cmdlines)
//! to decide whether a previously-recorded child is still alive and
//! still ours.

use std::collections::BTreeSet;

use ananke_db::Database;
use ananke_errors::ExpectedError;
use ananke_system::ProcFs;
use tracing::{info, warn};

/// Ownership label every ananke-managed container carries. Cleanup requires
/// it to match this installation's UUID; a name alone is never enough.
const OWNER_LABEL: &str = "io.ananke.owner";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrphanDisposition {
    Adopted {
        pid: i32,
        service_id: i64,
        run_id: i64,
    },
    Cleaned {
        pid: i32,
        service_id: i64,
        run_id: i64,
    },
    Ignored {
        pid: i32,
        reason: String,
    },
    /// A container row was reconciled (adopted or cleaned) without a host PID.
    Container {
        container_id: Option<String>,
        container_name: Option<String>,
        service_id: i64,
        run_id: i64,
        action: &'static str,
    },
}

/// Runs orphan recovery against the `running_services` table, dispatching
/// on workload kind. Native process rows use the existing PID/cmdline logic;
/// container rows are reconciled via the container engine (inspect by ID or
/// name, then verify ownership labels before removing). The top-level
/// [`reconcile`] calls both paths in sequence.
pub async fn reconcile(
    proc: &dyn ProcFs,
    engine: &dyn ananke_system::container::ContainerEngine,
    owner_uuid: Option<&str>,
    db: &Database,
) -> Vec<OrphanDisposition> {
    let rows = db.list_running().await.unwrap_or_default();

    let mut out = Vec::with_capacity(rows.len());
    // Split process vs container rows so each path is a private helper.
    let mut container_rows = Vec::new();
    let mut process_rows = Vec::new();
    for row in rows {
        let service_id = row.service_id;
        let run_id = row.run_id;
        if row.workload_kind.as_deref() == Some("container") {
            container_rows.push((service_id, run_id, row));
        } else {
            let pid = row.pid;
            process_rows.push((service_id, run_id, pid, row.command_line));
        }
    }

    // Process reconciliation is unchanged.
    for (service_id, run_id, pid, recorded_cmdline) in process_rows {
        reconcile_process_row(
            proc,
            db,
            service_id,
            run_id,
            pid,
            recorded_cmdline,
            &mut out,
        )
        .await;
    }

    // Container reconciliation requires a valid owner UUID. A missing or
    // corrupt owner disables the scan: emit a disposition that preserves the
    // row so the service is blocked, never blindly removed.
    let Some(owner) = owner_uuid else {
        for (service_id, run_id, _row) in container_rows {
            out.push(OrphanDisposition::Container {
                container_id: None,
                container_name: None,
                service_id,
                run_id,
                action: "blocked-no-owner",
            });
        }
        return out;
    };

    let mut handled: BTreeSet<(i64, i64)> = BTreeSet::new();
    // Container objects some record already accounts for, whatever the
    // outcome. The sweep below skips these: one held as a reconciliation
    // block is deliberately still there, not leaked.
    let mut resolved: BTreeSet<String> = BTreeSet::new();
    let ctx = Reconciler { engine, owner, db };
    for (service_id, run_id, row) in container_rows {
        handled.insert((service_id, run_id));
        reconcile_container_row(&ctx, service_id, run_id, row, &mut resolved, &mut out).await;
    }

    // Launch intents cover the windows a running row cannot: a crash before
    // create leaves an intent and no object, and a crash between create and
    // the row commit leaves an object with no row. Runs already reconciled
    // above are skipped — the row was the authority for those.
    reconcile_intents(&ctx, &handled, &mut resolved, &mut out).await;

    // Backstop for containers this installation owns but has no record of
    // at all — a leak from a path that dropped its row and its intent.
    // Scoped to our own owner label, so it can only ever see our own.
    sweep_owned_orphans(&ctx, &resolved, &mut out).await;

    out
}

/// Remove containers carrying this installation's owner label that no
/// remaining record accounts for.
///
/// Runs last, so anything a row or intent explains has already been cleaned
/// up and is gone from the listing. What is left is ours by label and
/// unaccounted for by construction, which is the only case where removing on
/// a label alone is safe.
async fn sweep_owned_orphans(
    ctx: &Reconciler<'_>,
    resolved: &BTreeSet<String>,
    out: &mut Vec<OrphanDisposition>,
) {
    for removal in remove_owned_containers(ctx.engine, ctx.owner, resolved).await {
        warn!(
            container = %removal.name,
            removed = removal.removed,
            "removed a container with no remaining record of it"
        );
        out.push(OrphanDisposition::Container {
            container_id: Some(removal.id),
            container_name: Some(removal.name),
            service_id: 0,
            run_id: 0,
            action: if removal.removed {
                "leaked-removed"
            } else {
                "leaked-remove-failed"
            },
        });
    }
}

/// One container the owner-scoped sweep acted on.
pub struct OwnedRemoval {
    pub id: String,
    pub name: String,
    /// Whether the removal succeeded. A failure is reported rather than
    /// retried: the caller decides whether that blocks anything.
    pub removed: bool,
}

/// Remove every container labelled with `owner`, except those in `skip`.
///
/// The owner label is the whole safety argument: it is minted once per
/// installation and applied by ananke alone, so a container carrying it is
/// this installation's by construction. Nothing here matches on names or on
/// the broader `io.ananke.managed` label, either of which could belong to a
/// second ananke sharing the runtime.
pub async fn remove_owned_containers(
    engine: &dyn ananke_system::container::ContainerEngine,
    owner: &str,
    skip: &BTreeSet<String>,
) -> Vec<OwnedRemoval> {
    let filters = [format!("label={OWNER_LABEL}={owner}")];
    let candidates = match engine.list(&filters).await {
        Ok(l) => l,
        Err(e) => {
            warn!(error = %e, "listing owned containers failed; any leak waits for the next start");
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for container in candidates {
        if skip.contains(&container.id) {
            continue;
        }
        // Defence in depth: the runtime applied the filter, but removal is
        // destructive enough to re-check the label being trusted.
        if container.owner.as_deref() != Some(owner) {
            continue;
        }
        let removed = remove_by_id(engine, &container.id).await.is_ok();
        out.push(OwnedRemoval {
            id: container.id,
            name: container.name,
            removed,
        });
    }
    out
}

/// The context every container-reconciliation step shares: the engine to
/// resolve per-runtime adapters from, this installation's owner UUID, and
/// the store holding the records being reconciled against.
struct Reconciler<'a> {
    engine: &'a dyn ananke_system::container::ContainerEngine,
    owner: &'a str,
    db: &'a Database,
}

/// Reconcile every launch intent not already covered by a running row.
async fn reconcile_intents(
    ctx: &Reconciler<'_>,
    handled: &BTreeSet<(i64, i64)>,
    resolved: &mut BTreeSet<String>,
    out: &mut Vec<OrphanDisposition>,
) {
    let (engine, owner, db) = (ctx.engine, ctx.owner, ctx.db);
    let intents = match db.list_launch_intents().await {
        Ok(i) => i,
        Err(e) => {
            warn!(error = %e, "listing launch intents failed; container recovery skipped");
            return;
        }
    };

    for intent in intents {
        if handled.contains(&(intent.service_id, intent.run_id)) {
            // The running row already resolved this run; the intent is just
            // stale bookkeeping.
            if let Err(e) = db.delete_launch_intent(intent.intent_id).await {
                warn!(error = %e, "deleting a resolved launch intent failed");
            }
            continue;
        }
        // An intent written by a different installation is not ours to act
        // on, even though it shares this database.
        if intent.owner_uuid != owner {
            out.push(OrphanDisposition::Container {
                container_id: intent.container_id,
                container_name: Some(intent.container_name),
                service_id: intent.service_id,
                run_id: intent.run_id,
                action: "intent-foreign",
            });
            continue;
        }

        // The intent records the exact executable, so a store-path or
        // otherwise non-default runtime stays reachable.
        let engine = engine.for_executable(&intent.runtime_executable);
        let engine = engine.as_ref();

        // Resolve the object: by recorded ID when create got far enough to
        // report one, otherwise by the generated name among this owner's
        // containers. The name lookup is scoped by the owner label, so it
        // never becomes a global scan of everything on the host.
        let found = match &intent.container_id {
            Some(id) => engine.inspect(id).await.ok().map(|i| (i.id, i.owner)),
            None => find_owned_by_name(engine, owner, &intent.container_name).await,
        };

        let action = match found {
            None => {
                // Crashed before create, or the object is already gone. The
                // intent is the only residue and is safe to drop.
                "intent-absent"
            }
            Some((id, owner_label)) => {
                resolved.insert(id.clone());
                if owner_label.as_deref() != Some(owner) {
                    // Something else answers to that name. Never touch it.
                    warn!(
                        service_id = intent.service_id,
                        container = %intent.container_name,
                        "intent resolved to a container this installation does not own"
                    );
                    out.push(OrphanDisposition::Container {
                        container_id: Some(id),
                        container_name: Some(intent.container_name),
                        service_id: intent.service_id,
                        run_id: intent.run_id,
                        action: "intent-foreign",
                    });
                    continue;
                }
                if remove_by_id(engine, &id).await.is_err() {
                    // Cleanup failed: keep the intent as a reconciliation
                    // block so the service isn't reprovisioned over a
                    // container that may still be running.
                    if let Err(e) = db.mark_intent_blocked(intent.intent_id).await {
                        warn!(error = %e, "marking a launch intent blocked failed");
                    }
                    out.push(OrphanDisposition::Container {
                        container_id: Some(id),
                        container_name: Some(intent.container_name),
                        service_id: intent.service_id,
                        run_id: intent.run_id,
                        action: "intent-blocked-remove-failed",
                    });
                    continue;
                }
                "intent-removed"
            }
        };

        // The evidence is only dropped once cleanup is confirmed.
        if let Err(e) = db.delete_launch_intent(intent.intent_id).await {
            warn!(error = %e, "deleting a reconciled launch intent failed");
        }
        out.push(OrphanDisposition::Container {
            container_id: intent.container_id,
            container_name: Some(intent.container_name),
            service_id: intent.service_id,
            run_id: intent.run_id,
            action,
        });
    }
}

/// An engine driving the binary for `runtime` (`"docker"` / `"podman"`),
/// falling back to the caller's own when the row predates the column or
/// names something unrecognised.
fn engine_for_runtime(
    engine: &dyn ananke_system::container::ContainerEngine,
    runtime: Option<&str>,
) -> std::sync::Arc<dyn ananke_system::container::ContainerEngine> {
    match runtime {
        Some(name @ ("docker" | "podman")) => engine.for_executable(name),
        _ => engine.for_executable(ananke_spawn::ContainerRuntime::Docker.executable()),
    }
}

/// Find a container by its generated name among those labelled with this
/// installation's owner UUID. Used when create succeeded but the ID never
/// made it back into the intent.
async fn find_owned_by_name(
    engine: &dyn ananke_system::container::ContainerEngine,
    owner: &str,
    name: &str,
) -> Option<(String, Option<String>)> {
    let filters = [format!("label={OWNER_LABEL}={owner}")];
    let summaries = match engine.list(&filters).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "listing owned containers failed; leaving the intent in place");
            return None;
        }
    };
    summaries
        .into_iter()
        .find(|s| s.name == name)
        .map(|s| (s.id, s.owner))
}

/// Reconcile one native-process row: adopt if the recorded cmdline matches
/// the live pid, otherwise clean it.
async fn reconcile_process_row(
    proc: &dyn ProcFs,
    db: &Database,
    service_id: i64,
    run_id: i64,
    pid: Option<i64>,
    recorded_cmdline: String,
    out: &mut Vec<OrphanDisposition>,
) {
    let Some(pid) = pid else {
        warn!(service_id, run_id, "process row with no pid; cleaning");
        cleanup_row(db, service_id, run_id).await;
        out.push(OrphanDisposition::Cleaned {
            pid: 0,
            service_id,
            run_id,
        });
        return;
    };
    let pid = pid as i32;
    match proc.cmdline(pid) {
        Some(live_cmdline) => {
            if live_cmdline == recorded_cmdline {
                info!(pid, service_id, run_id, "adopted orphan");
                out.push(OrphanDisposition::Adopted {
                    pid,
                    service_id,
                    run_id,
                });
            } else {
                warn!(
                    pid,
                    service_id,
                    run_id,
                    recorded = %recorded_cmdline,
                    live = %live_cmdline,
                    "unrelated process at recorded pid; cleaning row"
                );
                cleanup_row(db, service_id, run_id).await;
                out.push(OrphanDisposition::Cleaned {
                    pid,
                    service_id,
                    run_id,
                });
            }
        }
        None => {
            info!(pid, service_id, run_id, "dead child; cleaning row");
            cleanup_row(db, service_id, run_id).await;
            out.push(OrphanDisposition::Cleaned {
                pid,
                service_id,
                run_id,
            });
        }
    }
}

/// Reconcile one container row: inspect by ID (or name), verify ownership
/// labels, then remove and clean the row.
async fn reconcile_container_row(
    ctx: &Reconciler<'_>,
    service_id: i64,
    run_id: i64,
    row: ananke_db::models::RunningService,
    resolved: &mut BTreeSet<String>,
    out: &mut Vec<OrphanDisposition>,
) {
    let (owner, db) = (ctx.owner, ctx.db);
    let id = row.container_id.clone();
    let name = row.container_name.clone();
    // The row records which runtime created this container, and a config
    // that has since switched runtimes must not send `docker` after a
    // Podman container.
    let engine = engine_for_runtime(ctx.engine, row.runtime.as_deref());
    let engine = engine.as_ref();

    // Resolve the container identity: inspect by ID first, then fall back to
    // the generated name among this owner's containers. The fallback covers
    // a row written before the ID was known.
    let inspected = match &id {
        Some(id) => engine.inspect(id).await.ok().map(|i| (i.id, i.owner)),
        None => match &name {
            Some(name) => find_owned_by_name(engine, owner, name).await,
            None => None,
        },
    };

    // If ID inspect failed (or no ID), the list path is not reliably keyed
    // without a full scan; emit a blocked disposition preserving evidence.
    match inspected {
        Some((container_id, owner_label)) => {
            resolved.insert(container_id.clone());
            if owner_label.as_deref() != Some(owner) {
                // Foreign container: do not touch; preserve evidence.
                warn!(
                    service_id,
                    run_id,
                    container = ?name,
                    "container owner mismatch; leaving row for operator"
                );
                out.push(OrphanDisposition::Container {
                    container_id: Some(container_id),
                    container_name: Some(name.clone().unwrap_or_default()),
                    service_id,
                    run_id,
                    action: "foreign",
                });
                return;
            }
            // Remove it, and only then clean the row that is its evidence.
            let action = if remove_by_id(engine, &container_id).await.is_ok() {
                cleanup_row(db, service_id, run_id).await;
                "removed"
            } else {
                "blocked-remove-failed"
            };
            out.push(OrphanDisposition::Container {
                container_id: Some(container_id),
                container_name: Some(name.clone().unwrap_or_default()),
                service_id,
                run_id,
                action,
            });
        }
        None => {
            // No live container with that ID. Clean the row (it's stale).
            cleanup_row(db, service_id, run_id).await;
            out.push(OrphanDisposition::Container {
                container_id: id,
                container_name: name,
                service_id,
                run_id,
                action: "absent",
            });
        }
    }
}

async fn remove_by_id(
    engine: &dyn ananke_system::container::ContainerEngine,
    id: &str,
) -> Result<(), ExpectedError> {
    engine.remove(id).await
}

async fn cleanup_row(db: &Database, service_id: i64, run_id: i64) {
    if let Err(e) = db.delete_running(service_id, run_id).await {
        warn!(error = %e, "delete running_services row failed");
    }
}

#[cfg(test)]
mod tests {
    use ananke_db::models::RunningService;
    use ananke_system::{InMemoryProcFs, container::FakeContainerEngine};

    use super::*;

    async fn insert_row(db: &Database, service_id: i64, run_id: i64, pid: i32, cmdline: &str) {
        db.insert_running(&RunningService {
            service_id,
            run_id,
            pid: Some(pid as i64),
            spawned_at: 0,
            command_line: cmdline.to_string(),
            allocation: "{}".to_string(),
            state: "running".to_string(),
            workload_kind: Some("process".to_string()),
            runtime: None,
            container_name: None,
            container_id: None,
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn adopts_matching_cmdline() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        let engine = FakeContainerEngine::new();
        insert_row(&db, svc, 1, 1234, "llama-server -m x").await;
        proc.set_cmdline(1234, "llama-server -m x");
        let out = reconcile(&proc, &engine, None, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Adopted { .. }));
    }

    #[tokio::test]
    async fn cleans_missing_pid() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        let engine = FakeContainerEngine::new();
        insert_row(&db, svc, 1, 9999, "llama-server -m x").await;
        let out = reconcile(&proc, &engine, None, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
        assert!(db.list_running().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn cleans_mismatched_cmdline() {
        let db = Database::open_in_memory().await.unwrap();
        let svc = db.upsert_service("demo", 0).await.unwrap();
        let proc = InMemoryProcFs::new();
        let engine = FakeContainerEngine::new();
        insert_row(&db, svc, 1, 4242, "llama-server -m x").await;
        proc.set_cmdline(4242, "firefox");
        let out = reconcile(&proc, &engine, None, &db).await;
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], OrphanDisposition::Cleaned { .. }));
    }
}
