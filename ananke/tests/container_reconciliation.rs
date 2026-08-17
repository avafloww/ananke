//! Startup reconciliation across every crash window a container launch has.
//!
//! The launch sequence writes a durable intent, creates, attaches the ID,
//! commits the running row, then starts. A crash can land between any two of
//! those, so each window leaves a different residue: an intent with no
//! object, an object with no recorded ID, an object with no running row, or
//! a running row whose container is already gone. Reconciliation has to
//! converge from all of them — and has to refuse to touch anything it cannot
//! prove it owns.
#![cfg(feature = "test-fakes")]

use std::collections::{BTreeMap, BTreeSet};

use ananke::supervise::orphans::{OrphanDisposition, reconcile, remove_owned_containers};
use ananke_db::{
    Database,
    models::{ContainerLaunchIntent, RunningService},
};
use ananke_spawn::{ContainerIpc, ContainerNetwork, ContainerRuntime, ContainerSpec};
use ananke_system::{
    InMemoryProcFs,
    container::{ContainerEngine, FakeContainerEngine, FakeContainerState},
};

const OWNER: &str = "11111111-1111-4111-8111-111111111111";
const OTHER_OWNER: &str = "22222222-2222-4222-8222-222222222222";

/// A container spec labelled for `owner`, as a real launch would build it.
fn spec(name: &str, owner: &str, service: &str, run_id: i64) -> ContainerSpec {
    ContainerSpec {
        runtime: ContainerRuntime::Docker,
        runtime_executable: None,
        image: "ninfer:local".into(),
        entrypoint: None,
        workdir: None,
        command: vec!["ninfer-serve".into()],
        name: name.into(),
        labels: BTreeMap::from([
            ("io.ananke.managed".into(), "true".into()),
            ("io.ananke.owner".into(), owner.into()),
            ("io.ananke.service".into(), service.into()),
            ("io.ananke.run".into(), run_id.to_string()),
        ]),
        network: ContainerNetwork::Host,
        container_port: None,
        host_port: None,
        ipc: ContainerIpc::Private,
        gpu_devices: Vec::new(),
        mounts: Vec::new(),
        extra_publications: Vec::new(),
        env: BTreeMap::new(),
        env_passthrough: Vec::new(),
    }
}

fn intent(
    service_id: i64,
    run_id: i64,
    owner: &str,
    name: &str,
    container_id: Option<&str>,
) -> ContainerLaunchIntent {
    ContainerLaunchIntent {
        intent_id: 0,
        service_id,
        run_id,
        owner_uuid: owner.to_string(),
        workload_kind: "container".to_string(),
        runtime: "docker".to_string(),
        runtime_executable: "/nix/store/x/bin/docker".to_string(),
        container_name: name.to_string(),
        labels_json: format!("{{\"io.ananke.owner\":\"{owner}\"}}"),
        spec_json: "{}".to_string(),
        container_id: container_id.map(str::to_string),
        state: "intent".to_string(),
        created_at: 0,
    }
}

fn running_row(service_id: i64, run_id: i64, name: &str, id: Option<&str>) -> RunningService {
    RunningService {
        service_id,
        run_id,
        pid: None,
        spawned_at: 0,
        command_line: "ninfer-serve".to_string(),
        allocation: "{}".to_string(),
        state: "running".to_string(),
        workload_kind: Some("container".to_string()),
        runtime: Some("docker".to_string()),
        container_name: Some(name.to_string()),
        container_id: id.map(str::to_string),
        runtime_executable: None,
    }
}

/// The action of the single container disposition in `out`.
fn sole_container_action(out: &[OrphanDisposition]) -> &'static str {
    let containers: Vec<_> = out
        .iter()
        .filter_map(|d| match d {
            OrphanDisposition::Container { action, .. } => Some(*action),
            _ => None,
        })
        .collect();
    assert_eq!(
        containers.len(),
        1,
        "expected one container disposition: {out:?}"
    );
    containers[0]
}

#[tokio::test]
async fn launch_intent_crash_before_create_is_reconcilable() {
    // Crashed after writing the intent, before create: no object exists.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    db.insert_launch_intent(&intent(svc, 7, OWNER, "ananke-ninfer-7", None))
        .await
        .unwrap();

    let out = reconcile(
        &InMemoryProcFs::new(),
        &FakeContainerEngine::new(),
        Some(OWNER),
        &db,
    )
    .await;

    assert_eq!(sole_container_action(&out), "intent-absent");
    assert!(
        db.list_launch_intents().await.unwrap().is_empty(),
        "an intent with no object is safe to drop"
    );
}

#[tokio::test]
async fn create_crash_before_id_update_recovers_by_owned_name() {
    // Create succeeded, but the ID never made it into the intent. The
    // generated name plus the owner label is enough to find it again.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(svc, 7, OWNER, "ananke-ninfer-7", None))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "intent-removed");
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
    assert!(db.list_launch_intents().await.unwrap().is_empty());
}

#[tokio::test]
async fn intent_with_id_before_start_reconciles() {
    // Create succeeded and the ID was attached, but the row commit or start
    // never happened: there is an object, an intent, and no running row.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "intent-removed");
    assert_eq!(
        engine.find(&prepared.id).unwrap().operations,
        ["create", "remove"],
        "a container created but never started is removed, not started"
    );
}

#[tokio::test]
async fn running_row_reconciles_through_its_recorded_binary() {
    // `runtime` alone reconstructs a bare name. A service naming an explicit
    // binary — a Nix store path, a wrapper — would otherwise be reconciled
    // by shelling something that resolves to nothing, and the row cleaned as
    // absent while the container ran on.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();

    let mut row = running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id));
    row.runtime = Some("podman".to_string());
    row.runtime_executable = Some("/nix/store/x/bin/podman".to_string());
    db.insert_running(&row).await.unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "removed");
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
    assert!(db.list_running().await.unwrap().is_empty());
}

#[tokio::test]
async fn runtime_executable_override_reconciles_from_intent() {
    // The intent records the exact runtime executable, so a container stays
    // reachable even if the config that named it is gone.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    let recorded = intent(svc, 7, OWNER, "ananke-ninfer-7", Some(&prepared.id));
    assert_eq!(recorded.runtime_executable, "/nix/store/x/bin/docker");
    db.insert_launch_intent(&recorded).await.unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;
    assert_eq!(sole_container_action(&out), "intent-removed");
}

#[tokio::test]
async fn intent_survives_config_removal_and_runtime_change() {
    // Nothing in reconciliation reads the config: an intent for a service
    // that no longer exists in the TOML still resolves to its container.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("deleted-from-config", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec(
            "ananke-deleted-from-config-7",
            OWNER,
            "deleted-from-config",
            7,
        ))
        .await
        .unwrap();
    let mut recorded = intent(
        svc,
        7,
        OWNER,
        "ananke-deleted-from-config-7",
        Some(&prepared.id),
    );
    // The operator has since switched the service to Podman.
    recorded.runtime = "podman".to_string();
    recorded.runtime_executable = "podman".to_string();
    db.insert_launch_intent(&recorded).await.unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "intent-removed");
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
}

#[tokio::test]
async fn reconcile_created_container() {
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "removed");
    assert!(db.list_running().await.unwrap().is_empty());
}

#[tokio::test]
async fn reconcile_running_container() {
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    prepared.start().await.unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    // A survivor of the previous daemon is removed, not adopted: its logs
    // and exit status were being followed by a process that no longer exists.
    assert_eq!(sole_container_action(&out), "removed");
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
}

#[tokio::test]
async fn reconcile_exited_container() {
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    prepared.start().await.unwrap();
    assert!(engine.exit(&prepared.id, 1));
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "removed");
    assert!(db.list_running().await.unwrap().is_empty());
}

#[tokio::test]
async fn podman_rows_and_intents_resolve_their_own_runtime() {
    // One engine serves every service, so a row or intent naming Podman has
    // to steer the id-keyed operations at `podman` rather than whatever the
    // shared adapter defaults to.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();

    let mut podman_spec = spec("ananke-ninfer-7", OWNER, "ninfer", 7);
    podman_spec.runtime = ContainerRuntime::Podman;
    let prepared = engine.create(&podman_spec).await.unwrap();

    let mut row = running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id));
    row.runtime = Some("podman".to_string());
    db.insert_running(&row).await.unwrap();

    // An intent for a second run, recorded against a store-path binary.
    let mut recorded = intent(svc, 8, OWNER, "ananke-ninfer-8", None);
    recorded.runtime = "podman".to_string();
    recorded.runtime_executable = "/nix/store/x/bin/podman".to_string();
    db.insert_launch_intent(&recorded).await.unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    let actions: Vec<&str> = out
        .iter()
        .filter_map(|d| match d {
            OrphanDisposition::Container { action, .. } => Some(*action),
            _ => None,
        })
        .collect();
    assert_eq!(actions, ["removed", "intent-absent"], "{out:?}");
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
    assert!(db.list_running().await.unwrap().is_empty());
    assert!(db.list_launch_intents().await.unwrap().is_empty());
}

#[tokio::test]
async fn owner_sweep_removes_only_this_installations_containers() {
    // The sweep the daemon runs on shutdown, when a drain has overrun its
    // bound and the daemon is about to exit anyway. Its whole safety
    // argument is the owner label.
    let engine = FakeContainerEngine::new();
    let mine = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    mine.start().await.unwrap();
    let theirs = engine
        .create(&spec("ananke-other-9", OTHER_OWNER, "other", 9))
        .await
        .unwrap();

    let removed = remove_owned_containers(&engine, OWNER, &BTreeSet::new()).await;

    assert_eq!(removed.len(), 1);
    assert!(removed[0].removed);
    assert_eq!(removed[0].name, "ananke-ninfer-7");
    assert_eq!(
        engine.find(&mine.id).unwrap().state,
        FakeContainerState::Removed,
        "a running container is removed: the daemon is leaving and cannot supervise it"
    );
    assert_ne!(
        engine.find(&theirs.id).unwrap().state,
        FakeContainerState::Removed
    );
}

#[tokio::test]
async fn owner_sweep_is_a_no_op_after_a_clean_drain() {
    // The normal shutdown: every container was already removed by its
    // drain, so the sweep finds nothing and costs one listing.
    let engine = FakeContainerEngine::new();
    let gone = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    gone.remove().await.unwrap();

    let removed = remove_owned_containers(&engine, OWNER, &BTreeSet::new()).await;
    assert!(removed.is_empty());
}

#[tokio::test]
async fn unrecorded_owned_container_is_swept() {
    // A container this installation created but has no row or intent for —
    // leaked by a path that dropped both. The owner label is the only thing
    // tying it to us, and it is enough because nothing else can carry it.
    let db = Database::open_in_memory().await.unwrap();
    db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let ours = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    let theirs = engine
        .create(&spec("ananke-ninfer-9", OTHER_OWNER, "ninfer", 9))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "leaked-removed");
    assert_eq!(
        engine.find(&ours.id).unwrap().state,
        FakeContainerState::Removed
    );
    assert_ne!(
        engine.find(&theirs.id).unwrap().state,
        FakeContainerState::Removed,
        "another installation's container is never swept"
    );
}

#[tokio::test]
async fn sweep_skip_set_matches_listed_ids() {
    // The skip set is populated from `inspect` and tested against `list`.
    // If those disagreed on id form — full vs truncated — the set would be
    // silently inert and the sweep would remove containers a record had
    // deliberately retained.
    let engine = FakeContainerEngine::new();
    let c = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    let inspected = engine.inspect(&c.id).await.unwrap().expect("just created");
    let listed = engine.list(&[]).await.unwrap();

    assert_eq!(listed.len(), 1);
    assert_eq!(
        listed[0].id, inspected.id,
        "list and inspect must agree on the id form"
    );
    assert_eq!(listed[0].owner.as_deref(), Some(OWNER));

    let skip = BTreeSet::from([inspected.id.clone()]);
    assert!(
        remove_owned_containers(&engine, OWNER, &skip)
            .await
            .is_empty(),
        "an id in the skip set must be skipped"
    );
}

#[tokio::test]
async fn sweep_leaves_containers_a_record_explains() {
    // A container a running row accounts for is removed by that path, not
    // reported twice by the sweep behind it.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "removed");
    assert_eq!(
        engine
            .find(&prepared.id)
            .unwrap()
            .operations
            .iter()
            .filter(|o| *o == "remove")
            .count(),
        1
    );
}

#[tokio::test]
async fn reconcile_ignores_other_ananke_owner() {
    // Two ananke installations sharing one Docker daemon. A container
    // labelled for the other owner must survive untouched, and its row must
    // stay as evidence rather than being silently dropped.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OTHER_OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "foreign");
    assert_ne!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed,
        "another installation's container must never be removed"
    );
    assert_eq!(
        db.list_running().await.unwrap().len(),
        1,
        "the row is preserved as evidence"
    );
}

#[tokio::test]
async fn name_lookup_will_not_adopt_a_foreign_container() {
    // The name-based fallback is scoped by the owner label, so a container
    // that merely shares the generated name is invisible to it.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OTHER_OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(svc, 7, OWNER, "ananke-ninfer-7", None))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "intent-absent");
    assert_ne!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
}

#[tokio::test]
async fn invalid_owner_identity_disables_container_reconciliation_without_scanning() {
    // Without a usable owner UUID there is no way to prove ownership, so the
    // scan must not happen at all and nothing may be removed.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(svc, 8, OWNER, "ananke-ninfer-8", None))
        .await
        .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, None, &db).await;

    assert_eq!(sole_container_action(&out), "blocked-no-owner");
    assert_ne!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
    assert_eq!(db.list_running().await.unwrap().len(), 1);
    assert_eq!(
        db.list_launch_intents().await.unwrap().len(),
        1,
        "intents are left untouched when ownership cannot be established"
    );
}

#[tokio::test]
async fn unresolved_container_blocks_service_reprovision() {
    // Cleanup failed, so the evidence is retained and the intent is marked
    // blocked: the service must not be started over a container that may
    // still be running.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();
    engine.fail_remove();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "intent-blocked-remove-failed");
    let blocked = db.launch_intents_for_service(svc).await.unwrap();
    assert_eq!(blocked.len(), 1, "the evidence is retained");
    assert_eq!(blocked[0].state, "blocked");
}

#[tokio::test]
async fn reconcile_block_does_not_prevent_unrelated_native_provision() {
    // One service blocked on an unresolvable container must not stop an
    // unrelated native service from reconciling normally.
    let db = Database::open_in_memory().await.unwrap();
    let blocked_svc = db.upsert_service("ninfer", 0).await.unwrap();
    let native_svc = db.upsert_service("comfy", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        blocked_svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();
    engine.fail_remove();

    let proc = InMemoryProcFs::new();
    proc.set_cmdline(4242, "python main.py");
    db.insert_running(&RunningService {
        service_id: native_svc,
        run_id: 1,
        pid: Some(4242),
        spawned_at: 0,
        command_line: "python main.py".to_string(),
        allocation: "{}".to_string(),
        state: "running".to_string(),
        workload_kind: Some("process".to_string()),
        runtime: None,
        container_name: None,
        container_id: None,
        runtime_executable: None,
    })
    .await
    .unwrap();

    let out = reconcile(&proc, &engine, Some(OWNER), &db).await;

    assert!(
        out.iter()
            .any(|d| matches!(d, OrphanDisposition::Adopted { pid: 4242, .. })),
        "the unrelated native service is adopted as usual: {out:?}"
    );
    assert_eq!(sole_container_action(&out), "intent-blocked-remove-failed");
}

#[tokio::test]
async fn reconciliation_retry_unblocks_and_reprovisions_once() {
    // First pass: cleanup fails, evidence is retained, nothing is replaced.
    // Second pass: cleanup succeeds, and only then is the evidence dropped.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();

    engine.fail_remove();
    let first = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;
    assert_eq!(
        sole_container_action(&first),
        "intent-blocked-remove-failed"
    );
    assert_eq!(db.list_launch_intents().await.unwrap().len(), 1);
    assert_ne!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );

    let second = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;
    assert_eq!(sole_container_action(&second), "intent-removed");
    assert!(
        db.list_launch_intents().await.unwrap().is_empty(),
        "the block clears only after cleanup actually succeeded"
    );
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Removed
    );
}

#[tokio::test]
async fn resolved_run_drops_its_stale_intent() {
    // A running row and its intent describe the same run. The row is the
    // authority; the intent is dropped without a second removal attempt.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();

    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "removed");
    assert!(db.list_launch_intents().await.unwrap().is_empty());
    assert_eq!(
        engine
            .find(&prepared.id)
            .unwrap()
            .operations
            .iter()
            .filter(|o| *o == "remove")
            .count(),
        1,
        "the container is removed exactly once"
    );
}

#[tokio::test]
async fn unreachable_runtime_preserves_the_row() {
    // A runtime that cannot be reached is not evidence that the container is
    // gone. Cleaning the row here would strand a running container with no
    // record of it anywhere, so it is kept for the next start to resolve.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_running(&running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id)))
        .await
        .unwrap();

    engine.set_unreachable(true);
    let out = reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(sole_container_action(&out), "blocked-runtime-unreachable");
    assert_eq!(db.list_running().await.unwrap().len(), 1);
    assert_eq!(
        engine.find(&prepared.id).unwrap().state,
        FakeContainerState::Created,
        "an unreachable runtime must not be treated as a removal"
    );
}

#[tokio::test]
async fn unreachable_runtime_preserves_the_intent() {
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();
    db.insert_launch_intent(&intent(
        svc,
        7,
        OWNER,
        "ananke-ninfer-7",
        Some(&prepared.id),
    ))
    .await
    .unwrap();

    engine.set_unreachable(true);
    reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    assert_eq!(db.list_launch_intents().await.unwrap().len(), 1);
}

#[tokio::test]
async fn the_sweep_asks_every_recorded_runtime() {
    // The backstop runs `ps` per runtime the records name. Asking only the
    // process default would mean a Podman-only host runs `docker ps`, which
    // fails, so the sweep that exists to catch leaks silently finds none.
    let db = Database::open_in_memory().await.unwrap();
    let svc = db.upsert_service("ninfer", 0).await.unwrap();
    let engine = FakeContainerEngine::new();
    let prepared = engine
        .create(&spec("ananke-ninfer-7", OWNER, "ninfer", 7))
        .await
        .unwrap();

    // The row carries its ID, so it resolves through `inspect` alone. Any
    // listing that happens is the sweep's own.
    let mut row = running_row(svc, 7, "ananke-ninfer-7", Some(&prepared.id));
    row.runtime_executable = Some("/nix/store/x/bin/podman".to_string());
    db.insert_running(&row).await.unwrap();

    reconcile(&InMemoryProcFs::new(), &engine, Some(OWNER), &db).await;

    let scanned = engine.listed_under();
    assert!(
        scanned
            .iter()
            .any(|e| e.as_deref() == Some("/nix/store/x/bin/podman")),
        "the sweep never scanned the recorded runtime: {scanned:?}"
    );
}
