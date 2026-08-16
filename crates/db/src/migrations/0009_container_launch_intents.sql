-- Durable container launch intents. Inserted before invoking the runtime,
-- so a crash in any thinkable window (before create, after create but
-- before the ID update, or after the ID update but before start) leaves
-- enough information for startup reconciliation to find and clean up the
-- object without relying on the short-lived runtime CLI.

CREATE TABLE container_launch_intents (
    intent_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    service_id      INTEGER NOT NULL,
    run_id          INTEGER NOT NULL,
    owner_uuid      TEXT NOT NULL,
    workload_kind   TEXT NOT NULL,
    runtime         TEXT NOT NULL,
    runtime_executable TEXT NOT NULL,
    container_name  TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    spec_json       TEXT NOT NULL,
    container_id    TEXT,
    state           TEXT NOT NULL DEFAULT 'intent',
    created_at      INTEGER NOT NULL
);