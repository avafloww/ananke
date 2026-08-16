-- Extend running_services with managed-workload identity: workload kind,
-- runtime, container name, and container ID. Retains native-process command
-- compatibility; `pid` becomes nullable because a container workload may
-- have no host PID.
--
-- SQLite cannot ALTER column nullability, so this performs a full table
-- recreation: create the new schema, copy, drop the old table, rename, and
-- recreate the index.

CREATE TABLE running_services_new (
    service_id   INTEGER NOT NULL,
    run_id       INTEGER NOT NULL,
    pid          INTEGER,
    spawned_at   INTEGER NOT NULL,
    command_line TEXT NOT NULL,
    allocation   TEXT NOT NULL,
    state        TEXT NOT NULL,
    workload_kind   TEXT,
    runtime         TEXT,
    container_name  TEXT,
    container_id    TEXT,
    PRIMARY KEY (service_id, run_id),
    -- Carried over from 0001. A table recreation silently drops constraints
    -- that aren't restated, and this one is what stops a running row from
    -- outliving the service it belongs to.
    FOREIGN KEY (service_id) REFERENCES services(service_id)
);

INSERT INTO running_services_new
    (service_id, run_id, pid, spawned_at, command_line, allocation, state)
SELECT
    service_id, run_id, pid, spawned_at, command_line, allocation, state
FROM running_services;

DROP TABLE running_services;

ALTER TABLE running_services_new RENAME TO running_services;

CREATE INDEX idx_running_services_service_id ON running_services(service_id);