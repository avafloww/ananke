-- Installation metadata: a singleton table holding the stable owner UUID
-- that isolates this ananke instance from others sharing a container
-- runtime. Created once at first migration; the daemon creates the row
-- transactionally before any runtime scan or container launch.

CREATE TABLE installation_metadata (
    version    INTEGER PRIMARY KEY,
    owner_uuid TEXT NOT NULL UNIQUE
);