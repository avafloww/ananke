-- 0007_service_restart_counts: monotonic per-trigger restart tallies.
--
-- `service_restarts` is capped per service (see `SERVICE_RESTART_CAP`), which
-- makes it unsuitable as the source for a Prometheus counter: the cap is
-- shared across triggers, so a service with a `periodic` restart every 6h
-- evicts every `spec_collapse` row within a couple of weeks and the exported
-- `ananke_auto_restarts_total{trigger="spec_collapse"}` falls from 50 to 0.
-- Prometheus reads a falling counter as a process restart and attributes the
-- next firing as a fresh increment, so the eviction shows up on a dashboard
-- as a phantom restart spike during exactly the incident an operator is
-- trying to read.
--
-- This table is never pruned. It carries only counts, so it stays at one
-- small row per (service, trigger) pair forever.
--
-- The seed populates it from whatever `service_restarts` history survives, so
-- an existing deployment keeps the counts it can still see rather than
-- starting from zero.

CREATE TABLE service_restart_counts (
  service_id   INTEGER NOT NULL REFERENCES services(service_id),
  trigger_name TEXT NOT NULL,
  count        INTEGER NOT NULL,
  PRIMARY KEY (service_id, trigger_name)
);

INSERT INTO service_restart_counts (service_id, trigger_name, count)
  SELECT service_id, trigger_name, COUNT(*)
  FROM service_restarts
  GROUP BY service_id, trigger_name;

-- `query_service_restarts` scans a bare `at_ms` range when no service filter
-- is given, which the (service_id, at_ms) index cannot serve.
CREATE INDEX idx_service_restarts_at ON service_restarts(at_ms);
