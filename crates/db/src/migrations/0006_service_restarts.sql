-- 0006_service_restarts: durable record of auto-restart watchdog firings.
--
-- Watchdog firings were previously only broadcast on the live `/api/events`
-- WebSocket, so a firing with no browser attached left nothing behind but a
-- daemon log line. The 2026-07-24 incident made the cost concrete: the
-- generation-stall watchdog correctly restarted a wedged service at 14:37
-- and nobody knew until hours later. Each firing now also lands here, and
-- the service-detail endpoint serves the recent history.
--
-- trigger_name: which watchdog fired ("error_rate", "ttft_stall",
--               "generation_stall", "spec_collapse", or "periodic").
-- detail:      the human-readable reason string carried by the event.
-- run_id:      the run that was drained; nullable for forward compatibility
--              with restart-shaped events that have no run attached.
--
-- Retention: capped per service at insert time (see
-- `Database::insert_service_restart`), so the table stays small without a
-- background sweeper.

CREATE TABLE service_restarts (
  restart_id   INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  service_id   INTEGER NOT NULL REFERENCES services(service_id),
  run_id       INTEGER,
  at_ms        INTEGER NOT NULL,
  trigger_name TEXT NOT NULL,
  detail       TEXT NOT NULL
);

CREATE INDEX idx_service_restarts_service
  ON service_restarts(service_id, at_ms);
