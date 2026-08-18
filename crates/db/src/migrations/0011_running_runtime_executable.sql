-- Record the exact runtime binary a container was launched with, not just
-- which runtime it was.
--
-- `runtime` alone reconstructs a bare `docker` or `podman`, which is only
-- the same thing when the binary is on the daemon's PATH. A service naming
-- an explicit `runtime_executable` — a Nix store path, a wrapper — would,
-- after a restart, be reconciled by shelling a name that resolves to
-- nothing. `inspect` then fails, the row is cleaned as absent, and the
-- container keeps running with no record of it.
--
-- The launch intent already carries this. A plain ADD COLUMN suffices:
-- the column is nullable, and rows predating it fall back to the runtime's
-- default name, which is what they were launched with.

ALTER TABLE running_services ADD COLUMN runtime_executable TEXT;
