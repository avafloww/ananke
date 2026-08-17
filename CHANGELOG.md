# Changelog

Notable changes to ananke. Entries are grouped by release; the top section
collects what has landed since the last one.

The audience is an operator deciding whether to upgrade and what they will
have to change. Behaviour changes and anything that touches an existing
config belong here; refactors and internal tidying do not.

## Unreleased

### Breaking

- **Placeholders are now written `${name}`, not `{name}`.** The old form
  could not be distinguished from an argument's own braces without
  guessing, which made JSON arguments fail to launch and silently rewrote
  Jinja templates. Only `${` and `$$` are special now: a bare `{` never is,
  so JSON, Jinja, and format strings pass through exactly as written and
  need no escaping. `$$` is a literal `$`, which is how a literal
  `${port}` is written.

  Rename every placeholder in `command`, `shutdown_command`, `launcher`,
  `env` values, and `container.gpu_device`. A config still using `{name}`
  does not error — the text is simply literal now — so confirm a service's
  rendered command still contains the value you expect, through
  `GET /api/services/{name}/command` or the launch-command panel.

  An unknown name, or a `${` with no closing brace, is a config error. A
  known name written bare — `{port}` — is a warning naming the service and
  the entry, since it cannot be an error: a JSON or format-string argument
  may hold `{model}` on purpose. Write `$${model}` to say that and silence
  it.

### Added

- **Container workloads.** Either template can run its workload in a Docker
  or Podman container through a `[service.container]` block, without a
  wrapper script. ananke owns the whole lifecycle: it creates, starts,
  follows logs, reads the authoritative exit status, signals, removes, and
  reconciles leftovers after a crash. The template still decides what argv
  is produced; the block decides where it runs.

  Covers host and bridge networking with a loopback-only publication, bind
  mounts with automatic path translation for llama.cpp's model fields, CDI
  GPU injection scoped to the devices the allocator picked, explicit
  environment plus a name-only passthrough allowlist, and IPC mode. See
  [Container Workloads](docs/configuration.md#container-workloads).

- **A `combined` log stream** for container output. Neither runtime's CLI
  documents a framing that preserves stdout from stderr across the process
  boundary, so container output is labelled `combined` rather than being
  passed off as either. Native processes keep their split streams.
  Available through the API, the WebSocket, `anankectl logs --stream`, and
  the log viewer.

- **Container detail in the API and UI**: runtime, image, network mode, and
  live container identity, alongside the exact `create` argv a service
  would launch with. Passthrough variables appear by name only.

- **An offline config validator**:
  `cargo run -p ananke-supervise --example validate-config -- <file>`
  parses, validates, and renders the create argv each containerized service
  would launch with. `anankectl server-config validate` needs a running
  daemon; this does not.

### Fixed

- A `}` with nothing to close made the placeholder scanner loop forever,
  hanging config load.
- A typo in an `env` or `container.env` value now fails at config load,
  naming the variable, rather than at the launch it breaks.
- Container removal waits for the log follower to finish, so the tail of a
  stopped container's output — the part saying why it stopped — is no
  longer lost to the removal that deletes the runtime's log store.
- A runtime that cannot be reached is no longer read as a container that is
  not there. Reconciliation kept deleting the record in that case, which
  strands a running container with nothing pointing at it. The startup
  leak sweep also asks every runtime the records name, rather than
  whichever this process defaults to — on a Podman-only host it was
  running `docker ps` and finding nothing.
- The daemon removes every container it owns on shutdown when a drain
  overruns `shutdown_timeout`, rather than exiting and leaving the workload
  holding its reservation.
- `pid 0` is no longer registered for memory attribution. It parents init,
  so attributing its descendants summed the whole machine into one service.

### Notes

- Images are not pulled or built. ananke starts and stops what you point it
  at; the image must already be in the runtime's local store.
- A container cannot inherit `PR_SET_PDEATHSIG`, so one survives a
  `SIGKILL`ed daemon until startup reconciliation removes it. The allocator
  reads real free VRAM, so a survivor makes ananke conservative rather than
  wrong. See
  [What happens when ananke stops](docs/configuration.md#container-workloads).

## 0.2.0

Released before this file existed. See the git history for the changes it
carried.
