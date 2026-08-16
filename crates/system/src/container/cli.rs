//! Docker and Podman CLI adapters implementing the [`ContainerEngine`]
//! trait. Commands are always rendered as argv vectors, never shell
//! strings.

use std::process::Stdio;

use ananke_errors::ExpectedError;
use ananke_spawn::ContainerSpec;
use async_trait::async_trait;
use tokio::process::{Child, Command};
use tracing::warn;

use crate::container::types::{
    ContainerEngine, ContainerInspect, ContainerSummary, DynAsyncRead, ManagedContainer,
    PreparedContainer,
};

/// Production CLI adapter that dispatches Docker/Podman CLI commands. The
/// adapter carries a default executable (`docker` by default) that a
/// per-spec `runtime_executable` override can replace.
pub struct CliContainerEngine {
    executable: String,
}

impl CliContainerEngine {
    /// Build a Docker CLI adapter using `docker` on `$PATH`.
    pub fn docker() -> Self {
        Self {
            executable: "docker".into(),
        }
    }

    /// Build a Podman CLI adapter using `podman` on `$PATH`.
    pub fn podman() -> Self {
        Self {
            executable: "podman".into(),
        }
    }

    /// Build a CLI adapter with an explicit default executable path.
    pub fn with_executable(executable: String) -> Self {
        Self { executable }
    }
}

/// Docker-specific adapter. Thin alias over [`CliContainerEngine`].
pub type DockerCli = CliContainerEngine;

/// Podman-specific adapter. Thin alias over [`CliContainerEngine`].
pub type PodmanCli = CliContainerEngine;

#[async_trait]
impl ContainerEngine for CliContainerEngine {
    fn for_executable(&self, executable: &str) -> std::sync::Arc<dyn ContainerEngine> {
        std::sync::Arc::new(Self {
            executable: executable.to_string(),
        })
    }

    async fn create(&self, spec: &ContainerSpec) -> Result<PreparedContainer, ExpectedError> {
        let exe = crate::container::render::executable_for(spec);
        let argv = crate::container::render::render_create_argv(&exe, spec);
        let id = run_checked(&exe, "create", &argv).await?;
        if id.is_empty() {
            return Err(cli_err(format!("{exe} create: empty container ID")));
        }
        Ok(PreparedContainer {
            id,
            name: spec.name.clone(),
            runtime_executable: exe.clone(),
            runtime: spec.runtime,
            engine: std::sync::Arc::new(Self { executable: exe }),
        })
    }

    async fn inspect(&self, id: &str) -> Result<ContainerInspect, ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_inspect_argv(exe, id);
        let raw = run_checked(exe, &format!("inspect {id}"), &argv).await?;
        parse_inspect_output(&raw)
            .ok_or_else(|| cli_err(format!("{exe} inspect {id}: unparseable output `{raw}`")))
    }

    async fn remove(&self, id: &str) -> Result<(), ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_remove_argv(exe, id);
        run_idempotent(exe, &format!("rm {id}"), &argv).await
    }

    async fn list(&self, filters: &[String]) -> Result<Vec<ContainerSummary>, ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_list_argv(exe, filters);
        let raw = run_checked(exe, "ps", &argv).await?;
        // `ps` gives ids; everything else comes from `inspect`, whose output
        // shape is stable across runtimes. The candidate set is this
        // installation's own containers, so the extra calls are few.
        let mut out = Vec::new();
        for id in raw.lines().map(str::trim).filter(|l| !l.is_empty()) {
            match self.inspect(id).await {
                Ok(i) => out.push(ContainerSummary {
                    id: i.id,
                    name: i.name,
                    state: i.state,
                    owner: i.owner,
                }),
                // Raced with a removal between listing and inspecting.
                Err(e) if is_absent(&e.to_string()) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    async fn start(
        &self,
        prepared: &PreparedContainer,
    ) -> Result<Box<dyn ManagedContainer>, ExpectedError> {
        let exe = &prepared.runtime_executable;
        let argv = crate::container::render::render_start_argv(exe, &prepared.id);
        run_checked(exe, &format!("start {}", prepared.id), &argv).await?;
        // Inspect for the host PID. A failure here is not fatal — the
        // container is started — but it costs attribution, so it is logged.
        let host_pid = match self.inspect(&prepared.id).await {
            Ok(i) => i.host_pid,
            Err(e) => {
                warn!(error = %e, container = %prepared.id, "inspect after start failed; no host pid for attribution");
                None
            }
        };
        Ok(Box::new(CliRunningContainer {
            id: prepared.id.clone(),
            name: prepared.name.clone(),
            executable: prepared.runtime_executable.clone(),
            host_pid,
            follower: parking_lot::Mutex::new(None),
        }))
    }

    async fn remove_prepared(&self, prepared: &PreparedContainer) -> Result<(), ExpectedError> {
        let exe = &prepared.runtime_executable;
        let argv = crate::container::render::render_remove_argv(exe, &prepared.id);
        run_idempotent(exe, &format!("rm {}", prepared.id), &argv).await
    }
}

/// Production running-container handle.
pub struct CliRunningContainer {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) executable: String,
    pub(crate) host_pid: Option<u32>,
    /// The `logs --follow` child, held for as long as this handle is.
    ///
    /// It has `kill_on_drop` set, so *where* it is dropped decides whether
    /// the follower streams or dies: dropped at the end of the function
    /// that spawned it, the pipes reach the caller already at EOF. Owning
    /// it here ties the follower's life to the container's, which is what
    /// closes the leak without severing the stream.
    pub(crate) follower: parking_lot::Mutex<Option<Child>>,
}

#[async_trait]
impl ManagedContainer for CliRunningContainer {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn runtime_executable(&self) -> &str {
        &self.executable
    }

    fn host_pid(&self) -> Option<u32> {
        self.host_pid
    }

    fn logs(&self) -> Vec<DynAsyncRead> {
        match self.spawn_logs() {
            Ok(readers) => readers,
            Err(e) => {
                warn!(error = %e, container = %self.id, "log follower failed to start; this run has no captured output");
                Vec::new()
            }
        }
    }

    async fn wait(&self) -> Result<i32, ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_wait_argv(exe, &self.id);
        let raw = run_checked(exe, &format!("wait {}", self.id), &argv).await?;
        raw.parse::<i32>().map_err(|_| {
            cli_err(format!(
                "{exe} wait {}: unparseable exit code `{raw}`",
                self.id
            ))
        })
    }

    async fn terminate(&self) -> Result<(), ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_kill_argv(exe, &self.id, "TERM");
        // An already-exited container is not an error: TERM's purpose was
        // to make it stop, and it has.
        run_idempotent(exe, &format!("kill --signal TERM {}", self.id), &argv).await
    }

    async fn kill(&self) -> Result<(), ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_kill_argv(exe, &self.id, "KILL");
        run_idempotent(exe, &format!("kill --signal KILL {}", self.id), &argv).await
    }

    async fn remove(&self) -> Result<(), ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_remove_argv(exe, &self.id);
        run_idempotent(exe, &format!("rm {}", self.id), &argv).await
    }
}

impl CliRunningContainer {
    fn spawn_logs(&self) -> Result<Vec<DynAsyncRead>, ExpectedError> {
        let exe = &self.executable;
        let argv = crate::container::render::render_logs_argv(exe, &self.id);
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // The follower outlives this call and is reaped when the container
        // goes away and it reaches EOF; if the supervisor drops first, the
        // handle takes it down rather than leaving it attached.
        cmd.kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .map_err(|e| cli_err(format!("{exe} logs {}: {e}", self.id)))?;
        let mut readers: Vec<DynAsyncRead> = Vec::with_capacity(2);
        if let Some(out) = child.stdout.take() {
            readers.push(Box::pin(out));
        }
        if let Some(err) = child.stderr.take() {
            readers.push(Box::pin(err));
        }
        if readers.is_empty() {
            return Err(cli_err(format!("{exe} logs {}: no output pipes", self.id)));
        }
        // Hand the child to the container handle. Letting it fall out of
        // scope here would kill the follower before a single line was read.
        *self.follower.lock() = Some(child);
        Ok(readers)
    }
}

/// Build an error naming the operation and carrying the runtime's own words.
fn cli_err(context: String) -> ExpectedError {
    ExpectedError::config_unparseable(std::path::PathBuf::from("<container>"), context)
}

/// The last non-empty line of the runtime's stderr, which is where both
/// Docker and Podman put the actual diagnosis.
fn diagnosis(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .unwrap_or("no output")
        .to_string()
}

/// Run a runtime CLI command, returning its trimmed stdout.
///
/// A non-zero exit is an error carrying the runtime's stderr. Treating one
/// as success is how "the runtime is unavailable" becomes indistinguishable
/// from "the container is gone" — and every recovery path in this crate
/// depends on telling those apart.
async fn run_checked(exe: &str, what: &str, argv: &[String]) -> Result<String, ExpectedError> {
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    // The command is short-lived and always awaited to completion here, but
    // the future can still be dropped by a `select!` losing its race.
    cmd.kill_on_drop(true);
    let output = cmd
        .output()
        .await
        .map_err(|e| cli_err(format!("{exe} {what}: {e}")))?;
    if !output.status.success() {
        return Err(cli_err(format!(
            "{exe} {what}: {}",
            diagnosis(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Run a command whose target may legitimately be gone already.
///
/// Removal and signalling are idempotent by intent: a container that has
/// already exited or been removed is the outcome the caller wanted. Any
/// other failure is real and surfaces.
async fn run_idempotent(exe: &str, what: &str, argv: &[String]) -> Result<(), ExpectedError> {
    match run_checked(exe, what, argv).await {
        Ok(_) => Ok(()),
        Err(e) if is_absent(&e.to_string()) => Ok(()),
        Err(e) => Err(e),
    }
}

/// Whether a runtime error means "that container does not exist".
fn is_absent(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("no such container")
        || m.contains("no such object")
        || m.contains("no container with")
}

/// Parse the fixed-arity `inspect --format` line.
///
/// Every field is a scalar and the last is a single label value, so the
/// arity is known and a `|` inside an operator's label cannot shift the
/// others — `splitn` keeps the tail intact.
fn parse_inspect_output(raw: &str) -> Option<ContainerInspect> {
    let line = raw.lines().map(str::trim).find(|l| !l.is_empty())?;
    let parts: Vec<&str> = line.splitn(6, '|').collect();
    if parts.len() < 6 {
        return None;
    }
    let id = parts[0].trim().to_string();
    if id.is_empty() {
        return None;
    }
    Some(ContainerInspect {
        id,
        // Docker prefixes the name with `/`; Podman does not.
        name: parts[1].trim().trim_start_matches('/').to_string(),
        state: parts[2].trim().to_string(),
        exit_code: parts[3].trim().parse::<i32>().ok(),
        host_pid: parts[4].trim().parse::<u32>().ok().filter(|p| *p > 0),
        // Go's templater prints `<no value>` for a missing key, and an
        // empty string for a container with no labels at all.
        owner: match parts[5].trim() {
            "" | "<no value>" => None,
            v => Some(v.to_string()),
        },
    })
}
