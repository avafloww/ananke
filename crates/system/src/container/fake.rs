//! In-memory container engine for deterministic tests. Records each
//! create/start/wait/TERM/KILL/remove operation so tests can assert on
//! operation ordering and script outcomes without invoking Docker or Podman.

use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicU32},
};

use ananke_errors::ExpectedError;
use ananke_spawn::ContainerSpec;
use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::container::{
    PreparedContainer,
    types::{ContainerEngine, ContainerInspect, ContainerSummary, DynAsyncRead, ManagedContainer},
};

/// Externally-visible state of a fake container. Tests snapshot the vector
/// returned by [`FakeContainerEngine::snapshot`] and assert on transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeContainerState {
    /// Container created but not started.
    Created,
    /// Container started and running.
    Running,
    /// Container terminated via SIGTERM.
    Terminated,
    /// Container killed via SIGKILL.
    Killed,
    /// Container removed.
    Removed,
    /// Test-injected exit code.
    Exited { code: i32 },
}

/// Externally-visible snapshot of a fake container.
#[derive(Debug, Clone)]
pub struct FakeContainerSnapshot {
    pub id: String,
    pub name: String,
    pub image: String,
    pub command: Vec<String>,
    pub state: FakeContainerState,
    /// Records every lifecycle operation in order.
    pub operations: Vec<String>,
    pub exit_code: Option<i32>,
}

struct Inner {
    next_id: AtomicU32,
    slots: Vec<Slot>,
    /// Scripted outcomes for `create`: entries are consumed in order, `None`
    /// means succeed.
    create_outcomes: Vec<Option<String>>,
    /// Scripted outcomes for `start`.
    start_outcomes: Vec<bool>,
    /// Scripted outcomes for `wait`: `Some(code)` means exit, `None` means block.
    wait_outcomes: Vec<Option<i32>>,
    /// Scripted outcomes for TERM.
    term_outcomes: Vec<bool>,
    /// Scripted outcomes for KILL.
    kill_outcomes: Vec<bool>,
    /// Scripted outcomes for remove.
    remove_outcomes: Vec<bool>,
}

struct Slot {
    id: String,
    name: String,
    image: String,
    command: Vec<String>,
    labels: BTreeMap<String, String>,
    state: FakeContainerState,
    operations: Vec<String>,
    exit_code: Option<i32>,
    host_pid: Option<u32>,
}

impl Slot {
    fn snapshot(&self) -> FakeContainerSnapshot {
        FakeContainerSnapshot {
            id: self.id.clone(),
            name: self.name.clone(),
            image: self.image.clone(),
            command: self.command.clone(),
            state: self.state.clone(),
            operations: self.operations.clone(),
            exit_code: self.exit_code,
        }
    }
}

/// In-memory container engine. Every `create` records a new slot; tests
/// inspect state via [`snapshot`].
pub struct FakeContainerEngine {
    inner: Arc<Mutex<Inner>>,
    /// Woken whenever a container's exit code is set. A blocked `wait` parks
    /// on this rather than polling: a spin loop keeps the runtime busy, which
    /// stops `start_paused` tests from ever advancing their timers.
    exited: Arc<Notify>,
}

impl FakeContainerEngine {
    pub fn new() -> Self {
        Self {
            exited: Arc::new(Notify::new()),
            inner: Arc::new(Mutex::new(Inner {
                next_id: AtomicU32::new(1),
                slots: Vec::new(),
                create_outcomes: Vec::new(),
                start_outcomes: Vec::new(),
                wait_outcomes: Vec::new(),
                term_outcomes: Vec::new(),
                kill_outcomes: Vec::new(),
                remove_outcomes: Vec::new(),
            })),
        }
    }

    /// Snapshot every container this engine has created, in order.
    pub fn snapshot(&self) -> Vec<FakeContainerSnapshot> {
        self.inner
            .lock()
            .slots
            .iter()
            .map(|s| s.snapshot())
            .collect()
    }

    /// Find a slot by ID.
    pub fn find(&self, id: &str) -> Option<FakeContainerSnapshot> {
        self.snapshot().into_iter().find(|c| c.id == id)
    }

    /// Script the next `create` to fail with the given error string.
    pub fn fail_create(&self, error: &str) {
        self.inner.lock().create_outcomes.push(Some(error.into()));
    }

    /// Set create to succeed.
    pub fn succeed_create(&self) {
        self.inner.lock().create_outcomes.push(None);
    }

    /// Script the next `start` to fail or succeed.
    pub fn fail_start(&self) {
        self.inner.lock().start_outcomes.push(false);
    }

    pub fn succeed_start(&self) {
        self.inner.lock().start_outcomes.push(true);
    }

    /// Script the next `wait` to return the given exit code, or `None` to block.
    pub fn wait_exit(&self, code: i32) {
        self.inner.lock().wait_outcomes.push(Some(code));
    }

    pub fn wait_block(&self) {
        self.inner.lock().wait_outcomes.push(None);
    }

    /// Script the next `remove` to fail, so a test can exercise the
    /// reconciliation block that failed cleanup is supposed to leave behind.
    pub fn fail_remove(&self) {
        self.inner.lock().remove_outcomes.push(false);
    }

    pub fn succeed_remove(&self) {
        self.inner.lock().remove_outcomes.push(true);
    }

    /// Force a container to exit with a code.
    pub fn exit(&self, id: &str, code: i32) -> bool {
        let mut inner = self.inner.lock();
        let Some(slot) = inner.slots.iter_mut().find(|s| s.id == id) else {
            return false;
        };
        if matches!(slot.state, FakeContainerState::Running) {
            slot.state = FakeContainerState::Exited { code };
            slot.exit_code = Some(code);
            drop(inner);
            self.exited.notify_waiters();
            true
        } else {
            false
        }
    }
}

impl Default for FakeContainerEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContainerEngine for FakeContainerEngine {
    /// One fake engine backs every runtime: tests script outcomes on the
    /// shared state rather than per binary.
    fn for_executable(&self, _executable: &str) -> Arc<dyn ContainerEngine> {
        Arc::new(self.clone())
    }

    async fn create(&self, spec: &ContainerSpec) -> Result<PreparedContainer, ExpectedError> {
        let outcome = {
            let mut inner = self.inner.lock();
            let outcome = inner.create_outcomes.pop();
            inner
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            outcome
        };
        if let Some(Some(err)) = outcome {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                err,
            ));
        }
        let id = {
            let mut inner = self.inner.lock();
            let n = inner
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let id = format!("fake{n:04x}");
            inner.slots.push(Slot {
                id: id.clone(),
                name: spec.name.clone(),
                image: spec.image.clone(),
                command: spec.command.clone(),
                labels: spec.labels.clone(),
                state: FakeContainerState::Created,
                operations: vec!["create".into()],
                exit_code: None,
                host_pid: None,
            });
            id
        };
        Ok(PreparedContainer {
            id,
            name: spec.name.clone(),
            runtime_executable: spec
                .runtime_executable
                .clone()
                .unwrap_or_else(|| spec.runtime.executable().into()),
            runtime: spec.runtime,
            engine: std::sync::Arc::new(self.clone()),
        })
    }

    async fn start(
        &self,
        prepared: &PreparedContainer,
    ) -> Result<Box<dyn ManagedContainer>, ExpectedError> {
        let should_succeed = {
            let mut inner = self.inner.lock();
            let outcome = inner.start_outcomes.pop().unwrap_or(true);
            if outcome {
                let Some(slot) = inner.slots.iter_mut().find(|s| s.id == prepared.id) else {
                    return Err(ExpectedError::config_unparseable(
                        std::path::PathBuf::from("<fake>"),
                        "start: unknown container id".to_string(),
                    ));
                };
                slot.state = FakeContainerState::Running;
                slot.operations.push("start".into());
                slot.host_pid = Some(3000 + prepared.id.len() as u32);
            }
            outcome
        };
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "start failed".to_string(),
            ));
        }
        let host_pid = {
            let inner = self.inner.lock();
            inner
                .slots
                .iter()
                .find(|s| s.id == prepared.id)
                .and_then(|s| s.host_pid)
        };
        Ok(Box::new(FakeRunningContainer {
            engine: self.clone(),
            id: prepared.id.clone(),
            name: prepared.name.clone(),
            runtime_executable: prepared.runtime_executable.clone(),
            host_pid,
        }))
    }

    async fn remove_prepared(&self, prepared: &PreparedContainer) -> Result<(), ExpectedError> {
        let should_succeed = {
            let mut inner = self.inner.lock();
            let outcome = inner.remove_outcomes.pop().unwrap_or(true);
            if outcome && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == prepared.id) {
                slot.state = FakeContainerState::Removed;
                slot.operations.push("remove".into());
            }
            outcome
        };
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "remove failed".to_string(),
            ));
        }
        Ok(())
    }

    async fn inspect(&self, id: &str) -> Result<ContainerInspect, ExpectedError> {
        let inner = self.inner.lock();
        let slot = inner
            .slots
            .iter()
            .find(|s| s.id == id)
            .filter(|s| s.state != FakeContainerState::Removed);
        let Some(slot) = slot else {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                format!("inspect {id}: not found"),
            ));
        };
        let (state_str, exit_code) = match &slot.state {
            FakeContainerState::Created => ("created".into(), None),
            FakeContainerState::Running => ("running".into(), None),
            FakeContainerState::Terminated => ("exited".into(), Some(0)),
            FakeContainerState::Killed => ("exited".into(), Some(137)),
            FakeContainerState::Removed => ("removed".into(), None),
            FakeContainerState::Exited { code } => ("exited".into(), Some(*code)),
        };
        Ok(ContainerInspect {
            id: slot.id.clone(),
            name: slot.name.clone(),
            state: state_str,
            exit_code,
            host_pid: slot.host_pid,
            owner: slot
                .labels
                .get(crate::container::render::OWNER_LABEL)
                .cloned(),
        })
    }

    async fn remove(&self, id: &str) -> Result<(), ExpectedError> {
        let should_succeed = {
            let mut inner = self.inner.lock();
            let outcome = inner.remove_outcomes.pop().unwrap_or(true);
            if outcome && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == id) {
                slot.state = FakeContainerState::Removed;
                slot.operations.push("remove".into());
            }
            outcome
        };
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "remove failed".to_string(),
            ));
        }
        Ok(())
    }

    async fn list(&self, filters: &[String]) -> Result<Vec<ContainerSummary>, ExpectedError> {
        let inner = self.inner.lock();
        let mut out = Vec::new();
        for slot in &inner.slots {
            if slot.state == FakeContainerState::Removed {
                continue;
            }
            // Honour `label=key=value` filters. Reconciliation scopes its
            // lookups by the owner label, and a fake that ignored filters
            // would let that scoping silently regress.
            if !filters.iter().all(|f| slot_matches_filter(slot, f)) {
                continue;
            }
            let (state_str, _) = match &slot.state {
                FakeContainerState::Created => ("created".into(), None),
                FakeContainerState::Running => ("running".into(), None),
                FakeContainerState::Terminated => ("exited".into(), Some(0)),
                FakeContainerState::Killed => ("exited".into(), Some(137)),
                FakeContainerState::Removed => ("removed".into(), None),
                FakeContainerState::Exited { code } => ("exited".into(), Some(*code)),
            };
            out.push(ContainerSummary {
                id: slot.id.clone(),
                name: slot.name.clone(),
                state: state_str,
                owner: slot
                    .labels
                    .get(crate::container::render::OWNER_LABEL)
                    .cloned(),
            });
        }
        Ok(out)
    }
}

impl Clone for FakeContainerEngine {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            exited: Arc::clone(&self.exited),
        }
    }
}

/// Whether a slot satisfies one `--filter` expression. Only `label=k=v` is
/// understood; any other expression matches everything, mirroring the fact
/// that reconciliation only ever filters by label.
fn slot_matches_filter(slot: &Slot, filter: &str) -> bool {
    let Some(expr) = filter.strip_prefix("label=") else {
        return true;
    };
    match expr.split_once('=') {
        Some((key, value)) => slot.labels.get(key).map(String::as_str) == Some(value),
        None => slot.labels.contains_key(expr),
    }
}

/// Fake running-container handle.
pub struct FakeRunningContainer {
    engine: FakeContainerEngine,
    id: String,
    name: String,
    runtime_executable: String,
    host_pid: Option<u32>,
}

#[async_trait]
impl ManagedContainer for FakeRunningContainer {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn runtime_executable(&self) -> &str {
        &self.runtime_executable
    }

    fn host_pid(&self) -> Option<u32> {
        self.host_pid
    }

    fn logs(&self) -> Vec<DynAsyncRead> {
        vec![Box::pin(tokio::io::empty())]
    }

    async fn wait(&self) -> Result<i32, ExpectedError> {
        let outcome = {
            let mut inner = self.engine.inner.lock();
            // Pop a scripted outcome if any; default to exiting 0.
            let outcome = inner.wait_outcomes.pop().unwrap_or(Some(0));
            if let Some(code) = outcome
                && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == self.id)
            {
                slot.state = FakeContainerState::Exited { code };
                slot.operations.push("wait".into());
                slot.exit_code = Some(code);
            }
            outcome
        };
        match outcome {
            Some(code) => Ok(code),
            None => {
                // Block until something sets an exit code — `exit`, or a
                // KILL. Parking on the notify (rather than polling) leaves
                // the runtime idle, which is what lets a `start_paused` test
                // advance to its drain deadline.
                loop {
                    let waiter = self.engine.exited.notified();
                    {
                        let inner = self.engine.inner.lock();
                        if let Some(slot) = inner.slots.iter().find(|s| s.id == self.id)
                            && let Some(code) = slot.exit_code
                        {
                            return Ok(code);
                        }
                    }
                    waiter.await;
                }
            }
        }
    }

    async fn terminate(&self) -> Result<(), ExpectedError> {
        let should_succeed = {
            let mut inner = self.engine.inner.lock();
            let outcome = inner.term_outcomes.pop().unwrap_or(true);
            if outcome && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == self.id) {
                slot.state = FakeContainerState::Terminated;
                slot.operations.push("terminate".into());
            }
            outcome
        };
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "terminate failed".to_string(),
            ));
        }
        Ok(())
    }

    async fn kill(&self) -> Result<(), ExpectedError> {
        let should_succeed = {
            let mut inner = self.engine.inner.lock();
            let outcome = inner.kill_outcomes.pop().unwrap_or(true);
            if outcome && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == self.id) {
                slot.state = FakeContainerState::Killed;
                slot.operations.push("kill".into());
                // A killed container exits; SIGKILL is 9, so 137 the way
                // both runtimes report it.
                slot.exit_code = Some(137);
            }
            outcome
        };
        self.engine.exited.notify_waiters();
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "kill failed".to_string(),
            ));
        }
        Ok(())
    }

    async fn remove(&self) -> Result<(), ExpectedError> {
        let should_succeed = {
            let mut inner = self.engine.inner.lock();
            let outcome = inner.remove_outcomes.pop().unwrap_or(true);
            if outcome && let Some(slot) = inner.slots.iter_mut().find(|s| s.id == self.id) {
                slot.state = FakeContainerState::Removed;
                slot.operations.push("remove".into());
            }
            outcome
        };
        if !should_succeed {
            return Err(ExpectedError::config_unparseable(
                std::path::PathBuf::from("<fake>"),
                "remove failed".to_string(),
            ));
        }
        Ok(())
    }
}
