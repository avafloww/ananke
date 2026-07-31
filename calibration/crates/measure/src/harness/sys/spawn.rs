//! Linux-only: starts a llama-server in its own process group and signals it by
//! pid.
//!
//! Deliberately not `ananke`'s `ProcessSpawner`. That trait is async, carries
//! the supervisor's pdeathsig and log-capture machinery, and is shaped for a
//! long-lived child the daemon keeps alive. The harness wants the opposite: one
//! short-lived process, driven synchronously, whose whole output is a file it
//! reads back after the process is gone.
//!
//! A run is stopped by **pid**, never by pattern. `pkill -f llama-server` also
//! matches the shell that is driving the campaign, and the process group is what
//! catches anything the server started for itself.

use std::{
    io,
    path::{Path, PathBuf},
};

/// What to start, in full. The environment is an overlay on the harness's own,
/// because `CUDA_VISIBLE_DEVICES` is how a cell selects its cards and everything
/// else the operator exported (`LD_LIBRARY_PATH` for the driver, above all) has
/// to survive.
pub struct SpawnRequest<'a> {
    pub argv: &'a [String],
    pub env: Vec<(String, String)>,
    /// stdout and stderr are merged into this file: the loader writes the buffer
    /// sizes to stderr and the request log to stdout, and the parser reads both.
    pub log_path: &'a Path,
}

pub trait Spawner: Send + Sync {
    fn spawn(&self, request: SpawnRequest<'_>) -> io::Result<Box<dyn Child>>;
}

pub trait Child: Send {
    fn pid(&self) -> u32;
    /// The exit status, or `None` while the process is still running. Reaps, so
    /// a finished child does not linger as a zombie holding its port.
    fn exit_status(&mut self) -> Option<i32>;
    fn signal(&mut self, signal: Stop);
    /// Everything the process has written so far.
    fn log(&self) -> String;
}

/// The two-step stop every supervisor ends up with: ask, then insist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Term,
    Kill,
}

pub struct LocalSpawner;

impl Spawner for LocalSpawner {
    fn spawn(&self, request: SpawnRequest<'_>) -> io::Result<Box<dyn Child>> {
        let (program, arguments) = request
            .argv
            .split_first()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "empty argv"))?;
        if let Some(parent) = request.log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let log = std::fs::File::create(request.log_path)?;
        let errors = log.try_clone()?;
        let mut command = std::process::Command::new(program);
        command
            .args(arguments)
            .stdout(log)
            .stderr(errors)
            .stdin(std::process::Stdio::null());
        for (key, value) in request.env {
            command.env(key, value);
        }
        // Its own group, so one `killpg` reaches anything the server forked and
        // nothing reaches the campaign driver.
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let child = command.spawn()?;
        Ok(Box::new(LocalChild {
            child,
            log_path: request.log_path.to_owned(),
            status: None,
        }))
    }
}

struct LocalChild {
    child: std::process::Child,
    log_path: PathBuf,
    /// Cached, because `try_wait` reports a status once and then reaps.
    status: Option<i32>,
}

impl Child for LocalChild {
    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn exit_status(&mut self) -> Option<i32> {
        if self.status.is_some() {
            return self.status;
        }
        match self.child.try_wait() {
            // A signalled process has no exit code; the signal number stands in
            // for it so the status is never silently absent.
            Ok(Some(status)) => {
                self.status = Some(status.code().unwrap_or_else(|| {
                    std::os::unix::process::ExitStatusExt::signal(&status)
                        .map_or(-1, |signal| -signal)
                }));
                self.status
            }
            Ok(None) => None,
            // An un-waitable child is gone as far as this harness is concerned;
            // treating it as running would hang the phase that is waiting.
            Err(_) => {
                self.status = Some(-1);
                self.status
            }
        }
    }

    fn signal(&mut self, signal: Stop) {
        let signal = match signal {
            Stop::Term => nix::sys::signal::Signal::SIGTERM,
            Stop::Kill => nix::sys::signal::Signal::SIGKILL,
        };
        // The group id equals the pid because the spawn made the child a group
        // leader; a failure here means the group is already gone.
        let group = nix::unistd::Pid::from_raw(i32::try_from(self.child.id()).unwrap_or(-1));
        let _ = nix::sys::signal::killpg(group, signal);
    }

    fn log(&self) -> String {
        // Lossily: a truncated multi-byte sequence at the end of a killed run's
        // log is not a reason to lose the rest of it.
        std::fs::read(&self.log_path)
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default()
    }
}

/// A process that never existed: it holds a canned log, exits when told to, and
/// records every signal it was sent.
#[cfg(any(test, feature = "test-fakes"))]
pub struct FakeSpawner {
    processes: parking_lot::Mutex<Vec<std::sync::Arc<parking_lot::Mutex<FakeProcess>>>>,
    template: parking_lot::Mutex<FakeProcess>,
}

/// One fake process's whole observable state, so an assertion can ask what it
/// was started with and how it was stopped.
#[cfg(any(test, feature = "test-fakes"))]
#[derive(Debug, Clone, Default)]
pub struct FakeProcess {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub log: String,
    pub status: Option<i32>,
    pub signals: Vec<Stop>,
    /// The case that motivates the escalation: a wedged server that has to be
    /// `SIGKILL`ed before the next cell can bind the port.
    pub ignores_term: bool,
}

#[cfg(any(test, feature = "test-fakes"))]
impl Default for FakeSpawner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl FakeSpawner {
    pub fn new() -> Self {
        Self {
            processes: parking_lot::Mutex::new(Vec::new()),
            template: parking_lot::Mutex::new(FakeProcess::default()),
        }
    }

    /// What the next spawned process will have written.
    pub fn with_log(self, log: &str) -> Self {
        self.template.lock().log = log.to_owned();
        self
    }

    /// A process that is already gone when first polled — the `failed-to-load`
    /// path.
    pub fn exited_with(self, status: i32) -> Self {
        self.template.lock().status = Some(status);
        self
    }

    pub fn ignoring_term(self) -> Self {
        self.template.lock().ignores_term = true;
        self
    }

    pub fn processes(&self) -> Vec<FakeProcess> {
        self.processes
            .lock()
            .iter()
            .map(|process| process.lock().clone())
            .collect()
    }
}

#[cfg(any(test, feature = "test-fakes"))]
impl Spawner for FakeSpawner {
    fn spawn(&self, request: SpawnRequest<'_>) -> io::Result<Box<dyn Child>> {
        let mut process = self.template.lock().clone();
        process.argv = request.argv.to_vec();
        process.env = request.env;
        let shared = std::sync::Arc::new(parking_lot::Mutex::new(process));
        let mut processes = self.processes.lock();
        processes.push(shared.clone());
        Ok(Box::new(FakeChild {
            // Virtual, and deliberately far from any pid the box would hand
            // out, so a fake pid that escaped into a real syscall is obvious.
            pid: 900_000 + processes.len() as u32,
            process: shared,
        }))
    }
}

#[cfg(any(test, feature = "test-fakes"))]
struct FakeChild {
    pid: u32,
    process: std::sync::Arc<parking_lot::Mutex<FakeProcess>>,
}

#[cfg(any(test, feature = "test-fakes"))]
impl Child for FakeChild {
    fn pid(&self) -> u32 {
        self.pid
    }

    fn exit_status(&mut self) -> Option<i32> {
        self.process.lock().status
    }

    fn signal(&mut self, signal: Stop) {
        let mut process = self.process.lock();
        process.signals.push(signal);
        match signal {
            Stop::Term if process.ignores_term => {}
            Stop::Term => process.status = Some(0),
            Stop::Kill => process.status = Some(-9),
        }
    }

    fn log(&self) -> String {
        self.process.lock().log.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fake_child_records_its_argv_and_its_signals() {
        let spawner = FakeSpawner::new().with_log("loaded").ignoring_term();
        let argv = vec![
            "llama-server".to_owned(),
            "-c".to_owned(),
            "8192".to_owned(),
        ];
        let mut child = spawner
            .spawn(SpawnRequest {
                argv: &argv,
                env: vec![("CUDA_VISIBLE_DEVICES".to_owned(), "0".to_owned())],
                log_path: Path::new("/dev/null"),
            })
            .expect("the fake spawner never fails");
        assert_eq!(child.log(), "loaded");
        assert_eq!(child.exit_status(), None);
        child.signal(Stop::Term);
        assert_eq!(child.exit_status(), None, "this fake ignores SIGTERM");
        child.signal(Stop::Kill);
        assert_eq!(child.exit_status(), Some(-9));

        let processes = spawner.processes();
        assert_eq!(processes[0].argv, argv);
        assert_eq!(processes[0].signals, vec![Stop::Term, Stop::Kill]);
    }
}
