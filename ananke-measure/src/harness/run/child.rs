//! Starting and stopping the server under measurement.
//!
//! The stop is the part that matters operationally. A server must be *fully*
//! exited before the next run starts: a leftover process wins the port bind, and
//! every later cell then measures that same process at every point, which is how a
//! whole sweep is invalidated with nothing in the log to say so. So the stop asks
//! and then insists, and it waits for the exit rather than assuming it.

use std::{io, path::Path, time::Duration};

use crate::{
    harness::{
        cell,
        sys::{Child, Deps, SpawnRequest, Stop},
    },
    record::Factors,
};

/// How long a `SIGTERM`'d server is given. Generous, because tearing down a
/// 200 GiB mapping is not instant and the memory-breakdown table it prints on the
/// way out is part of the measurement.
const TERM_GRACE: Duration = Duration::from_secs(120);
const KILL_GRACE: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_secs(1);

pub(crate) fn spawn_server(
    deps: &Deps,
    factors: &Factors,
    binary: &str,
    log_path: &Path,
    port: u16,
) -> io::Result<Box<dyn Child>> {
    let argv = cell::argv(factors, binary, port);
    deps.spawner.spawn(SpawnRequest {
        argv: &argv,
        // The cell's card selection, as an overlay: everything else the operator
        // exported has to survive, `LD_LIBRARY_PATH` for the driver above all.
        env: vec![("CUDA_VISIBLE_DEVICES".to_owned(), factors.gpus.clone())],
        log_path,
    })
}

/// Stop a server, escalating if it will not go.
pub(crate) fn stop_child(deps: &Deps, child: &mut dyn Child) {
    if child.exit_status().is_some() {
        return;
    }
    child.signal(Stop::Term);
    if wait_for_exit(deps, child, TERM_GRACE) {
        return;
    }
    child.signal(Stop::Kill);
    wait_for_exit(deps, child, KILL_GRACE);
}

fn wait_for_exit(deps: &Deps, child: &mut dyn Child, timeout: Duration) -> bool {
    let deadline = deps.clock.elapsed() + timeout;
    while deps.clock.elapsed() < deadline {
        if child.exit_status().is_some() {
            return true;
        }
        deps.clock.sleep(POLL);
    }
    child.exit_status().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes},
        record::Runtime,
    };

    fn fakes(spawner: FakeSpawner) -> Fakes {
        Fakes::new(spawner, FakeProcFs::new(), FakeGpu::new(), FakeHttp::new())
    }

    #[test]
    fn the_spawn_carries_the_cells_cards_and_its_command_line() {
        let fakes = fakes(FakeSpawner::new());
        let factors = Factors {
            gpus: "0,1".to_owned(),
            runtime: Runtime::Ik,
            ..Factors::default()
        };
        spawn_server(
            &fakes.deps(),
            &factors,
            "ik-llama-server",
            Path::new("/dev/null"),
            18099,
        )
        .expect("the fake spawner never fails");

        let processes = fakes.spawner.processes();
        assert_eq!(
            processes[0].env,
            vec![("CUDA_VISIBLE_DEVICES".to_owned(), "0,1".to_owned())]
        );
        assert!(processes[0].argv.contains(&"--port".to_owned()));
        assert_eq!(processes[0].argv[0], "ik-llama-server");
    }

    #[test]
    fn a_server_that_exits_on_sigterm_is_not_killed() {
        let fakes = fakes(FakeSpawner::new());
        let deps = fakes.deps();
        let mut child = spawn_server(
            &deps,
            &Factors::default(),
            "llama-server",
            Path::new("/dev/null"),
            18099,
        )
        .expect("the fake spawner never fails");
        stop_child(&deps, child.as_mut());
        assert_eq!(fakes.spawner.processes()[0].signals, vec![Stop::Term]);
    }

    /// The case the escalation exists for: without it the port stays held and the
    /// next cell measures this process.
    #[test]
    fn a_wedged_server_is_killed_after_the_grace_period() {
        let fakes = fakes(FakeSpawner::new().ignoring_term());
        let deps = fakes.deps();
        let mut child = spawn_server(
            &deps,
            &Factors::default(),
            "llama-server",
            Path::new("/dev/null"),
            18099,
        )
        .expect("the fake spawner never fails");
        stop_child(&deps, child.as_mut());
        assert_eq!(
            fakes.spawner.processes()[0].signals,
            vec![Stop::Term, Stop::Kill]
        );
        assert!(deps.clock.elapsed() >= TERM_GRACE);
        assert_eq!(child.exit_status(), Some(-9));
    }

    #[test]
    fn a_server_that_has_already_exited_is_left_alone() {
        let fakes = fakes(FakeSpawner::new().exited_with(1));
        let deps = fakes.deps();
        let mut child = spawn_server(
            &deps,
            &Factors::default(),
            "llama-server",
            Path::new("/dev/null"),
            18099,
        )
        .expect("the fake spawner never fails");
        stop_child(&deps, child.as_mut());
        assert!(fakes.spawner.processes()[0].signals.is_empty());
    }
}
