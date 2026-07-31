//! Linux-only: what the box and the binary are, recorded beside every row.
//!
//! A constant fitted from these measurements is specific to a moment, a machine,
//! and a binary. Several terms are hardware-specific rather than universal — the
//! CUDA runtime's host footprint scales with driver and device count, CPU-side
//! expert dequant with core count and memory topology — so a constant fitted on
//! one box is only transferable to another if you can tell the two apart.
//!
//! This is the production edge, deliberately: nothing in the pure core reads a
//! `/sys` file or shells out, and a test passes the [`Hardware`] and provenance it
//! wants rather than probing for them.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{
    harness::sys::Deps,
    record::{Cpu, Factors, Hardware, Provenance},
};

pub(crate) fn hardware(deps: &Deps) -> Hardware {
    Hardware {
        gpus: deps.gpu.devices(),
        cpu: Cpu {
            model: deps.procfs.cpu_model(),
            cores_per_socket: lscpu("Core(s) per socket"),
            sockets: lscpu("Socket(s)"),
            threads: lscpu("CPU(s)"),
            numa_nodes: lscpu("NUMA node(s)"),
        },
        mem_total_gib: (deps.procfs.mem_total_gib() * 10.0).round() / 10.0,
        kernel: run("uname", &["-r"]),
        // The tuning skill's first sanity check: a `powersave` governor pins cores
        // to the base clock and silently halves CPU-bound throughput.
        cpu_governor: sys_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        transparent_hugepage: sys_file("/sys/kernel/mm/transparent_hugepage/enabled"),
    }
}

/// Facts that make a stale row identifiable later.
///
/// The timestamp is what lets a later reader place a row in time — against a
/// llama.cpp pin bump, a driver update, or a change to the box — without having to
/// remember when anything happened. The binary's hash is what answers "was this
/// the same program", which is what separates drift in the runtime from error in a
/// fit, and what tells a contributor's build from ours.
pub(crate) fn provenance(deps: &Deps, binary: &str, factors: Option<&Factors>) -> Provenance {
    let resolved = resolve(binary);
    let mut provenance = Provenance {
        measured_at_utc: deps.clock.now_utc(),
        measured_at_local: deps.clock.now_local(),
        host: run("uname", &["-n"]),
        binary: resolved.to_string_lossy().into_owned(),
        ananke_rev: some_or_unknown(run("git", &["rev-parse", "--short", "HEAD"])),
        runtime_version: binary_version(&resolved),
        runtime_sha256: file_sha256(&resolved),
        ananke_dirty: if run("git", &["status", "--porcelain"]).is_empty() {
            "no".to_owned()
        } else {
            "yes".to_owned()
        },
        ..Provenance::default()
    };
    if let Some(factors) = factors {
        let model = Path::new(&factors.model);
        provenance.model_file_at = mtime(model);
        set_model_identity(model, &mut provenance);
    }
    provenance
}

/// What identifies a model to a reader who does not have this machine.
///
/// `factors.model` is an absolute path under whatever the operator set as the
/// model directory, which is useless for joining one contributor's rows to
/// another's. The repo-and-file suffix, the byte total across shards, and the
/// quant string are all portable.
fn set_model_identity(path: &Path, provenance: &mut Provenance) {
    let components: Vec<String> = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    provenance.model_key = if components.len() >= 3 {
        components[components.len() - 3..].join("/")
    } else {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    provenance.model_quant = quant(path);
    // Spelled as digits in a string, which is how every committed row has it.
    provenance.model_bytes = model_bytes(path).to_string();
}

/// The quant string, taken from the file name's last matching token. Last rather
/// than first: a repo name can carry a quant-shaped word before the file's own.
fn quant(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    stem.split(['-', '.'])
        .rfind(|token| QUANT.is_match(token))
        .map(str::to_owned)
        .unwrap_or_else(|| "?".to_owned())
}

/// Total size across every shard of a model, in bytes.
pub(crate) fn model_bytes(first: &Path) -> u64 {
    let Ok(metadata) = first.metadata() else {
        return 0;
    };
    let name = first
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = SHARD.replace(&name, "");
    if stem == name {
        return metadata.len();
    }
    // Every shard, not just the one the plan names: a fit gate that weighed only
    // the first of nine would let a 205 GiB model through as 23.
    let Some(parent) = first.parent() else {
        return metadata.len();
    };
    let Ok(entries) = parent.read_dir() else {
        return metadata.len();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with(stem.as_ref()) && name.ends_with(".gguf") && SHARD.is_match(&name)
        })
        .filter_map(|entry| entry.metadata().ok())
        .map(|metadata| metadata.len())
        .sum()
}

/// On-disk size of the model, shards included, in GiB.
pub(crate) fn model_gib(factors: &Factors) -> f64 {
    model_bytes(Path::new(&factors.model)) as f64 / 1024.0_f64.powi(3)
}

/// Whether a cell can run without risking the machine.
///
/// An unattended sweep that pushes the box into swap or the OOM killer costs far
/// more than the row it was trying to collect, so a cell that cannot fit with
/// headroom to spare is skipped and recorded as skipped. `--no-mmap` charges the
/// whole model to anonymous memory; a mapped load can rely on page cache being
/// reclaimable and needs only the host-resident share, which is not known ahead of
/// time — so the conservative figure is used for both.
///
/// That conservatism is also the gate's limit, and the reason `--force` exists.
/// The check weighs the *model file*, and for heavy expert offload that is the
/// wrong quantity: GLM-5.2's file is 205 GiB but its process peaks at 187 GiB of
/// anonymous memory, because the GPU-resident share never touches host RAM. The
/// gate cannot tell the difference, so a cell that has been measured before
/// becomes unmeasurable — hence an override rather than a looser rule.
pub(crate) fn fits(deps: &Deps, factors: &Factors, headroom_gib: f64) -> bool {
    model_gib(factors) + headroom_gib <= deps.procfs.mem_available_gib()
}

/// What the server reports about itself.
///
/// Custom forks report `version: 0 (unknown)` and a nix build normalises the
/// binary's mtime to the epoch, so neither identifies anything; the hash beside it
/// does. Recorded anyway because a non-nix build does report a version.
fn binary_version(path: &Path) -> String {
    let Ok(output) = std::process::Command::new(path).arg("--version").output() else {
        return "?".to_owned();
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    text.trim()
        .lines()
        .next()
        .map(|line| line.chars().take(200).collect())
        .unwrap_or_else(|| "?".to_owned())
}

/// The binary's hash: an exact identity even when it reports build 0.
fn file_sha256(path: &Path) -> String {
    let Ok(mut file) = std::fs::File::open(path) else {
        return "?".to_owned();
    };
    let mut digest = Sha256::new();
    if std::io::copy(&mut file, &mut digest).is_err() {
        return "?".to_owned();
    }
    format!("{:x}", digest.finalize())[..16].to_owned()
}

/// A file's modification time, as an ISO 8601 UTC timestamp.
fn mtime(path: &Path) -> String {
    let Ok(modified) = path.metadata().and_then(|metadata| metadata.modified()) else {
        return "?".to_owned();
    };
    let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH) else {
        return "?".to_owned();
    };
    let seconds = i64::try_from(since_epoch.as_secs()).unwrap_or_default();
    jiff::Timestamp::from_second(seconds)
        .map(|stamp| {
            stamp
                .to_zoned(jiff::tz::TimeZone::UTC)
                .strftime("%Y-%m-%dT%H:%M:%S%:z")
                .to_string()
        })
        .unwrap_or_else(|_| "?".to_owned())
}

/// Where the binary actually is, so a row names a file rather than a `PATH` hit
/// that may since have moved.
fn resolve(binary: &str) -> PathBuf {
    let path = Path::new(binary);
    if path.components().count() > 1 {
        return path.canonicalize().unwrap_or_else(|_| path.to_owned());
    }
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|directory| directory.join(binary))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| candidate.canonicalize().ok())
        .unwrap_or_else(|| path.to_owned())
}

fn lscpu(field: &str) -> Option<String> {
    let prefix = format!("{field}:");
    run("lscpu", &[])
        .lines()
        .find(|line| line.starts_with(&prefix))
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| value.trim().to_owned())
}

fn sys_file(path: &str) -> String {
    std::fs::read_to_string(path)
        .map(|text| text.trim().to_owned())
        .unwrap_or_else(|_| "?".to_owned())
}

/// A probe that is not available leaves the field empty rather than ending the
/// run: a box without `lscpu` still produces measurements.
fn run(program: &str, arguments: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn some_or_unknown(value: String) -> String {
    if value.is_empty() {
        "?".to_owned()
    } else {
        value
    }
}

static QUANT: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)^(UD_)?(I?Q\d[A-Z0-9_]*|BF16|F16|F32)$")
        .expect("the quant pattern is a literal")
});

/// llama.cpp's shard suffix, which is also what says a model has shards at all.
static SHARD: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"-\d{5}-of-\d{5}\.gguf$").expect("the shard pattern is a literal")
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::sys::{FakeGpu, FakeHttp, FakeProcFs, FakeSpawner, Fakes};

    /// The gate is arithmetic over two numbers, and both directions matter: the
    /// refusal is what keeps an unattended sweep off the OOM killer, and it is also
    /// what `--force` exists to override.
    #[test]
    fn the_fit_gate_weighs_the_model_against_what_is_available() {
        let fakes = Fakes::new(
            FakeSpawner::new(),
            FakeProcFs::new().with_available_gib(100.0),
            FakeGpu::new(),
            FakeHttp::new(),
        );
        let deps = fakes.deps();
        // No such file, so the model weighs nothing: only the headroom is in play.
        let factors = Factors {
            model: "/nowhere/model.gguf".to_owned(),
            ..Factors::default()
        };
        assert!(fits(&deps, &factors, 30.0));
        assert!(!fits(&deps, &factors, 200.0));
    }

    #[test]
    fn the_quant_comes_from_the_file_name_not_the_repository() {
        // The repo says Q8_0 and the file says Q4_K_M; the file wins, which is
        // what the row has to say to be joinable against another contributor's.
        for (path, expected) in [
            (
                "/m/unsloth/GLM-5.2-GGUF/GLM-5.2-UD-Q4_K_XL-00001-of-00005.gguf",
                "Q4_K_XL",
            ),
            (
                "/m/LiquidAI/LFM2.5-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf",
                "Q8_0",
            ),
            ("/m/x/y/model-BF16.gguf", "BF16"),
            ("/m/x/y/no-quant-here.gguf", "?"),
        ] {
            assert_eq!(quant(Path::new(path)), expected, "{path}");
        }
    }

    #[test]
    fn the_model_key_is_the_portable_suffix_of_the_path() {
        // The last three components, which is what joins one contributor's rows to
        // another's: everything above them is whatever that operator set as the
        // model directory.
        let mut identity = Provenance::default();
        set_model_identity(
            Path::new(
                "/nowhere/at/all/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf",
            ),
            &mut identity,
        );
        assert_eq!(
            identity.model_key,
            "LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf"
        );
        assert_eq!(identity.model_quant, "Q8_0");
        // No such file, so the byte total is zero rather than a guess.
        assert_eq!(identity.model_bytes, "0");
    }
}
