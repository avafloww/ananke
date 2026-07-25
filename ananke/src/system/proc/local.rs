//! Linux-only: the real `/proc` reader plus its text parsers.
//!
//! Every method shells out to `std::fs` and re-parses the synthesised
//! kernel text on each call; the parsing helpers are kept free-standing
//! so their kernel-format assumptions (NUL-separated cmdline, the
//! parenthesised `comm` field in `/proc/<pid>/stat`, etc.) are testable
//! without touching the filesystem.

use std::io;

use crate::system::proc::{Meminfo, ProcFs};

/// Real `/proc` reader. Every method shells out to `std::fs`.
#[derive(Default, Clone, Copy)]
pub struct LocalProcFs;

impl ProcFs for LocalProcFs {
    fn meminfo(&self) -> io::Result<Meminfo> {
        let content = std::fs::read_to_string("/proc/meminfo")?;
        parse_meminfo(&content)
            .ok_or_else(|| io::Error::other("meminfo missing MemTotal or MemAvailable"))
    }

    fn vm_rss(&self, pid: u32) -> Option<u64> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        parse_vm_rss(&content)
    }

    fn comm(&self, pid: u32) -> Option<String> {
        std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .ok()
            .map(|s| s.trim().to_string())
    }

    fn cmdline(&self, pid: i32) -> Option<String> {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        Some(null_sep_to_space(&raw))
    }

    fn parent_pid(&self, pid: u32) -> Option<u32> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        parse_parent_pid(&content)
    }

    fn all_pids(&self) -> Vec<u32> {
        let dir = match std::fs::read_dir("/proc") {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        dir.filter_map(|entry| {
            let entry = entry.ok()?;
            entry.file_name().to_str()?.parse::<u32>().ok()
        })
        .collect()
    }

    fn cgroup_path(&self, pid: u32) -> Option<String> {
        let content = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
        parse_cgroup_v2(&content)
    }
}

fn parse_meminfo(content: &str) -> Option<Meminfo> {
    let mut total_kb = None;
    let mut avail_kb = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_kb(rest);
        }
    }
    Some(Meminfo {
        total_bytes: total_kb? * 1024,
        available_bytes: avail_kb? * 1024,
    })
}

/// Parse `/proc/<pid>/stat` field 4 (parent pid). The `comm` field at
/// index 1 is parenthesised and may itself contain spaces, parens, or
/// other punctuation, so a naive whitespace split misattributes the
/// later columns. The fix is to scan for the **last** `)` and split the
/// remainder; ppid is then the second whitespace-separated token (the
/// first being `state`).
fn parse_parent_pid(stat: &str) -> Option<u32> {
    let close = stat.rfind(')')?;
    let tail = stat.get(close + 1..)?;
    tail.split_whitespace().nth(1)?.parse::<u32>().ok()
}

/// Parse `/proc/<pid>/cgroup`. Returns the v2 unified-hierarchy path
/// (the value after `0::`); `None` when no `0::` line is present (cgroup
/// v1 hosts, or pid exited).
fn parse_cgroup_v2(content: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            return Some(rest.trim_end().to_string());
        }
    }
    None
}

fn parse_vm_rss(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn parse_kb(rest: &str) -> Option<u64> {
    let trimmed = rest.trim().trim_end_matches("kB").trim();
    trimmed.parse::<u64>().ok()
}

fn null_sep_to_space(bytes: &[u8]) -> String {
    let mut s: String = bytes
        .iter()
        .map(|b| if *b == 0 { ' ' } else { *b as char })
        .collect();
    // The kernel emits a trailing NUL after the last arg, which becomes a
    // trailing space here.
    s.truncate(s.trim_end().len());
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MEMINFO: &str = "\
MemTotal:       98765432 kB
MemFree:        12345678 kB
MemAvailable:   87654321 kB
Buffers:        1000000 kB
";

    const SAMPLE_STATUS: &str = "\
Name:\tllama-server
VmPeak:\t 1000000 kB
VmRSS:\t  524288 kB
";

    #[test]
    fn parses_meminfo() {
        let m = parse_meminfo(SAMPLE_MEMINFO).unwrap();
        assert_eq!(m.total_bytes, 98_765_432 * 1024);
        assert_eq!(m.available_bytes, 87_654_321 * 1024);
    }

    #[test]
    fn parses_vm_rss() {
        assert_eq!(parse_vm_rss(SAMPLE_STATUS), Some(524_288 * 1024));
        assert_eq!(parse_vm_rss("Name:\tfoo\n"), None);
    }

    #[test]
    fn null_sep_joins() {
        // `/proc/<pid>/cmdline` is NUL-separated with a trailing NUL.
        assert_eq!(
            null_sep_to_space(b"llama-server\0-m\0model.gguf\0"),
            "llama-server -m model.gguf"
        );
    }

    /// Field 2 (`comm`) is parenthesised; the only safe parser scans to
    /// the last `)`. The `kthreadd` worker name `(sd-pam)` and the
    /// llama-server pattern `llama-server` both round-trip correctly.
    #[test]
    fn parses_parent_pid_with_simple_comm() {
        let stat = "1234 (llama-server) S 4321 1234 1234 0 -1 4194304 ...";
        assert_eq!(parse_parent_pid(stat), Some(4321));
    }

    /// Pathological `comm` values: the kernel allows up to 15 bytes of any
    /// printable character including spaces, parens, and trailing `)`. The
    /// "scan to last `)`" rule must isolate the structural close.
    #[test]
    fn parses_parent_pid_with_parens_in_comm() {
        let stat = "42 ((sd-pam)) S 1 42 42 0 -1 ...";
        assert_eq!(parse_parent_pid(stat), Some(1));
    }

    #[test]
    fn parses_parent_pid_with_spaces_in_comm() {
        let stat = "99 (foo bar) S 7 99 99 0 -1 ...";
        assert_eq!(parse_parent_pid(stat), Some(7));
    }

    /// Cgroup v2 unified hierarchy: a single line `0::<path>`. v1 hosts
    /// emit `<n>:<controller>:<path>` for every controller; we ignore those.
    #[test]
    fn parses_cgroup_v2_path() {
        let content = "0::/system.slice/docker-abc.scope\n";
        assert_eq!(
            parse_cgroup_v2(content).as_deref(),
            Some("/system.slice/docker-abc.scope")
        );
    }

    #[test]
    fn parses_cgroup_v2_ignores_v1_lines() {
        let content = "12:cpu,cpuacct:/foo\n0::/system.slice/bar.scope\n";
        assert_eq!(
            parse_cgroup_v2(content).as_deref(),
            Some("/system.slice/bar.scope")
        );
    }

    #[test]
    fn parses_cgroup_v2_returns_none_on_v1_only() {
        let content = "12:cpu,cpuacct:/foo\n11:memory:/foo\n";
        assert_eq!(parse_cgroup_v2(content), None);
    }
}
