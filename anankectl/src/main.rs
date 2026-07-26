use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod client;
mod commands;
mod config;
mod output;

#[derive(Parser)]
#[command(name = "anankectl", version)]
struct Cli {
    /// Base URL for the management API. Falls back to `$ANANKE_ENDPOINT`,
    /// then the `endpoint` key in `anankectl config`, then the built-in
    /// default.
    #[arg(long, global = true)]
    endpoint: Option<String>,

    /// Emit responses as raw JSON instead of formatted text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum ServerConfigCommand {
    /// Show the current configuration.
    Show,
    /// Validate a configuration file or stdin.
    Validate {
        /// Path to the configuration file (reads stdin if not provided).
        file: Option<std::path::PathBuf>,
    },
    /// Reload the configuration.
    Reload,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Print the value of a config key.
    Get {
        /// Config key (e.g. `endpoint`).
        key: String,
    },
    /// Set a config key to a value.
    Set {
        /// Config key (e.g. `endpoint`).
        key: String,
        /// New value for the key.
        value: String,
    },
    /// Remove a config key.
    Unset {
        /// Config key (e.g. `endpoint`).
        key: String,
    },
    /// List all known config keys and their current values.
    List,
    /// Print the path to the client config file.
    Path,
    /// Open the client config file in `$EDITOR` (or `vi`).
    Edit,
}

#[derive(Subcommand)]
enum OneshotCommand {
    /// Submit a oneshot job from a TOML file.
    Submit {
        /// Path to the TOML file describing the job.
        file: std::path::PathBuf,
    },
    /// Submit a oneshot job built from inline flags and a trailing command.
    Run {
        /// Optional human-readable name.
        #[arg(long)]
        name: Option<String>,
        /// Eviction priority (higher wins).
        #[arg(long, default_value_t = 50)]
        priority: u8,
        /// Time-to-live duration string (e.g. "2h", "30m").
        #[arg(long)]
        ttl: Option<String>,
        /// Working directory for the spawned child.
        #[arg(long)]
        workdir: Option<std::path::PathBuf>,
        /// Device-placement mode.
        #[arg(long, default_value = "gpu-only")]
        placement: String,
        /// Static reservation in GiB; conflicts with --min-reserve-gb/--max-reserve-gb.
        /// The reservation lands on host RAM for a cpu-only service and on VRAM
        /// otherwise; `--vram-gb` remains accepted as the pre-rename spelling.
        #[arg(long, alias = "vram-gb", conflicts_with_all = ["min_reserve_gb", "max_reserve_gb"])]
        reserve_gb: Option<f32>,
        /// Dynamic lower bound for the reservation in GiB; requires --max-reserve-gb.
        #[arg(long, alias = "min-vram-gb", requires = "max_reserve_gb")]
        min_reserve_gb: Option<f32>,
        /// Dynamic upper bound for the reservation in GiB.
        #[arg(long, alias = "max-vram-gb")]
        max_reserve_gb: Option<f32>,
        /// Command and arguments to run.
        #[arg(trailing_var_arg = true, required = true)]
        command: Vec<String>,
    },
    /// List all known oneshot jobs.
    List,
    /// Cancel a oneshot job by ID.
    Kill {
        /// Oneshot job ID to cancel.
        id: String,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Show daemon, device, and service snapshot in one view.
    Status,
    /// List devices with reservations.
    Devices,
    /// List services.
    Services {
        /// Include disabled services.
        #[arg(long)]
        all: bool,
    },
    /// Show service detail.
    Show {
        /// Service name.
        name: String,
    },
    /// Start a service.
    Start {
        /// Service name.
        name: String,
    },
    /// Stop a service.
    Stop {
        /// Service name.
        name: String,
    },
    /// Restart a service.
    Restart {
        /// Service name.
        name: String,
    },
    /// Enable a service.
    Enable {
        /// Service name.
        name: String,
    },
    /// Disable a service.
    Disable {
        /// Service name.
        name: String,
    },
    /// Retry a service (enable then start).
    Retry {
        /// Service name.
        name: String,
    },
    /// Tail logs for a service.
    Logs {
        /// Service name.
        name: String,
        /// Follow new lines as they arrive.
        #[arg(long)]
        follow: bool,
        /// Filter to a specific run id.
        #[arg(long)]
        run: Option<i64>,
        /// Minimum timestamp: ms since epoch, RFC 3339, a local datetime
        /// (`2026-07-24 15:30`), a local date, or an age like `2h` / `30m`.
        #[arg(long, value_parser = parse_time_arg)]
        since: Option<i64>,
        /// Maximum timestamp; accepts the same forms as `--since`.
        #[arg(long, value_parser = parse_time_arg)]
        until: Option<i64>,
        /// Cap on number of historical lines returned.
        #[arg(long, default_value_t = 200)]
        limit: u32,
        /// Filter to stdout or stderr.
        #[arg(long)]
        stream: Option<String>,
    },
    /// Manage oneshot jobs.
    Oneshot {
        #[command(subcommand)]
        command: OneshotCommand,
    },
    /// Manage daemon configuration over the management API.
    ServerConfig {
        #[command(subcommand)]
        command: ServerConfigCommand,
    },
    /// Manage anankectl's own client-side configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Reload daemon configuration (alias for `server-config reload`).
    Reload,
    /// Talk to a model via the OpenAI-compatible API.
    Chat {
        /// Model (service) name. Omit to pick interactively from enabled services.
        model: Option<String>,
        /// User prompt.
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
        /// System prompt.
        #[arg(long, default_value = "You are a helpful assistant.")]
        system_prompt: String,
    },
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let endpoint = match config::resolve_endpoint(cli.endpoint) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("anankectl: {e}");
            return e.exit_code();
        }
    };
    let client = client::ApiClient::new(&endpoint);
    let result = match cli.command {
        Command::Status => commands::status::run(&client, cli.json).await,
        Command::Devices => commands::devices::run(&client, cli.json).await,
        Command::Services { all } => commands::services::run(&client, cli.json, all).await,
        Command::Show { name } => commands::show::run(&client, cli.json, &name).await,
        Command::Start { name } => commands::lifecycle::start(&client, cli.json, &name).await,
        Command::Stop { name } => commands::lifecycle::stop(&client, cli.json, &name).await,
        Command::Restart { name } => commands::lifecycle::restart(&client, cli.json, &name).await,
        Command::Enable { name } => commands::lifecycle::enable(&client, cli.json, &name).await,
        Command::Disable { name } => commands::lifecycle::disable(&client, cli.json, &name).await,
        Command::Retry { name } => commands::lifecycle::retry(&client, cli.json, &name).await,
        Command::Logs {
            name,
            follow,
            run,
            since,
            until,
            limit,
            stream,
        } => {
            commands::logs::run(
                &client, cli.json, &name, follow, run, since, until, limit, stream,
            )
            .await
        }
        Command::Oneshot { command } => match command {
            OneshotCommand::Submit { file } => {
                commands::oneshot::submit(&client, cli.json, &file).await
            }
            OneshotCommand::Run {
                name,
                priority,
                ttl,
                workdir,
                placement,
                reserve_gb,
                min_reserve_gb,
                max_reserve_gb,
                command,
            } => {
                commands::oneshot::run(
                    &client,
                    cli.json,
                    name,
                    priority,
                    ttl,
                    workdir,
                    placement,
                    reserve_gb,
                    min_reserve_gb,
                    max_reserve_gb,
                    command,
                )
                .await
            }
            OneshotCommand::List => commands::oneshot::list(&client, cli.json).await,
            OneshotCommand::Kill { id } => commands::oneshot::kill(&client, cli.json, &id).await,
        },
        Command::ServerConfig { command } => match command {
            ServerConfigCommand::Show => commands::server_config::show(&client, cli.json).await,
            ServerConfigCommand::Validate { file } => {
                commands::server_config::validate(&client, cli.json, file.as_deref()).await
            }
            ServerConfigCommand::Reload => commands::server_config::reload(&client, cli.json).await,
        },
        Command::Config { command } => match command {
            ConfigCommand::Get { key } => commands::config::get(cli.json, &key).await,
            ConfigCommand::Set { key, value } => {
                commands::config::set(cli.json, &key, &value).await
            }
            ConfigCommand::Unset { key } => commands::config::unset(cli.json, &key).await,
            ConfigCommand::List => commands::config::list(cli.json).await,
            ConfigCommand::Path => commands::config::path(cli.json).await,
            ConfigCommand::Edit => commands::config::edit().await,
        },
        Command::Reload => commands::server_config::reload(&client, cli.json).await,
        Command::Chat {
            model,
            prompt,
            system_prompt,
        } => {
            let prompt = prompt.join(" ");
            commands::chat::run(&client, cli.json, model.as_deref(), &prompt, &system_prompt).await
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("anankectl: {e}");
            e.exit_code()
        }
    }
}

/// Parse a `--since`/`--until` value into ms since epoch. Accepts, in order:
/// a raw integer (ms since epoch, the historical form), a relative age like
/// `2h` or `30m` (meaning that long ago), an RFC 3339 timestamp, or a civil
/// datetime / date interpreted in the system timezone.
fn parse_time_arg(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if let Ok(ms) = s.parse::<i64>() {
        return Ok(ms);
    }
    if let Some(ms_ago) = parse_relative_age_ms(s) {
        return Ok(jiff::Timestamp::now().as_millisecond() - ms_ago);
    }
    if let Ok(ts) = s.parse::<jiff::Timestamp>() {
        return Ok(ts.as_millisecond());
    }
    let tz = jiff::tz::TimeZone::system();
    if let Ok(dt) = s.parse::<jiff::civil::DateTime>()
        && let Ok(zoned) = tz.to_zoned(dt)
    {
        return Ok(zoned.timestamp().as_millisecond());
    }
    if let Ok(date) = s.parse::<jiff::civil::Date>()
        && let Ok(zoned) = tz.to_zoned(date.at(0, 0, 0, 0))
    {
        return Ok(zoned.timestamp().as_millisecond());
    }
    Err(format!(
        "cannot parse `{s}` as a timestamp; use ms since epoch, RFC 3339, \
         `YYYY-MM-DD[ HH:MM[:SS]]` (local time), or an age like `2h`"
    ))
}

/// Parse `90s` / `15m` / `2h` / `1d` into a millisecond count. `None` when
/// the string is not of that shape.
fn parse_relative_age_ms(s: &str) -> Option<i64> {
    let (num, unit) = s.split_at(s.len().checked_sub(1)?);
    let per_unit: i64 = match unit {
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return None,
    };
    let n: i64 = num.parse().ok().filter(|n| *n >= 0)?;
    n.checked_mul(per_unit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ms_passes_through() {
        assert_eq!(parse_time_arg("1784904497789"), Ok(1784904497789));
        assert_eq!(parse_time_arg("0"), Ok(0));
    }

    #[test]
    fn relative_ages_subtract_from_now() {
        let now = jiff::Timestamp::now().as_millisecond();
        let two_hours = parse_time_arg("2h").unwrap();
        let delta = now - two_hours;
        // Within a second of exactly two hours ago.
        assert!((delta - 7_200_000).abs() < 1_000, "delta = {delta}");
        assert!(parse_time_arg("30m").is_ok());
        assert!(parse_time_arg("90s").is_ok());
        assert!(parse_time_arg("1d").is_ok());
    }

    #[test]
    fn rfc3339_parses() {
        let ms = parse_time_arg("2026-07-24T15:38:47Z").unwrap();
        // 2026-07-24T15:38:47Z as a fixed, timezone-independent instant.
        assert_eq!(ms, 1784907527000);
    }

    #[test]
    fn civil_datetime_uses_system_timezone() {
        // The exact value depends on the host timezone; assert only that the
        // forms parse and order sensibly.
        let day = parse_time_arg("2026-07-24").unwrap();
        let noon = parse_time_arg("2026-07-24 12:00").unwrap();
        let noon_secs = parse_time_arg("2026-07-24 12:00:30").unwrap();
        assert!(day < noon);
        assert_eq!(noon_secs - noon, 30_000);
    }

    #[test]
    fn garbage_is_rejected_with_guidance() {
        let err = parse_time_arg("yesterday-ish").unwrap_err();
        assert!(err.contains("cannot parse"), "{err}");
        assert!(parse_time_arg("2x").is_err());
        assert!(parse_time_arg("").is_err());
    }
}
