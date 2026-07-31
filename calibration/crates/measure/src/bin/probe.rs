//! Attribute host-memory growth a single measurement cannot separate.
//!
//! The harness samples one process once per configuration, which cannot tell a term
//! allocated once from one that accumulates with use. This varies one thing at a time
//! against a fresh server and reports the series.
//!
//! ```sh
//! cargo run -p ananke-measure --bin probe -- --model /path/to/model.gguf
//! cargo run -p ananke-measure --bin probe -- --model … --only step,maps
//! ```
//!
//! It loads real models and reads real memory, so run it against an idle machine.
//! Nothing else may use the GPUs while it runs, for the same reason the campaign
//! says so: a second process changes the figures being read.
//!
//! Nothing is written. The output is for a reader.

use std::{path::PathBuf, process::ExitCode};

use ananke_measure::harness::{
    probe::{Options, Question, probe, render},
    sys::Deps,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "probe",
    about = "Attribute a llama-server's host-memory growth"
)]
struct Cli {
    /// The GGUF to load. A first shard is enough for a sharded model.
    #[arg(long)]
    model: String,

    /// Which questions to ask, comma-separated. All of them by default.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,

    /// The llama-server binary.
    #[arg(long, env = "MAINLINE_BIN", default_value = "llama-server")]
    binary: String,

    /// Cards to expose, as `CUDA_VISIBLE_DEVICES`.
    #[arg(long, default_value = "0")]
    gpus: String,

    #[arg(long, default_value_t = 32768)]
    context: u32,

    #[arg(long, default_value_t = 18080)]
    port: u16,

    /// Where each server's load log goes.
    #[arg(long, default_value = "/tmp")]
    log_dir: PathBuf,

    /// Print the plan and the number of servers it loads, then stop.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let questions = match parse_questions(&cli.only) {
        Ok(questions) => questions,
        Err(name) => {
            eprintln!(
                "unknown question `{name}`; expected any of {}",
                Question::ALL
                    .iter()
                    .map(|q| q.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return ExitCode::from(2);
        }
    };

    let stages = ananke_measure::harness::probe::plan::plan(&questions);
    println!(
        "{} server load(s), each a fresh process — the step being hunted happens once",
        stages.len()
    );
    for stage in &stages {
        println!("  {}", stage.label);
    }
    if cli.dry_run {
        return ExitCode::SUCCESS;
    }
    println!();

    let deps = Deps::local();
    let observations = probe(
        &deps,
        &Options {
            model: &cli.model,
            binary: &cli.binary,
            gpus: &cli.gpus,
            context: cli.context,
            port: cli.port,
            log_dir: &cli.log_dir,
            questions: questions.clone(),
        },
    );

    print!("{}", render(&observations, &questions, &cli.model));

    // A run that measured nothing is a failure: the caller asked a question and got
    // no answer, and an exit code is the only way a script notices.
    if observations.readings.is_empty() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn parse_questions(only: &[String]) -> Result<Vec<Question>, String> {
    if only.is_empty() {
        return Ok(Question::ALL.to_vec());
    }
    only.iter()
        .map(|name| Question::parse(name).ok_or_else(|| name.clone()))
        .collect()
}
