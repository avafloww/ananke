//! The calibration harness's entry point. Everything it does lives in the library
//! so the phases stay testable from inside the crate.

fn main() -> std::process::ExitCode {
    ananke_measure::harness::cli::main()
}
