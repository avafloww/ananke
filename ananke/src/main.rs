use ananke::{daemon::run, errors::ExpectedError};

// Keeps the error type in scope so a signature change surfaces here first.
fn _ensure_error_type_in_scope() -> Option<ExpectedError> {
    None
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ananke: {err}");
            std::process::ExitCode::from(err.exit_code())
        }
    }
}
