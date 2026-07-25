//! Interactive TUI chat interface using ratatui.

mod chat;
mod input;
mod render;
mod state;

use std::io;

use ananke_api::services::list::ServicesResponse;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use tokio::sync::mpsc;

use crate::{
    client::{ApiClient, ApiClientError},
    commands::tui::{
        chat::{SSEUpdate, WireMessage, chat_dispatcher},
        input::run_tui,
        state::TuiState,
    },
};

pub async fn run(
    client: &ApiClient,
    model: &str,
    system_prompt: &str,
) -> Result<(), ApiClientError> {
    // Discover the OpenAI port from the management API.
    let resp: ServicesResponse = client.get_json("/api/services").await?;
    let port = resp.openai_api_port;

    let openai_url = construct_openai_url(&client.endpoint, port)?;

    // Channels: TUI -> dispatcher carries new conversation submissions
    // and cancellation signals; dispatcher -> TUI carries SSE updates.
    let (req_tx, req_rx) = mpsc::channel::<Vec<WireMessage>>(8);
    let (cancel_tx, cancel_rx) = mpsc::channel::<()>(4);
    let (sse_tx, sse_rx) = mpsc::channel::<SSEUpdate>(64);

    let chat_handle = tokio::spawn(chat_dispatcher(
        openai_url,
        model.to_string(),
        req_rx,
        cancel_rx,
        sse_tx,
    ));

    // Set up terminal.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
        let _ = disable_raw_mode();
        return Err(e.into());
    }

    let state = TuiState::new(system_prompt, model.to_string());

    let result = tokio::task::spawn_blocking(move || {
        let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
        let mut terminal = ratatui::Terminal::new(backend)?;
        let mut state = state;
        let mut sse_rx = sse_rx;
        let req_tx = req_tx;
        let cancel_tx = cancel_tx;
        let inner = run_tui(&mut terminal, &mut state, &mut sse_rx, &req_tx, &cancel_tx);
        // Restore terminal regardless of inner result.
        let _ = execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = disable_raw_mode();
        let _ = terminal.show_cursor();
        inner
    })
    .await
    .map_err(|e| ApiClientError::Usage(format!("TUI panicked: {e}")))?;

    // Closing req_tx (dropped above) signals the dispatcher to exit; await it.
    let _ = chat_handle.await;

    result
}

fn construct_openai_url(mgmt: &reqwest::Url, port: u16) -> Result<reqwest::Url, ApiClientError> {
    let host = mgmt
        .host_str()
        .ok_or_else(|| ApiClientError::Usage("management endpoint has no host".into()))?;
    let mut openai = mgmt.clone();
    openai.set_scheme(mgmt.scheme()).ok();
    openai.set_host(Some(host)).ok();
    let _ = openai.set_port(Some(port));
    Ok(openai)
}
