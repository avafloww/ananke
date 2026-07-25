//! The blocking event loop: drains SSE updates into `TuiState`, draws each
//! frame, and translates crossterm key/mouse events into state mutations or
//! dispatcher actions.

use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, MouseEventKind};
use tokio::sync::mpsc;

use crate::{
    client::ApiClientError,
    commands::tui::{
        chat::{SSEUpdate, WireMessage},
        render::render,
        state::{TuiState, resolve_attachments},
    },
};

pub(crate) fn run_tui(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    sse_rx: &mut mpsc::Receiver<SSEUpdate>,
    req_tx: &mpsc::Sender<Vec<WireMessage>>,
    cancel_tx: &mpsc::Sender<()>,
) -> Result<(), ApiClientError> {
    loop {
        // Drain any pending SSE updates before drawing.
        while let Ok(update) = sse_rx.try_recv() {
            match update {
                SSEUpdate::Content(token) => state.append_token(token),
                SSEUpdate::Reasoning(token) => state.append_reasoning(token),
                SSEUpdate::Usage {
                    prompt_tokens,
                    completion_tokens,
                } => state.apply_usage(prompt_tokens, completion_tokens),
                SSEUpdate::Done => state.finish_streaming(),
                SSEUpdate::Cancelled => state.apply_cancel(),
                SSEUpdate::Error(e) => state.set_error(e),
            }
        }

        terminal.draw(|f| render(f, state))?;

        if !event::poll(std::time::Duration::from_millis(50))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if let Some(action) = handle_key(key, state) {
                    match action {
                        KeyAction::Quit => return Ok(()),
                        KeyAction::Submit => {
                            let user_input = std::mem::take(&mut state.input);
                            let trimmed = user_input.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            match resolve_attachments(trimmed) {
                                Ok(attachments) => {
                                    let history = state.submit(trimmed.to_string(), attachments);
                                    if req_tx.blocking_send(history).is_err() {
                                        state.set_error("chat dispatcher closed".to_string());
                                    }
                                }
                                Err(e) => {
                                    // Restore the input so the user can fix the
                                    // bad reference rather than retype the turn.
                                    state.input = user_input;
                                    state.error = Some(e);
                                }
                            }
                        }
                        KeyAction::Cancel => {
                            // Best-effort: dispatcher emits SSEUpdate::Cancelled
                            // which finalises the turn. If the channel is full
                            // there's already a pending cancel and that's fine.
                            let _ = cancel_tx.try_send(());
                        }
                    }
                }
            }
            Event::Mouse(me) => match me.kind {
                MouseEventKind::ScrollUp => state.scroll_up(3),
                MouseEventKind::ScrollDown => state.scroll_down(3),
                _ => {}
            },
            _ => {}
        }
    }
}

enum KeyAction {
    Quit,
    Submit,
    Cancel,
}

/// Handle a single key press, mutating `state` directly for in-line edits
/// and returning a `KeyAction` for things the caller has to coordinate
/// (channel sends, returning from the loop).
///
/// Note on Shift+Enter: many terminals don't deliver a `KeyCode::Enter`
/// with the SHIFT modifier — instead the user must bind Shift+Enter to
/// transmit a literal `\n`. That arrives at crossterm as `Ctrl+J`
/// (since 0x0A == ASCII Ctrl+J), so we treat both as "insert newline".
fn handle_key(key: crossterm::event::KeyEvent, state: &mut TuiState) -> Option<KeyAction> {
    use crossterm::event::KeyModifiers;
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    match key.code {
        KeyCode::Char('c') if ctrl => Some(KeyAction::Quit),
        KeyCode::Char('d') if ctrl && state.input.is_empty() => Some(KeyAction::Quit),
        // Cancel the in-flight turn while keeping any partial response.
        KeyCode::Char('x') if ctrl && state.streaming => Some(KeyAction::Cancel),
        // Toggle inline expansion of `@file` attachments in the history.
        KeyCode::Char('e') if ctrl => {
            state.expand_attachments = !state.expand_attachments;
            None
        }
        // Shift+Enter via terminal modifier reporting (rare).
        KeyCode::Enter if shift => {
            state.input.push('\n');
            None
        }
        // Shift+Enter via terminal config sending a literal LF (0x0A),
        // which crossterm decodes as Ctrl+J.
        KeyCode::Char('j') if ctrl => {
            state.input.push('\n');
            None
        }
        KeyCode::Enter => {
            if state.streaming {
                None
            } else {
                Some(KeyAction::Submit)
            }
        }
        KeyCode::Backspace => {
            state.input.pop();
            None
        }
        KeyCode::Up => {
            state.scroll_up(1);
            None
        }
        KeyCode::Down => {
            state.scroll_down(1);
            None
        }
        KeyCode::End if ctrl => {
            state.scroll_to_bottom();
            None
        }
        KeyCode::Esc => {
            state.input.clear();
            None
        }
        KeyCode::Char(c) => {
            state.input.push(c);
            None
        }
        _ => None,
    }
}
