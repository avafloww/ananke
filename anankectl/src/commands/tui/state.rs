//! Chat transcript state: message/attachment/turn-stats types and the
//! `TuiState` machine that the event loop mutates in response to input and
//! SSE updates.

use std::time::{Duration, Instant};

use ratatui::style::Color;

use crate::commands::tui::chat::WireMessage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MsgRole {
    System,
    Assistant,
    User,
}

impl MsgRole {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Assistant => "Assistant",
            Self::User => "User",
        }
    }

    pub(crate) fn color(&self) -> Color {
        match self {
            Self::System => Color::Magenta,
            Self::Assistant => Color::Cyan,
            Self::User => Color::Yellow,
        }
    }

    fn wire_role(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Assistant => "assistant",
            Self::User => "user",
        }
    }
}

pub(crate) struct TuiMsg {
    pub(crate) role: MsgRole,
    /// The visible final message body — what the model "says".
    pub(crate) content: String,
    /// Subdued reasoning/thinking content (if the model emits it via
    /// `delta.reasoning_content` or `delta.reasoning`). Rendered above
    /// `content` in a dim style.
    pub(crate) reasoning: String,
    pub(crate) streaming: bool,
    /// User cancelled this turn before the model finished. Surfaced in
    /// the assistant box title so the partial response is clearly marked
    /// as truncated rather than completed.
    pub(crate) cancelled: bool,
    /// Live decode/usage timing for assistant messages. Frozen once the
    /// turn finishes. `None` for user/system messages.
    pub(crate) stats: Option<TurnStats>,
    /// Files pulled in via `@path` references on this message. The full
    /// contents are appended to the wire message sent to the model (see
    /// [`TuiMsg::wire_content`]); the history view only shows a collapsed
    /// summary so the box isn't flooded with file bodies.
    pub(crate) attachments: Vec<Attachment>,
}

impl TuiMsg {
    /// The message body as sent to the model. For user messages carrying
    /// `@`-file attachments, each file's contents are appended as
    /// `\n{path}: {contents}` so the model sees the full text even though
    /// the rendered history only shows a collapsed summary.
    fn wire_content(&self) -> String {
        if self.attachments.is_empty() {
            return self.content.clone();
        }
        let mut out = self.content.clone();
        for att in &self.attachments {
            out.push('\n');
            out.push_str(&att.path);
            out.push_str(": ");
            out.push_str(&att.contents);
        }
        out
    }
}

/// A file referenced by `@path` in the user's input and read from disk at
/// submit time.
pub(crate) struct Attachment {
    /// The path as the user referenced it, with the leading `@` stripped.
    pub(crate) path: String,
    /// The full file contents. Appended to the wire message and shown
    /// inline when the user expands attachments.
    pub(crate) contents: String,
}

/// Live and final timing data for a single assistant turn.
pub(crate) struct TurnStats {
    /// When the request was dispatched.
    start: Instant,
    /// When the first decoded delta (content or reasoning) arrived.
    first_token_at: Option<Instant>,
    /// When the stream finished (either `[DONE]` or transport close).
    end: Option<Instant>,
    /// Decode chunks observed (content + reasoning), used as the live
    /// fallback when the server doesn't echo `usage`.
    content_chunks: u32,
    reasoning_chunks: u32,
    /// Token counts from the streamed `usage` chunk (if the server emits
    /// one — most llama.cpp-style servers do when `stream_options.include_usage`
    /// is set).
    pub(crate) prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

impl TurnStats {
    fn new(start: Instant) -> Self {
        Self {
            start,
            first_token_at: None,
            end: None,
            content_chunks: 0,
            reasoning_chunks: 0,
            prompt_tokens: None,
            completion_tokens: None,
        }
    }

    pub(crate) fn ttft(&self) -> Option<Duration> {
        self.first_token_at.map(|t| t.duration_since(self.start))
    }

    /// Tokens-per-second for decode, computed live from chunk counts.
    /// Returns `None` until at least 200 ms of decode have elapsed so the
    /// number doesn't whiplash on the first chunk.
    pub(crate) fn live_decode_rate(&self, now: Instant) -> Option<f64> {
        let first = self.first_token_at?;
        let elapsed = now.duration_since(first).as_secs_f64();
        if elapsed < 0.2 {
            return None;
        }
        let total = (self.content_chunks + self.reasoning_chunks) as f64;
        if total == 0.0 {
            return None;
        }
        Some(total / elapsed)
    }

    /// Final tokens-per-second for decode. Prefers `completion_tokens` from
    /// the usage chunk, falling back to chunk counts.
    pub(crate) fn final_decode_rate(&self) -> Option<f64> {
        let first = self.first_token_at?;
        let end = self.end?;
        let elapsed = end.duration_since(first).as_secs_f64();
        if elapsed <= 0.0 {
            return None;
        }
        let n = self
            .completion_tokens
            .map(|t| t as f64)
            .unwrap_or((self.content_chunks + self.reasoning_chunks) as f64);
        Some(n / elapsed)
    }

    /// Prompt-processing rate: `prompt_tokens / ttft`. Only available once
    /// usage has arrived.
    pub(crate) fn pp_rate(&self) -> Option<f64> {
        let pt = self.prompt_tokens? as f64;
        let ttft = self.ttft()?.as_secs_f64();
        if ttft <= 0.0 {
            return None;
        }
        Some(pt / ttft)
    }

    /// Live or final decoded-token count to display next to the rate.
    pub(crate) fn decoded_tokens(&self) -> u32 {
        self.completion_tokens
            .unwrap_or(self.content_chunks + self.reasoning_chunks)
    }
}

pub(crate) struct TuiState {
    pub(crate) messages: Vec<TuiMsg>,
    pub(crate) input: String,
    /// Lines to scroll back from the bottom. 0 means stuck to the latest output.
    pub(crate) scroll_offset: u16,
    pub(crate) streaming: bool,
    first_token: bool,
    pub(crate) error: Option<String>,
    pub(crate) model: String,
    /// When set, `@file` attachments render their full contents inline
    /// instead of a one-line summary. Toggled globally with Ctrl+E.
    pub(crate) expand_attachments: bool,
}

impl TuiState {
    pub(crate) fn new(system_prompt: &str, model: String) -> Self {
        let messages = if system_prompt.is_empty() {
            Vec::new()
        } else {
            vec![TuiMsg {
                role: MsgRole::System,
                content: system_prompt.to_string(),
                reasoning: String::new(),
                streaming: false,
                cancelled: false,
                stats: None,
                attachments: Vec::new(),
            }]
        };
        Self {
            messages,
            input: String::new(),
            scroll_offset: 0,
            streaming: false,
            first_token: false,
            error: None,
            model,
            expand_attachments: false,
        }
    }

    /// Push the user's message and an empty streaming assistant message,
    /// and return the wire-format history to send (excluding the empty
    /// assistant we just appended).
    pub(crate) fn submit(
        &mut self,
        content: String,
        attachments: Vec<Attachment>,
    ) -> Vec<WireMessage> {
        self.error = None;
        self.messages.push(TuiMsg {
            role: MsgRole::User,
            content,
            reasoning: String::new(),
            streaming: false,
            cancelled: false,
            stats: None,
            attachments,
        });
        let history = self
            .messages
            .iter()
            .map(|m| WireMessage {
                role: m.role.wire_role(),
                content: m.wire_content(),
            })
            .collect();
        self.messages.push(TuiMsg {
            role: MsgRole::Assistant,
            content: String::new(),
            reasoning: String::new(),
            streaming: true,
            cancelled: false,
            stats: Some(TurnStats::new(Instant::now())),
            attachments: Vec::new(),
        });
        self.streaming = true;
        self.first_token = false;
        self.scroll_offset = 0;
        history
    }

    pub(crate) fn append_token(&mut self, token: String) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        if !last.streaming {
            return;
        }
        let pushed = if !self.first_token {
            // Trim leading whitespace from the first content token to handle
            // models that emit "\n" in the initial delta event.
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.content.push_str(trimmed);
            self.first_token = true;
            true
        } else {
            last.content.push_str(&token);
            true
        };
        if pushed && let Some(stats) = last.stats.as_mut() {
            stats.first_token_at.get_or_insert_with(Instant::now);
            stats.content_chunks = stats.content_chunks.saturating_add(1);
        }
        self.scroll_offset = 0;
    }

    pub(crate) fn append_reasoning(&mut self, token: String) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        if !last.streaming {
            return;
        }
        // Trim leading whitespace if reasoning is starting fresh.
        let pushed = if last.reasoning.is_empty() {
            let trimmed = token.trim_start();
            if trimmed.is_empty() {
                return;
            }
            last.reasoning.push_str(trimmed);
            true
        } else {
            last.reasoning.push_str(&token);
            true
        };
        if pushed && let Some(stats) = last.stats.as_mut() {
            stats.first_token_at.get_or_insert_with(Instant::now);
            stats.reasoning_chunks = stats.reasoning_chunks.saturating_add(1);
        }
        self.scroll_offset = 0;
    }

    pub(crate) fn apply_usage(&mut self, prompt: Option<u32>, completion: Option<u32>) {
        let Some(last) = self.messages.last_mut() else {
            return;
        };
        let Some(stats) = last.stats.as_mut() else {
            return;
        };
        if let Some(p) = prompt {
            stats.prompt_tokens = Some(p);
        }
        if let Some(c) = completion {
            stats.completion_tokens = Some(c);
        }
    }

    pub(crate) fn finish_streaming(&mut self) {
        if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
            if let Some(stats) = last.stats.as_mut() {
                stats.end.get_or_insert_with(Instant::now);
            }
        }
        self.streaming = false;
    }

    /// Mark the active streaming turn as user-cancelled. Keeps whatever
    /// content/reasoning has accumulated so the user can still read it,
    /// flips `cancelled` so the title makes the truncation visible.
    pub(crate) fn apply_cancel(&mut self) {
        if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
            last.cancelled = true;
            if let Some(stats) = last.stats.as_mut() {
                stats.end.get_or_insert_with(Instant::now);
            }
        }
        self.streaming = false;
    }

    pub(crate) fn set_error(&mut self, error: String) {
        self.error = Some(error);
        // Drop the placeholder assistant message if it carries no body and
        // no reasoning — there is nothing worth keeping in history.
        if let Some(last) = self.messages.last()
            && last.streaming
            && last.content.is_empty()
            && last.reasoning.is_empty()
        {
            self.messages.pop();
        } else if let Some(last) = self.messages.last_mut()
            && last.streaming
        {
            last.streaming = false;
            if let Some(stats) = last.stats.as_mut() {
                stats.end.get_or_insert_with(Instant::now);
            }
        }
        self.streaming = false;
    }

    pub(crate) fn scroll_up(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount);
    }

    pub(crate) fn scroll_down(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    pub(crate) fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }
}

/// Extract `@path` file references from the user's input. A reference is a
/// whitespace-delimited token starting with `@`; the `@` is stripped and
/// the remainder taken verbatim as a path. Splitting on whitespace keeps
/// in-word `@` (e.g. an email address like `me@host`) from being treated
/// as a reference. Duplicate paths are collapsed, preserving first-seen
/// order.
fn parse_attachment_paths(input: &str) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for token in input.split_whitespace() {
        if let Some(path) = token.strip_prefix('@')
            && !path.is_empty()
            && !paths.iter().any(|p| p == path)
        {
            paths.push(path.to_string());
        }
    }
    paths
}

/// Read every `@path` reference in `input` from disk, in reference order.
/// Returns an error naming the first path that could not be read so the
/// turn isn't sent with a missing file silently dropped.
pub(crate) fn resolve_attachments(input: &str) -> Result<Vec<Attachment>, String> {
    let mut attachments = Vec::new();
    for path in parse_attachment_paths(input) {
        match std::fs::read_to_string(&path) {
            Ok(contents) => attachments.push(Attachment { path, contents }),
            Err(e) => return Err(format!("cannot read @{path}: {e}")),
        }
    }
    Ok(attachments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(content: &str, attachments: Vec<Attachment>) -> TuiMsg {
        TuiMsg {
            role: MsgRole::User,
            content: content.to_string(),
            reasoning: String::new(),
            streaming: false,
            cancelled: false,
            stats: None,
            attachments,
        }
    }

    #[test]
    fn parses_at_references_and_skips_in_word_at() {
        let paths = parse_attachment_paths("look at @src/main.rs and @Cargo.toml please");
        assert_eq!(paths, vec!["src/main.rs", "Cargo.toml"]);
        // An email-style in-word `@` is not a reference.
        assert!(parse_attachment_paths("ping me@example.com about it").is_empty());
        // A lone `@` with no path is ignored.
        assert!(parse_attachment_paths("just an @ sign").is_empty());
    }

    #[test]
    fn deduplicates_paths_preserving_order() {
        let paths = parse_attachment_paths("@a.txt @b.txt @a.txt");
        assert_eq!(paths, vec!["a.txt", "b.txt"]);
    }

    #[test]
    fn wire_content_appends_attachment_bodies() {
        let msg = user_msg(
            "explain @foo.rs",
            vec![Attachment {
                path: "foo.rs".to_string(),
                contents: "fn main() {}".to_string(),
            }],
        );
        assert_eq!(msg.wire_content(), "explain @foo.rs\nfoo.rs: fn main() {}");
    }

    #[test]
    fn wire_content_without_attachments_is_unchanged() {
        let msg = user_msg("just text", Vec::new());
        assert_eq!(msg.wire_content(), "just text");
    }
}
