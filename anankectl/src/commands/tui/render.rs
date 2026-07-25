//! ratatui rendering: lays out the header/messages/input/status panes and
//! turns `TuiState` into styled `Line`s and widgets each frame.

use std::time::Instant;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

use crate::commands::tui::state::{TuiMsg, TuiState};

const INPUT_MIN_CONTENT_ROWS: u16 = 3;
const INPUT_MAX_CONTENT_ROWS: u16 = 10;

pub(crate) fn render(f: &mut Frame, state: &TuiState) {
    // Layout: header (1) | messages (min) | input (grows) | status (1).
    // No borders on header/status, so Length(1) fits a single line. The
    // input grows with its content (between INPUT_MIN_CONTENT_ROWS and
    // INPUT_MAX_CONTENT_ROWS rows of content, plus two border rows); past
    // the cap the input area scrolls internally to keep the latest line
    // visible.
    let area = f.area();
    let inner_width = area.width.saturating_sub(2);
    let content_rows = input_content_rows(&state.input, inner_width);
    let input_height = content_rows.saturating_add(2);

    let chunks = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .split(area);

    render_header(f, chunks[0], state);
    render_messages(f, chunks[1], state);
    render_input(f, chunks[2], state);
    render_status(f, chunks[3], state);
}

fn render_header(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let (sym, sym_color, label, label_color) = if let Some(err) = &state.error {
        ("✗", Color::Red, format!("Error: {err}"), Color::Red)
    } else if state.streaming {
        ("●", Color::Cyan, "Streaming…".to_string(), Color::White)
    } else {
        ("●", Color::Green, "Ready".to_string(), Color::White)
    };
    let model_label = format!(" · {}", state.model);
    let line = Line::from(vec![
        Span::styled(format!("{sym} "), Style::default().fg(sym_color)),
        Span::styled(
            label,
            Style::default()
                .fg(label_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(model_label, Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn render_messages(f: &mut Frame, area: Rect, state: &TuiState) {
    if area.width < 4 || area.height == 0 {
        return;
    }
    // Each message gets its own bordered block; the inner content width is
    // 2 less than the area's width.
    let content_width = area.width.saturating_sub(2);

    // Build per-message lines + total bordered height up front so we can
    // determine scroll. Messages butt up directly so adjacent borders share
    // a row.
    let rendered: Vec<(Vec<Line>, u16)> = state
        .messages
        .iter()
        .map(|m| {
            let lines = build_message_lines(m, state.expand_attachments);
            let rows = visual_line_count(&lines, content_width);
            // u32 -> u16 with headroom for the +2 borders, never overflowing.
            let content = rows.min((u16::MAX - 2) as u32) as u16;
            (lines, content.saturating_add(2))
        })
        .collect();

    let total_height: u32 = rendered.iter().map(|(_, h)| *h as u32).sum();
    let visible_height = area.height as u32;

    // Anchor to the bottom; user scroll offset moves the window upward.
    let max_window_top = total_height.saturating_sub(visible_height);
    let window_top = max_window_top.saturating_sub(state.scroll_offset as u32);

    // Walk the virtual stack, emitting any visible portion of each message.
    let now = Instant::now();
    let mut virtual_y: u32 = 0;
    for (i, (lines, h)) in rendered.iter().enumerate() {
        let msg_top = virtual_y;
        let msg_bottom = virtual_y + *h as u32;
        virtual_y = msg_bottom;

        let win_top = window_top;
        let win_bottom = window_top + visible_height;
        if msg_bottom <= win_top || msg_top >= win_bottom {
            continue;
        }

        let clipped_top = msg_top.max(win_top);
        let clipped_bottom = msg_bottom.min(win_bottom);
        let render_y = area.y + (clipped_top - win_top) as u16;
        let render_h = (clipped_bottom - clipped_top) as u16;
        let internal_scroll = (clipped_top - msg_top) as u16;

        let rect = Rect {
            x: area.x,
            y: render_y,
            width: area.width,
            height: render_h,
        };

        let msg = &state.messages[i];
        let title_line = build_title_line(msg, now);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(msg.role.color()))
            .title(title_line);

        let paragraph = Paragraph::new(lines.clone())
            .wrap(Wrap { trim: false })
            .scroll((internal_scroll, 0))
            .block(block);
        f.render_widget(paragraph, rect);
    }
}

/// Build a title line for a message box: role label followed by any
/// per-turn stats (live during streaming, frozen after `Done`).
fn build_title_line(msg: &TuiMsg, now: Instant) -> Line<'static> {
    let dim = Style::default().fg(Color::DarkGray);
    let mut spans = vec![Span::styled(
        format!(" {} ", msg.role.label()),
        Style::default()
            .fg(msg.role.color())
            .add_modifier(Modifier::BOLD),
    )];
    if msg.cancelled {
        spans.push(Span::styled(
            "· cancelled ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    }
    if let Some(stats) = &msg.stats {
        let streaming = msg.streaming;
        let tokens = stats.decoded_tokens();
        if tokens > 0 {
            spans.push(Span::styled(format!("· {tokens} tok "), dim));
        }
        let rate = if streaming {
            stats.live_decode_rate(now)
        } else {
            stats.final_decode_rate()
        };
        if let Some(r) = rate {
            spans.push(Span::styled(format!("· {r:.1} tok/s "), dim));
        }
        if let Some(ttft) = stats.ttft() {
            spans.push(Span::styled(
                format!("· ttft {:.2}s ", ttft.as_secs_f64()),
                dim,
            ));
        }
        if let Some(pp) = stats.pp_rate() {
            let pt = stats.prompt_tokens.unwrap_or(0);
            spans.push(Span::styled(format!("· {pt} prompt @ {pp:.0} tok/s "), dim));
        }
    }
    Line::from(spans)
}

/// Build the visual lines for a single message: subdued reasoning first
/// (if any), then the content, with a streaming spinner on the trailing
/// line where appropriate.
fn build_message_lines(msg: &TuiMsg, expand_attachments: bool) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let reasoning_label = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC | Modifier::BOLD);

    let mut lines: Vec<Line<'static>> = Vec::new();

    if !msg.reasoning.is_empty() {
        lines.push(Line::from(Span::styled("thinking", reasoning_label)));
        for raw in msg.reasoning.split('\n') {
            lines.push(Line::from(Span::styled(raw.to_string(), dim)));
        }
        // If the model is still mid-reasoning (no content yet), mark the
        // last reasoning line with the spinner.
        if msg.streaming
            && msg.content.is_empty()
            && let Some(last) = lines.last_mut()
        {
            last.spans.push(Span::styled(" ⟳", dim));
        }
        // Only insert a separator once content actually exists.
        if !msg.content.is_empty() {
            lines.push(Line::from(""));
        }
    }

    if msg.content.is_empty() && msg.streaming && msg.reasoning.is_empty() {
        lines.push(Line::from(Span::styled("⟳ thinking…", dim)));
    } else if !msg.content.is_empty() {
        for raw in msg.content.split('\n') {
            lines.push(Line::from(raw.to_string()));
        }
        if msg.streaming
            && let Some(last) = lines.last_mut()
        {
            last.spans.push(Span::styled(" ⟳", dim));
        }
    } else if !msg.streaming
        && msg.content.is_empty()
        && msg.reasoning.is_empty()
        && msg.attachments.is_empty()
    {
        // System message with empty content (rare); render an empty line so
        // the box isn't zero-height.
        lines.push(Line::from(""));
    }

    if !msg.attachments.is_empty() {
        let attach_label = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        // Separate the attachment summary from any typed message body.
        if !msg.content.is_empty() {
            lines.push(Line::from(""));
        }
        for att in &msg.attachments {
            let line_count = att.contents.lines().count();
            let byte_count = att.contents.len();
            lines.push(Line::from(vec![
                Span::styled(format!("📎 {}", att.path), attach_label),
                Span::styled(format!(" ({line_count} lines, {byte_count} bytes)"), dim),
            ]));
            if expand_attachments {
                for raw in att.contents.split('\n') {
                    lines.push(Line::from(Span::styled(format!("  {raw}"), dim)));
                }
            }
        }
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

/// Estimate how many rendered rows a list of `Line`s occupies when wrapped
/// to `width`. Accounts for empty lines (1 row) and rounds up partial rows.
fn visual_line_count(lines: &[Line<'_>], width: u16) -> u32 {
    if width == 0 {
        return lines.len() as u32;
    }
    let w = width as u32;
    let mut total = 0u32;
    for line in lines {
        let len: u32 = line
            .spans
            .iter()
            .map(|s| s.content.chars().count() as u32)
            .sum();
        let rows = if len == 0 { 1 } else { len.div_ceil(w) };
        total = total.saturating_add(rows);
    }
    total
}

fn render_input(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let title = if state.streaming {
        "Input (waiting for response…)"
    } else {
        "Input"
    };
    let style = if state.streaming {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Gray))
        .title(title);
    let inner_width = block.inner(area).width;
    let inner_height = block.inner(area).height as u32;
    let total = wrapped_rows(&state.input, inner_width);
    // Anchor the cursor row (always at the end of input) to the bottom of
    // the box once content overflows.
    let scroll = total.saturating_sub(inner_height) as u16;
    f.render_widget(
        Paragraph::new(state.input.as_str())
            .style(style)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(block),
        area,
    );
}

/// Total wrapped row count for the input string at `width`. Empty lines
/// still occupy one row; trailing newline reserves a final empty row so
/// the cursor sits visibly on it.
fn wrapped_rows(s: &str, width: u16) -> u32 {
    if width == 0 {
        return 1;
    }
    let w = width as u32;
    // `split('\n')` yields a trailing "" after a terminal '\n', which is
    // exactly the row we want the cursor to live on.
    s.split('\n')
        .map(|seg| {
            let len = seg.chars().count() as u32;
            if len == 0 { 1 } else { len.div_ceil(w) }
        })
        .sum()
}

/// Visible content-row count for the input box, clamped between min and
/// max row constants. Adds one trailing row when the buffer is empty so
/// the box never starts at 0 content rows.
fn input_content_rows(input: &str, inner_width: u16) -> u16 {
    let total = wrapped_rows(input, inner_width).min(u16::MAX as u32) as u16;
    total.clamp(INPUT_MIN_CONTENT_ROWS, INPUT_MAX_CONTENT_ROWS)
}

fn render_status(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let mut spans = vec![
        Span::styled("Enter", Style::default().fg(Color::White)),
        Span::raw(" send · "),
        Span::styled("Shift+Enter", Style::default().fg(Color::White)),
        Span::raw(" newline · "),
        Span::styled("@path", Style::default().fg(Color::White)),
        Span::raw(" attach · "),
    ];
    // Only advertise the expand toggle once there's something to expand.
    if state.messages.iter().any(|m| !m.attachments.is_empty()) {
        let label = if state.expand_attachments {
            " collapse files · "
        } else {
            " expand files · "
        };
        spans.push(Span::styled("Ctrl+E", Style::default().fg(Color::White)));
        spans.push(Span::raw(label));
    }
    spans.extend([
        Span::styled("↑/↓/wheel", Style::default().fg(Color::White)),
        Span::raw(" scroll · "),
        Span::styled("Ctrl+X", Style::default().fg(Color::White)),
        Span::raw(" cancel · "),
        Span::styled("Esc", Style::default().fg(Color::White)),
        Span::raw(" clear · "),
        Span::styled("Ctrl+C", Style::default().fg(Color::White)),
        Span::raw(" quit"),
    ]);
    let line = Line::from(spans).style(Style::default().fg(Color::DarkGray));
    f.render_widget(Paragraph::new(line), area);
}
