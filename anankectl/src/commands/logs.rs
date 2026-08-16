//! `anankectl logs` command — paginated historical fetch with optional live tail.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use ananke_api::{
    internal::log_line::LogLine,
    services::{logs::LogsResponse, logs_stream::LogStreamMessage},
};
use futures::StreamExt;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use crate::{
    client::{ApiClient, ApiClientError},
    output,
};

// The signature mirrors the clap CLI surface, so the argument count is owned
// by the flag definitions rather than this function; grouping them into a
// struct would fight the derive.
#[expect(clippy::too_many_arguments)]
pub async fn run(
    client: &ApiClient,
    json: bool,
    name: &str,
    follow: bool,
    run: Option<i64>,
    since: Option<i64>,
    until: Option<i64>,
    limit: u32,
    stream: Option<String>,
) -> Result<(), ApiClientError> {
    let mut query: Vec<String> = Vec::new();
    if let Some(v) = run {
        query.push(format!("run={v}"));
    }
    if let Some(v) = since {
        query.push(format!("since={v}"));
    }
    if let Some(v) = until {
        query.push(format!("until={v}"));
    }
    query.push(format!("limit={limit}"));
    if let Some(v) = stream.as_deref() {
        query.push(format!("stream={v}"));
    }
    let path = format!("/api/services/{name}/logs?{}", query.join("&"));

    // The response is newest-first; print oldest-first by iterating in reverse.
    let response: LogsResponse = client.get_json(&path).await?;
    // Track the highest (run_id, seq) pair seen so we can dedup during the
    // live tail. Using the pair means a new run whose seq resets to 1 will
    // not be incorrectly suppressed by a higher seq from an earlier run.
    let mut max_seen: Option<(i64, i64)> = response.logs.first().map(|l| (l.run_id, l.seq));
    if json {
        output::print_json(&response);
    } else {
        for line in response.logs.iter().rev() {
            print_line(line);
        }
    }

    if !follow {
        return Ok(());
    }

    // Upgrade to a WebSocket for the live tail.
    let stream_path = format!("/api/services/{name}/logs/stream");
    let ws_url = client
        .endpoint
        .join(&stream_path)
        .map_err(|cause| ApiClientError::InvalidPath {
            endpoint: client.endpoint.to_string(),
            path: stream_path,
            cause,
        })?
        .to_string()
        .replace("http://", "ws://")
        .replace("https://", "wss://");

    let (mut ws, _) = connect_async(ws_url)
        .await
        .map_err(|e| ApiClientError::WebSocket(e.to_string()))?;

    while let Some(Ok(Message::Text(s))) = ws.next().await {
        let Ok(msg) = serde_json::from_str::<LogStreamMessage>(&s) else {
            continue;
        };
        match msg {
            LogStreamMessage::Line(line) => {
                // Skip lines we already printed from the historical fetch.
                if let Some(prev) = max_seen
                    && (line.run_id, line.seq) <= prev
                {
                    continue;
                }
                max_seen = Some((line.run_id, line.seq));
                print_line(&line);
            }
            LogStreamMessage::Overflow { dropped } => {
                eprintln!("[{dropped} frames dropped]");
            }
        }
    }

    Ok(())
}

fn print_line(line: &LogLine) {
    println!("{}", format_line(line));
}

/// Render one captured line for the terminal. Split out from the `println!`
/// so the stream label can be asserted without capturing stdout.
fn format_line(line: &LogLine) -> String {
    format!("[{}] {}", line.stream, line.line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anankectl_accepts_and_prints_combined_stream() {
        // The label is printed verbatim, so a container's merged output is
        // never displayed as though it came from the child's stdout.
        let line = |stream: &str| LogLine {
            timestamp_ms: 0,
            stream: stream.to_string(),
            line: "loading model".to_string(),
            run_id: 1,
            seq: 1,
        };
        assert_eq!(format_line(&line("combined")), "[combined] loading model");
        assert_eq!(format_line(&line("stdout")), "[stdout] loading model");
        assert_eq!(format_line(&line("stderr")), "[stderr] loading model");
    }
}
