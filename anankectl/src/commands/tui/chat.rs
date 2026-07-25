//! Wire-format types and the background dispatcher that streams chat
//! completions from the daemon's OpenAI-compatible endpoint over SSE.

use futures::StreamExt;
use tokio::sync::mpsc;

pub(crate) enum SSEUpdate {
    Content(String),
    Reasoning(String),
    Usage {
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    },
    Done,
    Cancelled,
    Error(String),
}

#[derive(serde::Serialize, Clone)]
pub(crate) struct WireMessage {
    pub(crate) role: &'static str,
    pub(crate) content: String,
}

#[derive(serde::Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    stream: bool,
    stream_options: StreamOptions,
}

#[derive(serde::Serialize)]
struct StreamOptions {
    include_usage: bool,
}

pub(crate) async fn chat_dispatcher(
    base: reqwest::Url,
    model: String,
    mut req_rx: mpsc::Receiver<Vec<WireMessage>>,
    mut cancel_rx: mpsc::Receiver<()>,
    sse_tx: mpsc::Sender<SSEUpdate>,
) {
    let client = reqwest::Client::new();
    while let Some(messages) = req_rx.recv().await {
        // Drain stale cancel signals so a Ctrl+X received between turns
        // doesn't kill the next one.
        while cancel_rx.try_recv().is_ok() {}
        tokio::select! {
            res = stream_one(&client, &base, &model, messages, &sse_tx) => {
                if let Err(e) = res {
                    let _ = sse_tx.send(SSEUpdate::Error(e)).await;
                }
            }
            _ = cancel_rx.recv() => {
                // Dropping the future drops the in-flight HTTP stream
                // (reqwest closes the connection on drop).
                let _ = sse_tx.send(SSEUpdate::Cancelled).await;
            }
        }
    }
}

async fn stream_one(
    client: &reqwest::Client,
    base: &reqwest::Url,
    model: &str,
    messages: Vec<WireMessage>,
    sse_tx: &mpsc::Sender<SSEUpdate>,
) -> Result<(), String> {
    let request = ChatRequest {
        model: model.to_string(),
        messages,
        stream: true,
        stream_options: StreamOptions {
            include_usage: true,
        },
    };

    let body = serde_json::to_vec(&request).map_err(|e| format!("serialise chat request: {e}"))?;
    let url = base
        .join("v1/chat/completions")
        .map_err(|e| format!("invalid openai path: {e}"))?;

    let resp = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {status}: {body_text}"));
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    loop {
        tokio::select! {
            chunk_result = stream.next() => {
                let Some(chunk_result) = chunk_result else {
                    let _ = sse_tx.send(SSEUpdate::Done).await;
                    return Ok(());
                };
                let chunk = chunk_result.map_err(|e| e.to_string())?;
                buf.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(newline_pos) = buf.find('\n') {
                    let line = buf[..newline_pos].to_string();
                    buf.replace_range(..=newline_pos, "");

                    if let Some(data) = line.trim().strip_prefix("data: ") {
                        if data == "[DONE]" {
                            let _ = sse_tx.send(SSEUpdate::Done).await;
                            return Ok(());
                        }
                        let DeltaParts { content, reasoning } = extract_delta(data);
                        if let Some(r) = reasoning
                            && !r.is_empty()
                            && sse_tx.send(SSEUpdate::Reasoning(r)).await.is_err()
                        {
                            return Ok(());
                        }
                        if let Some(c) = content
                            && !c.is_empty()
                            && sse_tx.send(SSEUpdate::Content(c)).await.is_err()
                        {
                            return Ok(());
                        }
                        if let Some(usage) = extract_usage(data)
                            && sse_tx
                                .send(SSEUpdate::Usage {
                                    prompt_tokens: usage.prompt_tokens,
                                    completion_tokens: usage.completion_tokens,
                                })
                                .await
                                .is_err()
                        {
                            return Ok(());
                        }
                    }
                }
            }
            _ = sse_tx.closed() => {
                return Ok(());
            }
        }
    }
}

struct DeltaParts {
    content: Option<String>,
    reasoning: Option<String>,
}

/// Pull both `delta.content` and reasoning text from one SSE payload.
/// Reasoning may arrive under `reasoning_content` (DeepSeek style) or
/// plain `reasoning`; we accept either.
fn extract_delta(data: &str) -> DeltaParts {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(data) else {
        return DeltaParts {
            content: None,
            reasoning: None,
        };
    };
    let Some(delta) = val
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
    else {
        return DeltaParts {
            content: None,
            reasoning: None,
        };
    };
    let content = delta
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let reasoning = delta
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()))
        .map(str::to_string);
    DeltaParts { content, reasoning }
}

struct UsageInfo {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

/// Pull `usage.{prompt,completion}_tokens` from an SSE payload. OpenAI-style
/// servers emit these in a final delta when `stream_options.include_usage`
/// is set; chunks without a `usage` object return `None`.
fn extract_usage(data: &str) -> Option<UsageInfo> {
    let val = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let usage = val.get("usage")?;
    let prompt = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let completion = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    if prompt.is_none() && completion.is_none() {
        return None;
    }
    Some(UsageInfo {
        prompt_tokens: prompt,
        completion_tokens: completion,
    })
}
