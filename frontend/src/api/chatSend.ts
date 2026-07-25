// Chat "send" request: kicks off the model if needed, then streams
// the OpenAI-compatible chat completion and folds deltas into the
// chat store snapshot.

import { api } from "./client.ts";
import { getSnapshot as getSystemSnapshot } from "./systemStore.ts";
import {
  getSnapshot,
  setSnapshot,
  EMPTY_STATS,
  type StreamStats,
} from "./chatStore.ts";

/// ISO8601 timestamp in the local timezone (e.g. 2026-06-27T14:30:00+02:00).
function localISO(): string {
  const d = new Date();
  const off = d.getTimezoneOffset();
  const sign = off <= 0 ? "+" : "-";
  const abs = Math.abs(off);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  const local = new Date(d.getTime() - off * 60000).toISOString().slice(0, 19);
  return `${local}${sign}${hh}:${mm}`;
}

type ApiContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type ApiMessage = {
  role: string;
  content: string | ApiContentPart[];
};

export async function send(
  selectedModel: string,
  openaiBaseUrl: string,
): Promise<void> {
  const snap = getSnapshot();
  if (
    !selectedModel ||
    !snap.input.trim() ||
    snap.chatState.kind === "streaming"
  )
    return;

  // Create the controller early so cancel() works during the start loop.
  const controller = new AbortController();

  // Check if the model is running; start it if not.
  const sysSnap = getSystemSnapshot();
  const svc = sysSnap.services.find((s) => s.name === selectedModel);
  const needsStart = svc != null && svc.state !== "running";
  if (needsStart) {
    setSnapshot((prev) => ({
      ...prev,
      chatState: { kind: "starting", controller },
    }));
    try {
      await api.start(selectedModel);
      const deadline = Date.now() + 3 * 60 * 1000;
      while (Date.now() < deadline) {
        if (controller.signal.aborted) return;
        await new Promise((r) => setTimeout(r, 2000));
        if (controller.signal.aborted) return;
        const resp = await api.listServices();
        const s = resp.services.find((x) => x.name === selectedModel);
        if (s?.state === "running") break;
        if (s?.state === "failed") {
          setSnapshot((prev) => ({
            ...prev,
            chatState: {
              kind: "error",
              message: `service ${selectedModel} failed to start`,
            },
          }));
          return;
        }
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      setSnapshot((prev) => ({
        ...prev,
        chatState: {
          kind: "error",
          message: `failed to start ${selectedModel}: ${e instanceof Error ? e.message : String(e)}`,
        },
      }));
      return;
    }
  }

  // Re-read after the (possibly long) start sequence — the user may
  // have typed more or added attachments.
  const snap2 = getSnapshot();
  const currentInput = snap2.input.trim();
  if (!currentInput) return;

  const imageAttachments = snap2.attachments.filter((a) => a.type === "image");
  const textAttachments = snap2.attachments.filter((a) => a.type === "text");

  let fullContent = currentInput;
  for (const att of textAttachments) {
    fullContent += `\n\n${att.name}:\n${att.content}`;
  }

  const displayImages = imageAttachments.map((a) => a.content);

  setSnapshot((prev) => ({
    ...prev,
    messages: [
      ...prev.messages,
      {
        role: "user",
        content: currentInput,
        images: displayImages.length > 0 ? displayImages : undefined,
        timestamp: localISO(),
      },
    ],
    attachments: [],
    input: "",
    stats: { ...EMPTY_STATS },
  }));

  setSnapshot((prev) => ({
    ...prev,
    chatState: { kind: "streaming", controller },
  }));

  const startTime = performance.now();
  let firstTokenTime: number | null = null;
  const finalStats: StreamStats = { ...EMPTY_STATS };

  function attachStats() {
    setSnapshot((prev) => {
      const next = [...prev.messages];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, stats: { ...finalStats } };
      }
      return { ...prev, messages: next };
    });
  }

  // Build API messages from the snapshot at send time.
  const apiMessages: ApiMessage[] = [];
  if (snap2.systemPrompt.trim()) {
    apiMessages.push({ role: "system", content: snap2.systemPrompt.trim() });
  }
  for (const m of snap2.messages) {
    apiMessages.push({ role: m.role, content: m.content });
  }
  if (imageAttachments.length > 0) {
    const parts: ApiContentPart[] = [{ type: "text", text: fullContent }];
    for (const img of imageAttachments) {
      parts.push({ type: "image_url", image_url: { url: img.content } });
    }
    apiMessages.push({ role: "user", content: parts });
  } else {
    apiMessages.push({ role: "user", content: fullContent });
  }

  try {
    const resp = await fetch(`${openaiBaseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: selectedModel,
        messages: apiMessages,
        stream: true,
        stream_options: { include_usage: true },
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      setSnapshot((prev) => ({
        ...prev,
        input: "",
        chatState: {
          kind: "error",
          message: `${resp.status} ${resp.statusText}: ${text}`,
        },
      }));
      return;
    }

    const reader = resp.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") continue;

        try {
          const chunk = JSON.parse(data) as {
            choices?: Array<{
              delta?: {
                content?: string;
                reasoning_content?: string;
              };
            }>;
            usage?: {
              prompt_tokens?: number;
              completion_tokens?: number;
            };
            timings?: {
              prompt_ms?: number;
              predicted_ms?: number;
              prompt_n?: number;
            };
          };

          const delta = chunk.choices?.[0]?.delta;
          const contentDelta = delta?.content;
          const reasoningDelta = delta?.reasoning_content;

          if (contentDelta || reasoningDelta) {
            if (firstTokenTime === null) {
              firstTokenTime = performance.now();
              finalStats.ttftMs = firstTokenTime - startTime;
              setSnapshot((prev) => ({
                ...prev,
                stats: { ...finalStats },
                input: "",
              }));
            }
            setSnapshot((prev) => {
              const next = [...prev.messages];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: contentDelta
                    ? last.content + contentDelta
                    : last.content,
                  reasoning: reasoningDelta
                    ? (last.reasoning ?? "") + reasoningDelta
                    : last.reasoning,
                };
              } else {
                next.push({
                  role: "assistant",
                  content: contentDelta ?? "",
                  reasoning: reasoningDelta || undefined,
                  timestamp: localISO(),
                });
              }
              return { ...prev, messages: next };
            });
          }

          if (chunk.usage) {
            const elapsed = (performance.now() - startTime) / 1000;
            const completionTokens = chunk.usage.completion_tokens ?? 0;
            finalStats.promptTokens = chunk.usage.prompt_tokens ?? null;
            finalStats.completionTokens = completionTokens;
            finalStats.predictedPerSecond =
              elapsed > 0 ? completionTokens / elapsed : null;

            // Engine-reported phase timings (llama.cpp). The `timings`
            // object sits next to `usage` in the final chunk and carries
            // `prompt_ms` (prefill), `predicted_ms` (decode), and
            // `prompt_n` (tokens actually evaluated — excludes cache-served
            // tokens, unlike `usage.prompt_tokens`). When present, these
            // yield the true prefill and decode rates. When absent
            // (non-llama.cpp engines), the split fields stay null and the
            // UI falls back to the effective rate above.
            const timings = chunk.timings;
            if (timings) {
              const promptMs = timings.prompt_ms;
              const predictedMs = timings.predicted_ms;
              const promptN = timings.prompt_n;
              if (
                promptMs != null &&
                promptMs > 0 &&
                promptN != null &&
                promptN > 0
              ) {
                finalStats.inputTokPerSec = promptN / (promptMs / 1000);
              }
              if (
                predictedMs != null &&
                predictedMs > 0 &&
                completionTokens > 0
              ) {
                finalStats.outputTokPerSec =
                  completionTokens / (predictedMs / 1000);
              }
            }

            setSnapshot((prev) => ({ ...prev, stats: { ...finalStats } }));
          }
        } catch {
          // Skip unparseable lines.
        }
      }
    }
    setSnapshot((prev) => ({ ...prev, input: "" }));
    attachStats();
    setSnapshot((prev) => ({ ...prev, chatState: { kind: "idle" } }));
  } catch (e) {
    setSnapshot((prev) => ({ ...prev, input: "" }));
    attachStats();
    if (e instanceof DOMException && e.name === "AbortError") {
      setSnapshot((prev) => ({ ...prev, chatState: { kind: "idle" } }));
    } else {
      setSnapshot((prev) => ({
        ...prev,
        chatState: {
          kind: "error",
          message: e instanceof Error ? e.message : String(e),
        },
      }));
    }
  }
}
