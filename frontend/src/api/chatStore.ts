// Chat session store. Holds messages, system prompt, streaming state,
// and input/attachments at module level so they survive navigation
// (component unmount/remount) within the same tab session. The store
// is the single source of truth — ChatView reads from it via
// useSyncExternalStore and mutates it through the exported actions.

import { useSyncExternalStore } from "react";

export type StreamStats = {
  ttftMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  /// Decode throughput: completion tokens divided by the engine-reported
  /// decode window (`timings.predicted_ms`). Null when the engine does
  /// not emit timings, in which case `predictedPerSecond` is the fallback.
  outputTokPerSec: number | null;
  /// Prefill throughput: cache-aware prompt tokens (`timings.prompt_n`)
  /// divided by prefill time (`timings.prompt_ms`). Null when absent.
  inputTokPerSec: number | null;
  /// Effective end-to-end rate: completion tokens over total wall-clock
  /// elapsed. Always computable; shown only as a fallback when the engine
  /// does not provide the input/output split.
  predictedPerSecond: number | null;
};

export type Attachment = {
  name: string;
  size: number;
  type: "text" | "image";
  content: string;
};

export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
  reasoning?: string;
  images?: string[];
  stats?: StreamStats;
  timestamp: string;
};

type ChatState =
  | { kind: "idle" }
  | { kind: "starting"; controller: AbortController }
  | { kind: "streaming"; controller: AbortController }
  | { kind: "error"; message: string };

type ChatSnapshot = {
  messages: Message[];
  systemPrompt: string;
  currentModel: string | null;
  chatState: ChatState;
  stats: StreamStats;
  input: string;
  attachments: Attachment[];
};

export const EMPTY_STATS: StreamStats = {
  ttftMs: null,
  promptTokens: null,
  completionTokens: null,
  outputTokPerSec: null,
  inputTokPerSec: null,
  predictedPerSecond: null,
};

let snapshot: ChatSnapshot = {
  messages: [],
  systemPrompt: "",
  currentModel: null,
  chatState: { kind: "idle" },
  stats: EMPTY_STATS,
  input: "",
  attachments: [],
};

const listeners = new Set<() => void>();

export function setSnapshot(
  updater: (prev: ChatSnapshot) => ChatSnapshot,
): void {
  snapshot = updater(snapshot);
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function getSnapshot(): ChatSnapshot {
  return snapshot;
}

// --- Actions ---

export function setInput(value: string): void {
  setSnapshot((prev) => ({ ...prev, input: value }));
}

export function saveSystemPrompt(value: string): void {
  setSnapshot((prev) => ({ ...prev, systemPrompt: value }));
  const model = snapshot.currentModel;
  if (model) {
    try {
      localStorage.setItem(`ananke-chat-sys-${model}`, value);
    } catch {
      // localStorage unavailable.
    }
  }
}

export function selectModel(name: string | null): void {
  let prompt = "";
  if (name) {
    try {
      prompt = localStorage.getItem(`ananke-chat-sys-${name}`) ?? "";
    } catch {
      // localStorage unavailable.
    }
  }
  setSnapshot(() => ({
    messages: [],
    systemPrompt: prompt,
    currentModel: name,
    chatState: { kind: "idle" },
    stats: { ...EMPTY_STATS },
    input: "",
    attachments: [],
  }));
}

export function addAttachment(att: Attachment): void {
  setSnapshot((prev) => ({
    ...prev,
    attachments: [...prev.attachments, att],
  }));
}

export function removeAttachment(index: number): void {
  setSnapshot((prev) => ({
    ...prev,
    attachments: prev.attachments.filter((_, i) => i !== index),
  }));
}

export function clearConversation(): void {
  // Abort any in-flight send so it does not re-add the user message
  // after the conversation is cleared. Without this, a queued-start
  // send() that is still polling for the model to come online would
  // re-read the snapshot after loading completes and push the user
  // message back into the cleared conversation.
  if (
    snapshot.chatState.kind === "streaming" ||
    snapshot.chatState.kind === "starting"
  ) {
    snapshot.chatState.controller.abort();
  }
  setSnapshot((prev) => ({
    ...prev,
    messages: [],
    stats: { ...EMPTY_STATS },
    chatState: { kind: "idle" },
  }));
}

export function cancel(): void {
  if (
    snapshot.chatState.kind === "streaming" ||
    snapshot.chatState.kind === "starting"
  ) {
    snapshot.chatState.controller.abort();
    setSnapshot((prev) => ({ ...prev, chatState: { kind: "idle" } }));
  }
}

// --- Send ---

export { send } from "./chatSend.ts";

// --- Hook ---

export function useChat(): ChatSnapshot {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
