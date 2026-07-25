// Chat interface — a web equivalent of `anankectl chat`. The operator
// picks a model, enters a system prompt, and chats with streaming
// responses, token stats, and file attachments.
//
// Chat state (messages, system prompt, input, attachments, streaming
// state) lives in the module-level chatStore so it survives navigation
// away and back within the same tab session.

import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router-dom";

import { useServices, useInfo } from "../../api/hooks.ts";
import { openaiBaseUrlFromListen } from "../../util.ts";
import {
  addAttachment,
  cancel as cancelSend,
  clearConversation,
  removeAttachment,
  saveSystemPrompt,
  selectModel,
  send,
  setInput,
  useChat,
} from "../../api/chatStore.ts";
import { Spinner } from "../ui/Spinner.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { ExternalLinkIcon, TrashIcon } from "../ui/icons.tsx";
import { useStickToBottom } from "../../hooks/useStickToBottom.ts";
import { ModelDropdown } from "./ChatModelDropdown.tsx";
import { MessageBubble } from "./ChatMessageBubble.tsx";

export function ChatView() {
  const { t } = useTranslation();
  const services = useServices();
  const info = useInfo();
  const [searchParams, setSearchParams] = useSearchParams();
  const chat = useChat();

  const chatModels = (services.data ?? [])
    .filter(
      (s) =>
        s.modality !== "embedding" && !s.name.toLowerCase().includes("comfyui"),
    )
    .sort((a, b) => {
      const ar = a.state === "running" ? 0 : 1;
      const br = b.state === "running" ? 0 : 1;
      if (ar !== br) return ar - br;
      return a.name.localeCompare(b.name);
    });
  const paramModel = searchParams.get("model");

  // Sync URL → store, but only when the URL explicitly specifies a
  // model AND there is no active conversation. The store is the source
  // of truth for session persistence; navigating to /chat without
  // ?model= should not wipe the session, and navigating from a service
  // detail view with ?model=foo should not clobber an existing
  // conversation with a different model.
  useEffect(() => {
    if (
      paramModel &&
      paramModel !== chat.currentModel &&
      chat.messages.length === 0
    ) {
      selectModel(paramModel);
    }
  }, [paramModel, chat.currentModel, chat.messages.length]);

  // The store's currentModel is the effective selection — it survives
  // navigation away and back even when the URL lacks ?model=.
  const selectedModel = chat.currentModel;

  const { scrollRef, onScroll } = useStickToBottom(chat.messages);

  async function handleSend() {
    if (!selectedModel) return;
    const baseUrl = openaiBaseUrlFromListen(
      info.data?.openai_listen ?? "0.0.0.0:7070",
    );
    await send(selectedModel, baseUrl);
  }

  function handleSelectModel(name: string) {
    setSearchParams({ model: name });
  }

  async function handleFiles(files: FileList) {
    const svc = selectedModel
      ? (services.data ?? []).find((s) => s.name === selectedModel)
      : null;
    const hasVision = svc?.has_mmproj ?? false;

    for (const file of Array.from(files)) {
      if (file.type.startsWith("image/")) {
        if (!hasVision) continue;
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            addAttachment({
              name: file.name,
              size: file.size,
              type: "image",
              content: result,
            });
          }
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            addAttachment({
              name: file.name,
              size: file.size,
              type: "text",
              content: result,
            });
          }
        };
        reader.readAsText(file);
      }
    }
  }

  if (services.isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  const inputDisabled = !selectedModel || chat.chatState.kind === "starting";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <ViewHeader>
        <h1 className="eyebrow !text-primary">{t("chat.title")}</h1>
      </ViewHeader>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-auto px-4 py-4"
      >
        {chat.messages.length === 0 ? (
          <>
            <EmptyState message={t("chat.emptyState")} />
            {chat.chatState.kind === "starting" && (
              <div className="flex items-center justify-center gap-2 py-2 text-sm text-tertiary">
                <Spinner />
                {t("chat.starting", { model: selectedModel })}
              </div>
            )}
          </>
        ) : (
          chat.messages.map((msg, i) => (
            <MessageBubble
              key={msg.timestamp}
              message={msg}
              modelName={selectedModel}
              liveStats={
                chat.chatState.kind === "streaming" &&
                i === chat.messages.length - 1 &&
                msg.role === "assistant"
                  ? chat.stats
                  : null
              }
            />
          ))
        )}
        {chat.messages.length > 0 && chat.chatState.kind === "starting" && (
          <div className="flex items-center gap-2 py-2 text-sm text-tertiary">
            <Spinner />
            {t("chat.starting", { model: selectedModel })}
          </div>
        )}
        {chat.chatState.kind === "error" && (
          <div className="mt-2 rounded-sm border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {chat.chatState.message}
          </div>
        )}
      </div>

      {/* Composer block: sits at the bottom of the flex column, below
          the flex-1 message list, so it never scrolls with the
          conversation on mobile or desktop. */}
      <div className="shrink-0 bg-surface">
        {/* System prompt */}
        {selectedModel && (
          <details className="shrink-0 border-t border-border-default px-4 py-2">
            <summary className="eyebrow cursor-pointer select-none hover:text-secondary">
              {t("chat.systemPrompt")}
            </summary>
            <textarea
              value={chat.systemPrompt}
              onChange={(e) => saveSystemPrompt(e.target.value)}
              placeholder={t("chat.systemPromptPlaceholder")}
              className="mt-1 h-20 w-full resize-none rounded-sm border border-border-default bg-base px-2 py-1 text-xs text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
            />
          </details>
        )}

        {/* Attachments preview */}
        {chat.attachments.length > 0 && (
          <div className="shrink-0 flex flex-wrap items-center gap-2 border-t border-border-default px-4 py-2">
            {chat.attachments.map((att, i) => (
              <div
                key={i}
                className="flex items-center gap-1 rounded-sm bg-elevated px-2 py-0.5 text-xs text-secondary"
              >
                {att.type === "image" && (
                  <img
                    src={att.content}
                    alt={att.name}
                    className="h-6 w-6 rounded object-cover"
                  />
                )}
                <span>{att.name}</span>
                <button
                  onClick={() => removeAttachment(i)}
                  className="text-tertiary hover:text-danger"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Composer: model picker + stats sit next to the input, so the
          controls you use most are all within reach of the textbox. */}
        <div className="shrink-0 border-t border-border-default px-4 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <ModelDropdown
              models={chatModels}
              selected={selectedModel}
              onSelect={handleSelectModel}
              className="min-w-0 flex-1"
            />
            {selectedModel && (
              <CopyButton
                value={selectedModel}
                className="h-7 rounded-md bg-elevated px-2 text-xs font-medium text-primary hover:bg-border-strong"
              />
            )}
            {selectedModel && (
              <ButtonLink
                to={`/services/${encodeURIComponent(selectedModel)}`}
                variant="secondary"
                size="sm"
                className="w-7 justify-center px-0"
              >
                <ExternalLinkIcon />
              </ButtonLink>
            )}
            {selectedModel && (
              <Button
                variant="secondary"
                size="sm"
                className="w-7 justify-center px-0"
                onClick={clearConversation}
              >
                <TrashIcon />
              </Button>
            )}
          </div>
          <div className="flex items-stretch gap-2">
            <textarea
              value={chat.input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  // On touch devices, Enter inserts a newline instead of
                  // sending. Coarse-pointer devices are phones/tablets.
                  if (window.matchMedia("(pointer: coarse)").matches) return;
                  e.preventDefault();
                  void handleSend();
                }
              }}
              placeholder={
                selectedModel
                  ? chat.chatState.kind === "starting"
                    ? t("chat.startingModelPlaceholder")
                    : t("chat.typePlaceholder")
                  : t("chat.selectModelFirst")
              }
              disabled={inputDisabled}
              rows={3}
              className="flex-1 resize-none overflow-y-auto rounded-sm border border-border-default bg-base px-2 py-2.5 text-base text-primary placeholder:text-tertiary focus:border-accent focus:outline-none disabled:opacity-50 md:text-sm"
            />
            <label className="flex w-10 shrink-0 self-stretch cursor-pointer items-center justify-center rounded-md bg-elevated text-lg text-secondary hover:bg-border-strong">
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) handleFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              +
            </label>
            {chat.chatState.kind === "streaming" ||
            chat.chatState.kind === "starting" ? (
              <button
                onClick={cancelSend}
                disabled={chat.chatState.kind === "starting"}
                className="shrink-0 self-stretch rounded-md bg-danger px-3 text-sm font-medium text-white hover:bg-danger/90 disabled:opacity-40"
              >
                {t("chat.stop")}
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!selectedModel || !chat.input.trim()}
                className="inline-flex shrink-0 self-stretch items-center justify-center gap-1.5 rounded-md bg-action-iris px-3 text-sm font-medium text-white transition-[filter] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {t("chat.send")}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
