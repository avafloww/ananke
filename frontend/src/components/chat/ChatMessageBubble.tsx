// A single chat message bubble (user/assistant/system), including the
// per-message token/throughput stats and the reasoning-trace disclosure.

import { useTranslation } from "react-i18next";

import { formatTokenRate } from "../../util.ts";
import { type Message, type StreamStats } from "../../api/chatStore.ts";
import { MarkdownContent } from "./ChatMarkdown.tsx";

export function MessageBubble({
  message,
  modelName,
  liveStats,
}: {
  message: Message;
  modelName: string | null;
  liveStats: StreamStats | null;
}) {
  const { t } = useTranslation();
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const isAssistant = message.role === "assistant";

  const label = isAssistant ? (modelName ?? t("chat.assistant")) : message.role;
  const displayStats = liveStats ?? message.stats ?? null;

  return (
    <div className={`mb-4 ${isSystem ? "opacity-60" : ""}`}>
      <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-0">
        <span className="eyebrow min-w-0 shrink truncate">{label}</span>
        <span className="shrink-0 font-mono text-xs text-tertiary">
          {message.timestamp}
        </span>
        {/* Forces the stats onto their own row on mobile, where the
            label/timestamp/stats combination is too wide to fit on
            one line; hidden on larger screens so it never forces a
            break there. */}
        <span className="basis-full sm:hidden" />
        {isAssistant && displayStats && displayStats.promptTokens !== null && (
          <span className="flex items-center gap-3 text-xs text-tertiary">
            <span>
              {t("chat.promptTokens", { value: displayStats.promptTokens })}
            </span>
            {displayStats.completionTokens !== null && (
              <span>
                {t("chat.outputTokens", {
                  value: displayStats.completionTokens,
                })}
              </span>
            )}
            {displayStats.inputTokPerSec !== null &&
            displayStats.outputTokPerSec !== null ? (
              <>
                <span>
                  {t("chat.inputTokensPerSecond", {
                    value: formatTokenRate(displayStats.inputTokPerSec),
                  })}
                </span>
                <span>
                  {t("chat.outputTokensPerSecond", {
                    value: formatTokenRate(displayStats.outputTokPerSec),
                  })}
                </span>
              </>
            ) : (
              displayStats.predictedPerSecond !== null && (
                <span>
                  {t("chat.tokensPerSecond", {
                    value: displayStats.predictedPerSecond.toFixed(1),
                  })}
                </span>
              )
            )}
          </span>
        )}
      </div>
      <div
        className={`rounded-md px-4 py-3 text-sm ring-1 ring-inset ${
          isUser
            ? "bg-accent/10 ring-accent/20"
            : "bg-elevated ring-border-default/60"
        } ${isSystem ? "text-secondary" : "text-primary"}`}
      >
        {isAssistant && message.reasoning && (
          <details
            open
            className="open:mb-2 open:border-b open:border-border-default open:pb-2 [&_summary]:list-none"
          >
            <summary className="cursor-pointer select-none text-xs text-secondary hover:text-primary">
              {t("chat.reasoning")}
            </summary>
            <div className="mt-1 max-h-40 overflow-y-auto break-words text-xs text-secondary">
              <MarkdownContent>{message.reasoning}</MarkdownContent>
            </div>
          </details>
        )}
        {isAssistant ? (
          message.content ? (
            <div className="flex flex-col gap-3 overflow-hidden break-words">
              <MarkdownContent>{message.content}</MarkdownContent>
            </div>
          ) : null
        ) : (
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        )}
        {message.images && message.images.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.images.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`attachment ${i + 1}`}
                className="max-h-40 object-cover"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
