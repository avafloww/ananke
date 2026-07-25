// Oneshot list row and detail panel for OneshotsView.

import { useTranslation } from "react-i18next";

import type { OneshotStatus } from "../../api/client.ts";
import { formatDuration, relativeTime } from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { Badge } from "../ui/Badge.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { LogsViewer } from "../logs/LogsViewer.tsx";

export function OneshotRow({
  oneshot,
  selected,
  onSelect,
  onDelete,
  deletePending,
}: {
  oneshot: OneshotStatus;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  deletePending: boolean;
}) {
  const { t } = useTranslation();
  const isTerminal = oneshot.state === "ended" || oneshot.state === "evicted";

  return (
    <div
      className={`flex cursor-pointer items-center gap-3 px-4 py-2 transition-colors hover:bg-elevated/60 ${
        selected ? "bg-elevated/40" : ""
      }`}
      onClick={onSelect}
    >
      <StatusDot state={oneshot.state === "running" ? "running" : "idle"} />
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
        <span className="truncate font-mono text-sm text-primary">
          {oneshot.name}
        </span>
        {oneshot.state === "running" && (
          <Badge variant="success">{t("oneshots.running")}</Badge>
        )}
        {oneshot.state === "ended" && (
          <Badge variant={oneshot.exit_code === 0 ? "neutral" : "danger"}>
            {oneshot.exit_code != null
              ? t("oneshots.exitCode", { code: oneshot.exit_code })
              : t("oneshots.ended")}
          </Badge>
        )}
      </div>
      <span className="ml-auto shrink-0 font-mono text-xs text-tertiary">
        :{oneshot.port}
      </span>
      <span className="shrink-0 font-mono text-xs text-tertiary">
        {relativeTime(oneshot.submitted_at_ms)}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        disabled={deletePending || isTerminal}
        title={t("oneshots.kill")}
        className="inline-flex h-7 w-7 items-center justify-center rounded-md text-danger transition-colors hover:bg-danger/15 disabled:opacity-30"
      >
        <KillIcon />
      </button>
    </div>
  );
}

export function OneshotDetail({ oneshot }: { oneshot: OneshotStatus }) {
  const { t } = useTranslation();
  const submitted = new Date(oneshot.submitted_at_ms).toLocaleString();
  const started = oneshot.started_at_ms
    ? new Date(oneshot.started_at_ms).toLocaleString()
    : null;
  const ended = oneshot.ended_at_ms
    ? new Date(oneshot.ended_at_ms).toLocaleString()
    : null;
  const duration =
    oneshot.started_at_ms && oneshot.ended_at_ms
      ? formatDuration(oneshot.ended_at_ms - oneshot.started_at_ms)
      : null;

  return (
    <Card header={oneshot.name} bodyClassName="p-0">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 border-b border-border-default px-4 py-3 text-sm sm:grid-cols-4">
        <DetailField label={t("oneshots.id")} value={oneshot.id} mono />
        <DetailField
          label={t("oneshots.port")}
          value={`:${oneshot.port}`}
          mono
        />
        <DetailField label={t("oneshots.state")} value={oneshot.state} />
        <DetailField
          label={t("oneshots.exit")}
          value={oneshot.exit_code != null ? String(oneshot.exit_code) : "—"}
          mono
        />
        <DetailField label={t("oneshots.submitted")} value={submitted} />
        {started && (
          <DetailField label={t("oneshots.started")} value={started} />
        )}
        {ended && <DetailField label={t("oneshots.endedAt")} value={ended} />}
        {duration && (
          <DetailField label={t("oneshots.duration")} value={duration} mono />
        )}
      </div>
      <LogsViewer name={oneshot.id} />
    </Card>
  );
}

function DetailField({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-xs text-tertiary">{label}</dt>
      <dd className={`text-primary ${mono ? "font-mono text-xs" : "text-sm"}`}>
        {value}
      </dd>
    </div>
  );
}

function KillIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
