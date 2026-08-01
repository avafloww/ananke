// Service detail page (`/services/:name`). Shows model info, VRAM
// estimate, placement preview, launch command, and a logs viewer.

import { useParams, Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { useServiceDetail, useLifecycle } from "../../api/hooks.ts";
import type { RestartEvent } from "../../api/client.ts";
import { formatTimestamp, relativeTime, serviceProxyUrl } from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Stat } from "../ui/Stat.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { Button } from "../ui/Button.tsx";
import { ButtonLink } from "../ui/ButtonLink.tsx";
import { buttonClassName } from "../ui/buttonStyles.ts";
import {
  ChatIcon,
  DisableIcon,
  ExternalLinkIcon,
  PlayIcon,
  PowerIcon,
  RestartIcon,
  StopIcon,
} from "../ui/icons.tsx";
import { LogsViewer } from "../logs/LogsViewer.tsx";
import {
  ModelInfoGrid,
  ConfigGrid,
  ServingGrid,
  EstimateGrid,
} from "./ServiceInfoCards.tsx";
import { PlacementSection } from "./ServicePlacement.tsx";
import { LaunchCommandSection } from "./ServiceLaunchCommand.tsx";
import { ServiceMetrics } from "./ServiceMetrics.tsx";

export function ServiceDetailView() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const detail = useServiceDetail(name ?? null);
  const lifecycle = useLifecycle();

  if (detail.isPending) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  if (detail.error || !detail.data) {
    return (
      <div className="p-4 text-sm text-danger">
        {detail.error?.message ?? t("serviceDetail.notFound")}
      </div>
    );
  }

  const d = detail.data;
  const pending = lifecycle.isPending && lifecycle.variables?.name === name;

  return (
    <div className="flex h-full flex-col">
      {/* Header — fixed height to align with the sidebar wordmark and
          the other views' headers, forming one continuous rule. */}
      <ViewHeader>
        <Link to="/" className="text-tertiary hover:text-secondary">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
        </Link>
        <StatusDot state={d.state} className="h-2.5 w-2.5" />
        <h1 className="min-w-0 truncate font-mono text-sm font-semibold tracking-[0.02em] text-primary">
          {d.name}
        </h1>
        {d.model_info?.has_mmproj && <Badge variant="vision">vision</Badge>}
        {d.modality === "embedding" && (
          <Badge variant="embedding">embedding</Badge>
        )}
        {d.modality === "transcription" && (
          <Badge variant="transcription">transcription</Badge>
        )}
        <div className="ml-auto flex flex-wrap items-center gap-4">
          <Stat label={t("serviceDetail.port")} value={`:${d.port}`} />
          <Stat label={t("serviceDetail.pid")} value={d.pid ?? "—"} />
          <Stat label={t("serviceDetail.priority")} value={d.priority} />
          <Stat label={t("serviceDetail.lifecycle")} value={d.lifecycle} />
        </div>
      </ViewHeader>

      <div className="flex-1 space-y-4 overflow-auto p-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Lifecycle actions */}
          <Card header={t("serviceDetail.actions")} className="lg:col-span-2">
            <div className="flex flex-wrap items-center gap-2">
              <a
                href={serviceProxyUrl(d.port)}
                target="_blank"
                rel="noopener noreferrer"
                className={buttonClassName("blue")}
              >
                <ExternalLinkIcon />
                {t("serviceDetail.open")}
              </a>
              {d.modality !== "embedding" && d.modality !== "transcription" && (
                <ButtonLink
                  variant="iris"
                  to={`/chat?model=${encodeURIComponent(d.name)}`}
                >
                  <ChatIcon className="w-3.5 h-3.5" />
                  {t("serviceDetail.chat")}
                </ButtonLink>
              )}
              <LifecycleActions
                state={d.state}
                pending={pending}
                onAction={(action) =>
                  lifecycle.mutate({ action, name: d.name })
                }
              />
            </div>
          </Card>

          {/* Model info */}
          {d.model_info && (
            <Card header={t("serviceDetail.model")}>
              <ModelInfoGrid model={d.model_info} />
            </Card>
          )}

          {/* Configuration */}
          <Card header={t("serviceDetail.configuration")}>
            <ConfigGrid detail={d} />
          </Card>

          {/* Serving (llama-cpp services only) */}
          {d.serving && (
            <Card
              header={t("serviceDetail.serving")}
              bodyClassName="max-h-72 overflow-y-auto p-4"
            >
              <ServingGrid serving={d.serving} runtime={d.runtime ?? null} />
            </Card>
          )}

          {/* Memory estimate */}
          {d.estimate && (
            <Card header={t("serviceDetail.memoryEstimate")}>
              <EstimateGrid
                estimate={d.estimate}
                observedPeakBytes={d.observed_peak_bytes}
              />
            </Card>
          )}
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Placement */}
          <Card header={t("serviceDetail.placement")} className="lg:col-span-2">
            <PlacementSection
              current={d.current_allocation}
              placementOverride={d.placement_override}
              placement={d.placement_preview ?? null}
            />
          </Card>

          {/* Launch command */}
          <Card
            header={t("serviceDetail.launchCommand")}
            className="lg:col-span-2"
          >
            <LaunchCommandSection name={d.name} />
          </Card>
        </div>

        {/* Per-service stats */}
        {d.modality !== "embedding" && d.modality !== "transcription" && (
          <ServiceMetrics name={d.name} />
        )}

        {/* Auto-restart history */}
        {(d.recent_restarts?.length ?? 0) > 0 && (
          <Card header={t("serviceDetail.autoRestarts")}>
            <RestartHistory restarts={d.recent_restarts ?? []} />
          </Card>
        )}

        {/* Logs */}
        <Card header={t("serviceDetail.logs")} bodyClassName="p-0">
          <LogsViewer name={d.name} />
        </Card>
      </div>
    </div>
  );
}

function RestartHistory({ restarts }: { restarts: readonly RestartEvent[] }) {
  return (
    <ul className="flex flex-col gap-2 text-sm">
      {restarts.map((r) => (
        <li
          key={`${r.at_ms}-${r.trigger}`}
          className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-3"
        >
          <span
            className="shrink-0 text-tertiary tabular-nums"
            title={formatTimestamp(r.at_ms)}
          >
            {relativeTime(r.at_ms)}
          </span>
          <Badge variant="warning">{r.trigger}</Badge>
          <span className="text-primary">{r.detail}</span>
        </li>
      ))}
    </ul>
  );
}

function LifecycleActions({
  state,
  pending,
  onAction,
}: {
  state: string;
  pending: boolean;
  onAction: (
    action: "start" | "stop" | "restart" | "enable" | "disable",
  ) => void;
}) {
  const { t } = useTranslation();
  const canStart = ["idle", "stopped", "failed", "evicted"].includes(state);
  const canStop = ["running", "starting", "draining"].includes(state);
  const isDisabled = state.startsWith("disabled");

  return (
    <div className="flex flex-wrap items-center gap-2">
      {canStart && (
        <Button
          variant="green"
          onClick={() => onAction("start")}
          disabled={pending}
        >
          <PlayIcon />
          {t("services.actions.start")}
        </Button>
      )}
      {canStop && (
        <>
          <Button
            variant="red"
            onClick={() => onAction("stop")}
            disabled={pending}
          >
            <StopIcon />
            {t("services.actions.stop")}
          </Button>
          <Button
            variant="cyan"
            onClick={() => onAction("restart")}
            disabled={pending}
          >
            <RestartIcon />
            {t("services.actions.restart")}
          </Button>
        </>
      )}
      {isDisabled ? (
        <Button
          variant="orange"
          onClick={() => onAction("enable")}
          disabled={pending}
        >
          <PowerIcon />
          {t("services.actions.enable")}
        </Button>
      ) : (
        <Button
          variant="magenta"
          onClick={() => onAction("disable")}
          disabled={pending}
        >
          <DisableIcon />
          {t("services.actions.disable")}
        </Button>
      )}
    </div>
  );
}
