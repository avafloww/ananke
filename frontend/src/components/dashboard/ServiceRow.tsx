// Service list row (status, sparkline, lifecycle actions) for DashboardView.

import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import type { FitVerdict, ServiceSummary } from "../../api/client.ts";
import { formatBytes, serviceProxyUrl } from "../../util.ts";
import { StatusDot } from "../ui/StatusDot.tsx";
import { Sparkline } from "../ui/Sparkline.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";
import {
  DisableIcon,
  PlayIcon,
  PowerIcon,
  RestartIcon,
  StarIcon,
  StopIcon,
} from "../ui/icons.tsx";

export function ServiceRow({
  svc,
  sparkData,
  xMin,
  xMax,
  pending,
  onAction,
  favourite,
  onToggleFavourite,
}: {
  svc: ServiceSummary;
  sparkData?: { ts: number[]; counts: number[] };
  xMin: number;
  xMax: number;
  pending: boolean;
  onAction: (
    action: "start" | "stop" | "restart" | "enable" | "disable",
  ) => void;
  favourite: boolean;
  onToggleFavourite: () => void;
}) {
  const { t } = useTranslation();

  const canStart = ["idle", "stopped", "failed", "evicted"].includes(svc.state);
  const canStop = ["running", "starting", "draining"].includes(svc.state);
  const isDisabled = svc.state.startsWith("disabled");

  // Sparkline data — zeroFillGaps (applied inside Sparkline) handles
  // the no-data and single-point cases.
  const sparkLineData: (number | null)[][] = sparkData
    ? [sparkData.ts, sparkData.counts]
    : [[], []];

  return (
    // `py-1.5` against the old `py-2`: the row grew a second line, so the same
    // padding would make it half again as tall and the list would stop being
    // scannable at a glance.
    <div className="flex items-center gap-3 px-4 py-1.5 transition-colors hover:bg-elevated/60">
      <button
        onClick={onToggleFavourite}
        className={`shrink-0 transition-colors ${
          favourite ? "text-warning" : "text-tertiary hover:text-secondary"
        }`}
        title={favourite ? "Unfavourite" : "Favourite"}
      >
        <StarIcon filled={favourite} />
      </button>
      <Link
        to={`/services/${encodeURIComponent(svc.name)}`}
        className="flex min-w-0 flex-1 items-center gap-3 overflow-hidden"
      >
        <StatusDot state={svc.state} />
        {/* Name over footprint. The tags beside it stay centred against the
            pair rather than against the name alone, which is why the stack is
            its own flex item instead of the row being two lines. */}
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span
            className={`min-w-0 truncate font-mono text-sm ${fitVerdictColor(svc.fit_verdict)}`}
            title={fitVerdictTitle(svc.fit_verdict, t)}
          >
            {svc.name}
          </span>
          <Footprint svc={svc} />
        </div>
        {svc.has_mmproj && (
          <span className="shrink-0 font-mono text-xs text-vision">vision</span>
        )}
        {svc.modality === "embedding" && (
          <span className="shrink-0 font-mono text-xs text-embedding">
            embedding
          </span>
        )}
        {(svc.inflight_count ?? 0) > 0 && (
          <span className="shrink-0 font-mono text-xs text-accent">
            {svc.inflight_count} in-flight
          </span>
        )}
        <div className="hidden h-6 w-20 shrink-0 sm:block">
          <Sparkline
            data={sparkLineData}
            color={CHART_PALETTE[0]}
            height={24}
            xMin={xMin}
            xMax={xMax}
          />
        </div>
      </Link>
      <a
        href={serviceProxyUrl(svc.port)}
        target="_blank"
        rel="noopener noreferrer"
        className="hidden shrink-0 rounded-[3px] bg-elevated px-1.5 py-0.5 font-mono text-xs text-accent ring-1 ring-inset ring-border-strong transition-colors hover:bg-border-strong sm:inline"
      >
        :{svc.port}
      </a>
      {/* Sized to hold the three-button case at full size with a tight
          gap, so the right-aligned columns line up across rows however
          many actions a row shows. */}
      <div className="flex w-[110px] shrink-0 items-center justify-end gap-1.5 border-l border-border-default pl-3">
        {canStart && (
          <IconButton
            label={t("services.actions.start")}
            variant="primary"
            onClick={() => onAction("start")}
            disabled={pending}
          >
            <PlayIcon />
          </IconButton>
        )}
        {canStop && (
          <>
            <IconButton
              label={t("services.actions.restart")}
              variant="secondary"
              onClick={() => onAction("restart")}
              disabled={pending}
            >
              <RestartIcon />
            </IconButton>
            <IconButton
              label={t("services.actions.stop")}
              variant="danger"
              onClick={() => onAction("stop")}
              disabled={pending}
            >
              <StopIcon />
            </IconButton>
          </>
        )}
        {isDisabled ? (
          <IconButton
            label={t("services.actions.enable")}
            variant="secondary"
            onClick={() => onAction("enable")}
            disabled={pending}
          >
            <PowerIcon />
          </IconButton>
        ) : (
          <IconButton
            label={t("services.actions.disable")}
            variant="ghost"
            onClick={() => onAction("disable")}
            disabled={pending}
          >
            <DisableIcon />
          </IconButton>
        )}
      </div>
    </div>
  );
}

/// Where the service's memory would land, per device.
///
/// The breakdown is shown in preference to the total because it is the question
/// a full box actually raises — a service is rarely short of memory in general,
/// it is short of memory on one card. The total remains the fallback for a
/// placement that names no device, which is the same condition that leaves
/// `footprint_bytes` unset.
function Footprint({ svc }: { svc: ServiceSummary }) {
  const { t } = useTranslation();
  const devices = svc.footprint_devices ?? [];
  // An unplaceable service has no reservation to report, so the figure is the
  // model's demand — say so rather than letting it read as memory the service
  // is already holding.
  const needed =
    svc.fit_verdict?.kind === "does_not_fit"
      ? ` ${t("services.footprintNeeded")}`
      : "";

  if (devices.length === 0) {
    if (svc.footprint_bytes == null) return null;
    return (
      <span className="truncate font-mono text-xs text-tertiary">
        {formatBytes(svc.footprint_bytes)}
        {needed}
      </span>
    );
  }

  // The device and its size read as one phrase, so the cell is what gets the
  // fixed width rather than the two fields inside it — spacing them separately
  // aligns the columns at the cost of pulling the pair apart. Whole cells of a
  // uniform width put every row's second device at the same offset just as well.
  return (
    <span className="flex items-baseline overflow-hidden font-mono text-xs text-tertiary">
      {devices.map((dev, index) => (
        <span key={dev.device} className="flex shrink-0 items-baseline">
          {index > 0 && (
            <span className="px-1.5 text-border-strong" aria-hidden="true">
              /
            </span>
          )}
          {/* `w-[15ch]` is exact because the row is monospaced, and is the
              widest the pair can be: `gpu0` at four, a space, and ten for the
              longest `formatBytes` output — `1023.9 MiB`, not the `4.57 GiB`
              the common case suggests, since the sub-gibibyte branch carries an
              extra digit. */}
          <span className="w-[15ch] shrink-0">
            <span className="text-brass">{deviceLabel(dev.device)}</span>{" "}
            <span className="text-secondary">{formatBytes(dev.bytes)}</span>
          </span>
        </span>
      ))}
      {needed && <span className="shrink-0 pl-1">{needed.trim()}</span>}
    </span>
  );
}

/// `"gpu:0"` reads as `gpu0` here. The colon earns its place in a config file
/// and in the API, where the slot is an identifier; in a row of several devices
/// it is punctuation competing with the separator between them.
function deviceLabel(device: string): string {
  return device.replace(":", "");
}

type IconButtonProps = {
  label: string;
  variant: "primary" | "secondary" | "ghost" | "danger";
  onClick: () => void;
  disabled: boolean;
  children: React.ReactNode;
};

const ICON_VARIANT: Record<IconButtonProps["variant"], string> = {
  primary: "text-accent hover:bg-accent/15",
  secondary: "text-secondary hover:bg-elevated",
  ghost: "text-tertiary hover:bg-elevated hover:text-secondary",
  danger: "text-danger hover:bg-danger/15",
};

function IconButton({
  label,
  variant,
  onClick,
  disabled,
  children,
}: IconButtonProps) {
  return (
    <button
      type="button"
      title={label}
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
        onClick();
      }}
      disabled={disabled}
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md transition-colors disabled:opacity-40 ${ICON_VARIANT[variant]}`}
    >
      {children}
    </button>
  );
}

function fitVerdictColor(verdict: FitVerdict | null | undefined): string {
  switch (verdict?.kind) {
    case "does_not_fit":
      return "text-danger";
    case "needs_eviction":
      return "text-warning";
    default:
      return "text-primary";
  }
}

function fitVerdictTitle(
  verdict: FitVerdict | null | undefined,
  t: TFunction,
): string | undefined {
  switch (verdict?.kind) {
    case "does_not_fit": {
      // Reuse the same i18n strings the detail view renders, rather than a
      // third English spelling of a sentence that already exists twice.
      const detail = verdict.shortfalls
        .map((s) =>
          t("serviceDetail.shortfall", {
            device: s.device,
            requested: formatBytes(s.requested_bytes),
            available: formatBytes(s.available_bytes),
          }),
        )
        .join("; ");
      const headline = t("serviceDetail.noPlacementFits");
      return detail ? `${headline.replace(/\.$/, "")} — ${detail}` : headline;
    }
    case "needs_eviction":
      return t("serviceDetail.needsEviction");
    default:
      return undefined;
  }
}
