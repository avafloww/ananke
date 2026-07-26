// Placement preview + current/override pledge display for
// ServiceDetailView's "Placement" card.

import { useTranslation } from "react-i18next";

import type {
  DevicePlacement,
  DeviceShortfall,
  PlacementPreview,
  ServiceDetail,
} from "../../api/client.ts";
import { formatBytes } from "../../util.ts";
import { Badge } from "../ui/Badge.tsx";
import { Bar, type BarSegment } from "../ui/Bar.tsx";

export function PlacementSection({
  current,
  placementOverride,
  placement,
}: {
  current: ServiceDetail["current_allocation"];
  placementOverride: ServiceDetail["placement_override"];
  placement: PlacementPreview | null;
}) {
  const { t } = useTranslation();
  const hasCurrent = Object.keys(current).length > 0;
  const hasOverride = Object.keys(placementOverride).length > 0;
  const hasPlacement = placement !== null;

  if (!hasCurrent && !hasOverride && !hasPlacement) return null;

  return (
    <div className="space-y-3">
      {hasPlacement && (
        <div>
          <div className="mb-1 flex items-center gap-2">
            <span className="text-xs text-tertiary">
              {t("serviceDetail.preview")}
            </span>
            <FitBadge verdict={placement.verdict} />
          </div>
          {placement.devices.length > 0 ? (
            <div className="space-y-1.5">
              {placement.devices.map((dev) => (
                <PlacementBar
                  key={dev.device}
                  device={dev}
                  verdict={placement.verdict}
                />
              ))}
            </div>
          ) : (
            <div className="space-y-1">
              <span className="text-xs text-tertiary">
                {t("serviceDetail.noPlacementFits")}
              </span>
              {placement.verdict.kind === "does_not_fit" && (
                <ShortfallList shortfalls={placement.verdict.shortfalls} />
              )}
            </div>
          )}
          {placement.expert_offload_bytes > 0 && (
            <div className="mt-1.5 text-xs text-tertiary">
              {t("serviceDetail.expertOffload", {
                layers: placement.expert_offload_layers,
                bytes: formatBytes(placement.expert_offload_bytes),
                count: placement.expert_offload_layers,
              })}
            </div>
          )}
        </div>
      )}

      {(hasCurrent || hasOverride) && (
        <div className="grid grid-cols-2 gap-x-4 text-sm">
          {hasCurrent && (
            <div>
              <div className="mb-0.5 text-xs text-tertiary">
                {t("serviceDetail.currentPledge")}
              </div>
              <ul className="font-mono text-xs text-primary">
                {Object.entries(current).map(([slot, mb]) => (
                  <li key={slot}>
                    <span className="text-tertiary">{slot}:</span>{" "}
                    {mb.toLocaleString()} MiB
                  </li>
                ))}
              </ul>
            </div>
          )}
          {hasOverride && (
            <div>
              <div
                className="mb-0.5 text-xs text-tertiary"
                title={t("serviceDetail.configuredOverrideTitle")}
              >
                {t("serviceDetail.configuredOverride")}
              </div>
              <ul className="font-mono text-xs text-primary">
                {Object.entries(placementOverride).map(([slot, mb]) => (
                  <li key={slot}>
                    <span className="text-tertiary">{slot}:</span>{" "}
                    {mb.toLocaleString()} MiB
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Per-device breakdown of why a placement failed. Naming the device matters:
// the binding constraint is frequently host RAM rather than the GPUs, and the
// ids match `GET /api/devices` so they can be cross-referenced.
function ShortfallList({
  shortfalls,
}: {
  readonly shortfalls: readonly DeviceShortfall[];
}) {
  const { t } = useTranslation();
  // An empty list means the failure had no device to point at — a config
  // error, or no eligible device at all. Say so rather than rendering a bare
  // "no placement is possible" with nothing under it.
  if (shortfalls.length === 0) {
    return (
      <p className="text-xs text-tertiary">
        {t("serviceDetail.noEligibleDevices")}
      </p>
    );
  }
  return (
    <ul className="font-mono text-xs text-tertiary">
      {shortfalls.map((s) => (
        <li key={s.device}>
          {t("serviceDetail.shortfall", {
            device: s.device,
            requested: formatBytes(s.requested_bytes),
            available: formatBytes(s.available_bytes),
          })}
        </li>
      ))}
    </ul>
  );
}

function PlacementBar({
  device,
  verdict,
}: {
  device: DevicePlacement;
  verdict: PlacementPreview["verdict"];
}) {
  const { t } = useTranslation();
  const {
    device: name,
    bytes,
    max_bytes,
    used_by_others_bytes,
    total_bytes,
  } = device;
  const hasBar = total_bytes > 0;
  const canGrow = max_bytes > bytes;

  const thisVariant: BarSegment["variant"] =
    verdict.kind === "needs_eviction" ? "growth" : "used";

  const segments: BarSegment[] = [
    {
      variant: "reserved",
      bytes: used_by_others_bytes,
      label: t("serviceDetail.usedByOthers"),
    },
    {
      variant: thisVariant,
      bytes,
      label: t("serviceDetail.thisService"),
    },
    ...(canGrow
      ? [
          {
            variant: thisVariant,
            bytes: max_bytes - bytes,
            label: t("serviceDetail.growthHeadroom"),
          } satisfies BarSegment,
        ]
      : []),
    {
      variant: "headroom",
      bytes: Math.max(
        0,
        total_bytes - used_by_others_bytes - Math.max(bytes, max_bytes),
      ),
    },
  ];

  return (
    <div className="text-xs">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-primary">{name}</span>
        <span className="font-mono text-tertiary">
          {formatBytes(bytes)}
          {canGrow && <>&ndash;{formatBytes(max_bytes)}</>}
          {hasBar && <> / {formatBytes(total_bytes)}</>}
        </span>
      </div>
      {hasBar && (
        <Bar total={total_bytes} segments={segments} className="mt-0.5" />
      )}
    </div>
  );
}

function FitBadge({ verdict }: { verdict: PlacementPreview["verdict"] }) {
  const { t } = useTranslation();
  const map: Record<
    PlacementPreview["verdict"]["kind"],
    { variant: "success" | "warning" | "danger"; label: string }
  > = {
    fits: { variant: "success", label: t("serviceDetail.fitsNow") },
    needs_eviction: {
      variant: "warning",
      label: t("serviceDetail.needsEviction"),
    },
    does_not_fit: { variant: "danger", label: t("serviceDetail.doesNotFit") },
  };
  const { variant, label } = map[verdict.kind];
  return <Badge variant={variant}>{label}</Badge>;
}
