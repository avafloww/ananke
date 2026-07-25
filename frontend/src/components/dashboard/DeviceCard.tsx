// Device VRAM card (usage bar + memory sparkline) for DashboardView.

import { useMemo } from "react";
import { Link } from "react-router-dom";

import type { DeviceSummary, DeviceSampleResponse } from "../../api/client.ts";
import { formatBytes } from "../../util.ts";
import { Bar, type BarSegment } from "../ui/Bar.tsx";
import { Badge } from "../ui/Badge.tsx";
import { Chart } from "../ui/Chart.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";

export function DeviceCard({
  device,
  samples,
  xMin,
  xMax,
}: {
  device: DeviceSummary;
  samples: DeviceSampleResponse[];
  xMin: number;
  xMax: number;
}) {
  const total = device.total_bytes;
  const used = total - device.free_bytes;
  const pledged = device.reservations.reduce((sum, r) => sum + r.bytes, 0);
  const pledgedExtra = pledged > used ? pledged - used : 0;

  const segments: BarSegment[] = [
    { variant: "used", bytes: used, label: "used" },
    { variant: "growth", bytes: pledgedExtra, label: "pledged" },
    { variant: "headroom", bytes: Math.max(0, total - used - pledgedExtra) },
  ];

  const chartData = useMemo(() => {
    const filtered = samples
      .filter((s) => s.device === device.id)
      .sort((a, b) => a.timestamp_ms - b.timestamp_ms);
    return [
      filtered.map((s) => Math.floor(s.timestamp_ms / 1000)),
      filtered.map((s) => s.used_bytes / 1e9),
    ] as (number | null)[][];
  }, [samples, device.id]);

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-xs text-primary">{device.id}</span>
        <span className="text-xs text-tertiary">{device.name}</span>
      </div>
      <Chart
        data={chartData}
        series={[
          {
            label: "Used",
            stroke: CHART_PALETTE[0],
            fill: "rgba(139,124,248,0.08)",
            unit: "GB",
          },
        ]}
        height={100}
        xMin={xMin}
        xMax={xMax}
      />
      <Bar total={total} segments={segments} />
      <div className="text-xs text-tertiary">
        {formatBytes(used)} / {formatBytes(total)}
        {pledged > 0 && <> · {formatBytes(pledged)} pledged</>}
      </div>
      {device.reservations.length > 0 && (
        <div className="space-y-0.5">
          {device.reservations.map((r) => (
            <div key={r.service} className="flex items-center gap-2 text-sm">
              <Link
                to={`/services/${encodeURIComponent(r.service)}`}
                className="font-mono text-xs text-accent hover:underline"
              >
                {r.service}
              </Link>
              <span className="font-mono text-xs text-tertiary">
                {formatBytes(r.bytes)}
              </span>
              {r.elastic && <Badge variant="accent">elastic</Badge>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
