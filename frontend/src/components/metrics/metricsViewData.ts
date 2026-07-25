// Pure data-shaping helpers for MetricsView charts.

import type { DeviceSampleResponse } from "../../api/client.ts";
import type {
  AggregatedBucket,
  PerServiceSeries,
} from "../../api/metrics-aggregate.ts";

type MemorySeries = {
  device: string;
  data: { ts: number; usedBytes: number }[];
};

export function groupDeviceSamples(
  samples: DeviceSampleResponse[],
): MemorySeries[] {
  const byDevice = new Map<string, { ts: number; usedBytes: number }[]>();
  for (const s of samples) {
    const arr = byDevice.get(s.device) ?? [];
    arr.push({
      ts: Math.floor(s.timestamp_ms / 1000),
      usedBytes: s.used_bytes,
    });
    byDevice.set(s.device, arr);
  }
  return Array.from(byDevice.entries()).map(([device, data]) => ({
    device,
    data: data.sort((a, b) => a.ts - b.ts),
  }));
}

export type ChartField =
  | "requestCount"
  | "promptTokens"
  | "completionTokens"
  | "errorCount"
  | "totalDurationMs"
  | "outputTps"
  | "inputTps"
  | "effectiveTps";

export function resolveField(
  b: AggregatedBucket,
  field: ChartField,
): number | null {
  switch (field) {
    case "totalDurationMs":
      return b.timedRequests > 0 ? b.totalDurationMs / b.timedRequests : null;
    case "outputTps":
      return b.outputTpsRequests > 0
        ? b.totalWeightedOutputTps / b.outputTpsRequests
        : null;
    case "inputTps":
      return b.inputTpsRequests > 0
        ? b.totalWeightedInputTps / b.inputTpsRequests
        : null;
    case "effectiveTps":
      return b.effectiveTpsRequests > 0
        ? b.totalWeightedEffectiveTps / b.effectiveTpsRequests
        : null;
    default:
      return b[field] as number;
  }
}

// Whether any bucket in the window carries an input/output split. When
// none do (all non-streaming with no engine timings), the token-rate
// panel falls back to the effective line instead of showing two zeros.
export function hasSplitTps(buckets: AggregatedBucket[]): boolean {
  return buckets.some((b) => b.outputTpsRequests > 0 || b.inputTpsRequests > 0);
}

export function toLineData(
  buckets: AggregatedBucket[],
  field: ChartField,
): (number | null)[][] {
  return [buckets.map((b) => b.ts), buckets.map((b) => resolveField(b, field))];
}

export function toMultiSeriesData(
  seriesList: PerServiceSeries[],
  field: ChartField,
): (number | null)[][] {
  const allTs = new Set<number>();
  for (const s of seriesList) {
    for (const b of s.buckets) allTs.add(b.ts);
  }
  const ts = Array.from(allTs).sort((a, b) => a - b);
  const result: (number | null)[][] = [ts];
  for (const s of seriesList) {
    const map = new Map(s.buckets.map((b) => [b.ts, b]));
    const values = ts.map((t) => {
      const b = map.get(t);
      if (!b) return null;
      return resolveField(b, field);
    });
    result.push(values);
  }
  return result;
}

export function toMemoryData(series: MemorySeries[]): (number | null)[][] {
  if (series.length === 0) return [[]];
  const allTs = new Set<number>();
  for (const s of series) {
    for (const d of s.data) allTs.add(d.ts);
  }
  const ts = Array.from(allTs).sort((a, b) => a - b);
  const result: (number | null)[][] = [ts];
  for (const s of series) {
    const map = new Map(s.data.map((d) => [d.ts, d.usedBytes / 1e9]));
    result.push(ts.map((t) => map.get(t) ?? null));
  }
  return result;
}
