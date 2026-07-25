// Per-service stats card + charts (request rate, token throughput,
// tokens/sec, errors/restarts) for ServiceDetailView.

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { useMetrics, useRestarts } from "../../api/hooks.ts";
import {
  aggregateBuckets,
  toErrorsRestartsData,
} from "../../api/metrics-aggregate.ts";
import { bucketMsOf, metricsWindow } from "../../util.ts";
import { Card } from "../ui/Card.tsx";
import { Stat } from "../ui/Stat.tsx";
import { TimeWindowSelect } from "../ui/TimeWindowSelect.tsx";
import {
  TIME_WINDOW_PRESETS,
  windowLabel,
  type TimeWindow,
} from "../ui/timeWindow.ts";
import { Chart } from "../ui/Chart.tsx";
import { CHART_PALETTE } from "../ui/chart-palette.ts";

export function ServiceMetrics({ name }: { name: string }) {
  const { t } = useTranslation();
  const [timeWindow, setTimeWindow] = useState<TimeWindow>({
    kind: "relative",
    durationMs: TIME_WINDOW_PRESETS[0]!.durationMs,
  });
  // Freeze `now` per window selection so the query key doesn't churn.
  const { since, until, end, bucket } = useMemo(
    () => metricsWindow(timeWindow),
    [timeWindow],
  );
  const xMin = since / 1000;
  const xMax = end / 1000;

  const metrics = useMetrics({ service: name, since, until, bucket });
  const buckets = aggregateBuckets(metrics.data?.buckets ?? []);
  const restarts = useRestarts(name, since, until ?? end);
  const errorsRestartsData = toErrorsRestartsData(
    buckets,
    restarts.data ?? [],
    since,
    end,
    bucketMsOf(bucket),
  );

  const totalRequests = buckets.reduce((s, b) => s + b.requestCount, 0);
  const totalErrors = buckets.reduce((s, b) => s + b.errorCount, 0);
  const totalInputTokens = buckets.reduce((s, b) => s + b.promptTokens, 0);
  const totalOutputTokens = buckets.reduce((s, b) => s + b.completionTokens, 0);
  const avgLatency =
    buckets.reduce((s, b) => s + b.totalDurationMs, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.timedRequests, 0),
    );
  const avgOutputTps =
    buckets.reduce((s, b) => s + b.totalWeightedOutputTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.outputTpsRequests, 0),
    );
  const avgInputTps =
    buckets.reduce((s, b) => s + b.totalWeightedInputTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.inputTpsRequests, 0),
    );
  const avgEffectiveTps =
    buckets.reduce((s, b) => s + b.totalWeightedEffectiveTps, 0) /
    Math.max(
      1,
      buckets.reduce((s, b) => s + b.effectiveTpsRequests, 0),
    );
  // When no request in the window carries an input/output split (all
  // non-streaming with no engine timings), show the end-to-end effective throughput
  // instead of two zeros.
  const hasSplitTps = buckets.some(
    (b) => b.outputTpsRequests > 0 || b.inputTpsRequests > 0,
  );

  return (
    <div className="space-y-4">
      <Card
        header={t("serviceDetail.stats")}
        headerAction={<TimeWindowSelect onChange={setTimeWindow} />}
      >
        {/* The effective throughput stat is always shown; the input/output
            decode rates are added alongside it when the window has
            split-capable rows. The column count tracks the tile count. */}
        <div
          className={`grid grid-cols-2 gap-4 ${
            hasSplitTps ? "sm:grid-cols-8" : "sm:grid-cols-6"
          }`}
        >
          <Stat
            label={t("serviceDetail.requestsInPeriod", {
              range: windowLabel(timeWindow, t),
            })}
            value={totalRequests}
          />
          <Stat label={t("serviceDetail.errors")} value={totalErrors} />
          <Stat
            label={t("serviceDetail.inputTokens")}
            value={totalInputTokens.toLocaleString()}
          />
          <Stat
            label={t("serviceDetail.outputTokens")}
            value={totalOutputTokens.toLocaleString()}
          />
          <Stat
            label={t("serviceDetail.avgLatency")}
            value={totalRequests > 0 ? `${Math.round(avgLatency)}ms` : "—"}
          />
          {hasSplitTps && (
            <>
              <Stat
                label={t("serviceDetail.avgTpsIn")}
                value={totalRequests > 0 ? avgInputTps.toFixed(1) : "—"}
              />
              <Stat
                label={t("serviceDetail.avgTpsOut")}
                value={totalRequests > 0 ? avgOutputTps.toFixed(1) : "—"}
              />
            </>
          )}
          <Stat
            label={t("serviceDetail.avgTpsEffective")}
            value={totalRequests > 0 ? avgEffectiveTps.toFixed(1) : "—"}
          />
        </div>
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card header={t("stats.requestRate")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              buckets.map((b) => b.requestCount),
            ]}
            series={[
              {
                label: t("stats.requests"),
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
            ]}
          />
        </Card>
        <Card header={t("stats.tokenThroughput")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              buckets.map((b) => b.promptTokens),
              buckets.map((b) => b.completionTokens),
            ]}
            series={[
              {
                label: t("stats.tokensIn"),
                stroke: CHART_PALETTE[0],
                fill: "rgba(139,124,248,0.08)",
              },
              {
                label: t("stats.tokensOut"),
                stroke: CHART_PALETTE[1],
                fill: "rgba(69,201,138,0.08)",
              },
            ]}
          />
        </Card>
        {/* The end-to-end effective line is always shown; the input/output
            decode rates are overlaid on top of it when available. */}
        <Card header={t("stats.tokensPerSecond")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={[
              buckets.map((b) => b.ts),
              ...(hasSplitTps
                ? [
                    buckets.map((b) =>
                      b.inputTpsRequests > 0
                        ? b.totalWeightedInputTps / b.inputTpsRequests
                        : null,
                    ),
                    buckets.map((b) =>
                      b.outputTpsRequests > 0
                        ? b.totalWeightedOutputTps / b.outputTpsRequests
                        : null,
                    ),
                  ]
                : []),
              buckets.map((b) =>
                b.effectiveTpsRequests > 0
                  ? b.totalWeightedEffectiveTps / b.effectiveTpsRequests
                  : null,
              ),
            ]}
            series={[
              ...(hasSplitTps
                ? [
                    {
                      label: t("stats.tpsIn"),
                      stroke: CHART_PALETTE[0],
                      fill: "rgba(139,124,248,0.08)",
                      unit: "tok/s",
                    },
                    {
                      label: t("stats.tpsOut"),
                      stroke: CHART_PALETTE[1],
                      fill: "rgba(69,201,138,0.08)",
                      unit: "tok/s",
                    },
                  ]
                : []),
              {
                label: t("stats.tpsEffective"),
                stroke: CHART_PALETTE[2],
                fill: "rgba(224,168,60,0.08)",
                unit: "tok/s",
              },
            ]}
          />
        </Card>
        {/* Errors and auto-restarts on one card: the error storm and the
            watchdog firing it provokes belong on the same axis. */}
        <Card header={t("stats.errorsRestarts")}>
          <Chart
            xMin={xMin}
            xMax={xMax}
            data={errorsRestartsData}
            series={[
              {
                label: t("stats.errors"),
                stroke: CHART_PALETTE[5],
                fill: "rgba(239,90,90,0.08)",
              },
              {
                label: t("stats.restarts"),
                stroke: CHART_PALETTE[2],
                fill: "rgba(224,168,60,0.08)",
              },
            ]}
          />
        </Card>
      </div>
    </div>
  );
}
