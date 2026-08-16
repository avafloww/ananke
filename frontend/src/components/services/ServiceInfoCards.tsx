// Info-card grids for ServiceDetailView: model metadata, service
// configuration, serving/runtime knobs, and the VRAM estimate.

import { useTranslation } from "react-i18next";

import type {
  EstimateSummary,
  IkParams,
  ModelInfo,
  RuntimeInfo,
  ServingConfig,
  ServiceDetail,
} from "../../api/client.ts";
import {
  formatBytes,
  formatDuration,
  formatParameterCount,
  formatTimestamp,
  relativeTime,
} from "../../util.ts";
import { CopyButton } from "../ui/CopyButton.tsx";

export function ModelInfoGrid({ model }: { model: ModelInfo }) {
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {model.model_name && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.name")}</dt>
          <dd className="text-primary">{model.model_name}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.file")}</dt>
      <dd className="flex items-center gap-1">
        <span className="font-mono text-xs text-primary">
          {model.file_name}
        </span>
        <CopyButton value={model.file_name} />
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.architecture")}</dt>
      <dd className="font-mono text-primary">{model.architecture}</dd>
      {model.parameter_count !== undefined &&
        model.parameter_count !== null && (
          <>
            <dt className="text-tertiary">{t("serviceDetail.parameters")}</dt>
            <dd
              className="font-mono text-primary"
              title={`${model.parameter_count.toLocaleString()} parameters`}
            >
              {formatParameterCount(model.parameter_count)}
            </dd>
          </>
        )}
      <dt className="text-tertiary">{t("serviceDetail.onDisk")}</dt>
      <dd className="font-mono text-primary">
        {formatBytes(model.total_tensor_bytes)}
      </dd>
      {model.block_count !== undefined && model.block_count !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.layers")}</dt>
          <dd className="font-mono text-primary">{model.block_count}</dd>
        </>
      )}
      {model.trained_context_length !== undefined &&
        model.trained_context_length !== null && (
          <>
            <dt className="text-tertiary">
              {t("serviceDetail.trainedContext")}
            </dt>
            <dd className="font-mono text-primary">
              {t("serviceDetail.tokensValue", {
                value: model.trained_context_length.toLocaleString(),
              })}
            </dd>
          </>
        )}
      {model.shard_count > 1 && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.shards")}</dt>
          <dd className="font-mono text-primary">{model.shard_count}</dd>
        </>
      )}
      {model.license && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.license")}</dt>
          <dd className="font-mono text-xs text-primary">{model.license}</dd>
        </>
      )}
    </dl>
  );
}

export function ConfigGrid({ detail }: { detail: ServiceDetail }) {
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      <dt className="text-tertiary">{t("serviceDetail.template")}</dt>
      <dd className="font-mono text-primary">{detail.template}</dd>
      {detail.container && (
        <>
          <dt className="text-tertiary">
            {t("serviceDetail.containerRuntime")}
          </dt>
          <dd className="font-mono text-primary">
            {detail.container.runtime} · {detail.container.image}
          </dd>
          <dt className="text-tertiary">
            {t("serviceDetail.containerNetwork")}
          </dt>
          <dd className="font-mono text-primary">{detail.container.network}</dd>
          {detail.container.container_id && (
            <>
              <dt className="text-tertiary">
                {t("serviceDetail.containerId")}
              </dt>
              <dd className="flex items-center gap-1">
                <span className="font-mono text-xs text-primary">
                  {detail.container.container_id.slice(0, 12)}
                </span>
                <CopyButton value={detail.container.container_id} />
              </dd>
            </>
          )}
          {detail.container.container_name && (
            <>
              <dt className="text-tertiary">
                {t("serviceDetail.containerName")}
              </dt>
              <dd className="font-mono text-xs text-primary">
                {detail.container.container_name}
              </dd>
            </>
          )}
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.context")}</dt>
      <dd className="font-mono text-primary">
        {detail.estimate
          ? t("serviceDetail.tokensValue", {
              value: detail.estimate.configured_context.toLocaleString(),
            })
          : "—"}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.idleTimeout")}</dt>
      <dd className="font-mono text-primary">
        {detail.lifecycle === "persistent"
          ? t("serviceDetail.neverPersistent")
          : formatDuration(detail.idle_timeout_ms)}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.lastUsed")}</dt>
      <dd className="font-mono text-primary">
        {detail.last_used_ms != null
          ? `${relativeTime(detail.last_used_ms)} (${formatTimestamp(detail.last_used_ms)})`
          : "—"}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.runId")}</dt>
      <dd className="font-mono text-primary">{detail.run_id ?? "—"}</dd>
      <dt className="text-tertiary">{t("serviceDetail.privatePort")}</dt>
      <dd className="font-mono text-primary">:{detail.private_port}</dd>
      {detail.rolling_mean != null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.estimatorDrift")}</dt>
          <dd className="font-mono text-primary">
            {detail.rolling_mean.toFixed(3)}×{" "}
            <span className="text-tertiary">
              {t("serviceDetail.samples", { value: detail.rolling_samples })}
            </span>
          </dd>
        </>
      )}
      {detail.rolling_mean_host != null && (
        <>
          <dt className="text-tertiary">
            {t("serviceDetail.estimatorDriftHost")}
          </dt>
          <dd className="font-mono text-primary">
            {detail.rolling_mean_host.toFixed(3)}×{" "}
            <span className="text-tertiary">
              {t("serviceDetail.samples", {
                value: detail.rolling_samples_host,
              })}
            </span>
          </dd>
        </>
      )}
    </dl>
  );
}

// The Serving card: runtime kind (+ik knobs), binary, and the curated
// perf/memory knobs, with derived values (per-slot context, fit
// margins) that no config key or argv flag states directly. Paired
// values share a row to keep the card near the Model card's height;
// the card body scroll-caps as insurance.
export function ServingGrid({
  serving,
  runtime,
}: {
  serving: ServingConfig;
  runtime: RuntimeInfo | null;
}) {
  const { t } = useTranslation();
  const flag = (on: boolean) =>
    on ? t("serviceDetail.flagOn") : t("serviceDetail.flagOff");
  const binaryName = serving.binary.split("/").at(-1) ?? serving.binary;
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {runtime && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.runtime")}</dt>
          <dd className="font-mono text-primary">{runtime.kind}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.binary")}</dt>
      <dd className="flex items-center gap-1">
        <span className="font-mono text-xs text-primary">{binaryName}</span>
        <CopyButton value={serving.binary} />
      </dd>
      {runtime?.ik && <IkParamRows ik={runtime.ik} />}
      <dt className="text-tertiary">{t("serviceDetail.kvCache")}</dt>
      <dd className="font-mono text-primary">
        {serving.cache_type_k} / {serving.cache_type_v}
        {serving.flash_attn && <span className="text-tertiary"> · fa</span>}
      </dd>
      <dt className="text-tertiary">{t("serviceDetail.parallelSlots")}</dt>
      <dd className="font-mono text-primary">
        {serving.parallel}
        {serving.kv_unified && (
          <span className="text-tertiary">
            {" "}
            · {t("serviceDetail.kvUnified")}
          </span>
        )}
      </dd>
      {serving.effective_context_per_slot != null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.perSlotContext")}</dt>
          <dd className="font-mono text-primary">
            {t("serviceDetail.tokensValue", {
              value: serving.effective_context_per_slot.toLocaleString(),
            })}
          </dd>
        </>
      )}
      {serving.spec_type && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.specDecode")}</dt>
          <dd className="font-mono text-primary">
            {serving.spec_type}
            {serving.draft_model && (
              <span className="text-tertiary"> · {serving.draft_model}</span>
            )}
          </dd>
        </>
      )}
      {serving.expert_offload !== "off" && (
        <>
          <dt className="text-tertiary">
            {t("serviceDetail.expertOffloadMode")}
          </dt>
          <dd className="font-mono text-primary">{serving.expert_offload}</dd>
        </>
      )}
      {(serving.batch_size != null || serving.ubatch_size != null) && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.batchSizes")}</dt>
          <dd className="font-mono text-primary">
            {serving.batch_size ?? "—"} / {serving.ubatch_size ?? "—"}
          </dd>
        </>
      )}
      {(serving.threads != null || serving.threads_batch != null) && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.threadsRow")}</dt>
          <dd className="font-mono text-primary">
            {serving.threads ?? "—"} / {serving.threads_batch ?? "—"}
          </dd>
        </>
      )}
      {serving.numa && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.numaRow")}</dt>
          <dd className="font-mono text-primary">{serving.numa}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.memoryFlags")}</dt>
      <dd className="font-mono text-primary">
        {flag(serving.mmap)} / {flag(serving.mlock)}
      </dd>
    </dl>
  );
}

// ik_llama.cpp runtime parameters, rendered inside ConfigGrid's <dl>.
function IkParamRows({ ik }: { ik: IkParams }) {
  const { t } = useTranslation();
  const flag = (on: boolean) =>
    on ? t("serviceDetail.flagOn") : t("serviceDetail.flagOff");
  return (
    <>
      {ik.mla !== undefined && ik.mla !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.ikMla")}</dt>
          <dd className="font-mono text-primary">{ik.mla}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.ikDsa")}</dt>
      <dd className="font-mono text-primary">{flag(ik.dsa)}</dd>
      {ik.attn_max_batch !== undefined && ik.attn_max_batch !== null && (
        <>
          <dt className="text-tertiary">{t("serviceDetail.ikAmb")}</dt>
          <dd className="font-mono text-primary">{ik.attn_max_batch}</dd>
        </>
      )}
      <dt className="text-tertiary">{t("serviceDetail.ikRtr")}</dt>
      <dd className="font-mono text-primary">{flag(ik.runtime_repack)}</dd>
    </>
  );
}

export function EstimateGrid({
  estimate,
  observedPeakBytes,
}: {
  estimate: EstimateSummary;
  observedPeakBytes: number;
}) {
  const { t } = useTranslation();
  const total =
    estimate.weights_bytes +
    estimate.kv_bytes_for_context +
    estimate.compute_buffer_bytes_per_device;
  return (
    <div className="space-y-2">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-tertiary">{t("serviceDetail.weights")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.weights_bytes)}
        </dd>
        <dt className="text-tertiary">
          {t("serviceDetail.kvAtContext", {
            ctx: estimate.configured_context.toLocaleString(),
          })}
        </dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.kv_bytes_for_context)}
        </dd>
        <dt className="text-tertiary">{t("serviceDetail.computeDev")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(estimate.compute_buffer_bytes_per_device)}
        </dd>
        <dt className="text-tertiary">{t("serviceDetail.total")}</dt>
        <dd className="font-mono text-primary">
          {formatBytes(total)}
          {observedPeakBytes > 0 && (
            <span className="text-tertiary">
              {" "}
              {t("serviceDetail.peak", {
                bytes: formatBytes(observedPeakBytes),
              })}
            </span>
          )}
        </dd>
      </dl>
    </div>
  );
}
