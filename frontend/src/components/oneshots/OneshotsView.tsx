// Oneshot job management (`/oneshots`). Lists active and historical
// oneshot jobs, with a submit form for creating new command-template
// jobs and a detail panel showing status + captured logs.

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  useOneshots,
  useCreateOneshot,
  useDeleteOneshot,
} from "../../api/hooks.ts";
import type { OneshotRequest } from "../../api/client.ts";
import { Button } from "../ui/Button.tsx";
import { Card } from "../ui/Card.tsx";
import { ViewHeader } from "../ui/ViewHeader.tsx";
import { Spinner } from "../ui/Spinner.tsx";
import { EmptyState } from "../ui/EmptyState.tsx";
import { OneshotForm, type OneshotFormState } from "./OneshotForm.tsx";
import { OneshotRow, OneshotDetail } from "./OneshotList.tsx";

export function OneshotsView() {
  const { t } = useTranslation();
  const oneshots = useOneshots();
  const createMut = useCreateOneshot();
  const deleteMut = useDeleteOneshot();
  const [showForm, setShowForm] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const sorted = useMemo(() => {
    if (!oneshots.data) return [];
    return [...oneshots.data].sort(
      (a, b) => b.submitted_at_ms - a.submitted_at_ms,
    );
  }, [oneshots.data]);

  const selected = sorted.find((o) => o.id === selectedId) ?? null;

  function handleSubmit(form: OneshotFormState) {
    const argv = form.command.trim().split(/\s+/).filter(Boolean);
    if (argv.length === 0) return;

    const req: OneshotRequest = {
      template: "command",
      command: argv,
      name: form.name.trim() || null,
      workdir: form.workdir.trim() || null,
      port: form.port ? Number(form.port) : null,
      priority: form.priority,
      ttl: form.ttl.trim() || null,
      allocation:
        form.allocationMode === "static"
          ? { mode: "static", reserve_gb: Number(form.reserveGb) }
          : {
              mode: "dynamic",
              min_reserve_gb: Number(form.minReserveGb),
              max_reserve_gb: Number(form.maxReserveGb),
            },
      devices: { placement: form.placement },
      health: form.healthPath.trim()
        ? {
            http: form.healthPath.trim(),
            timeout: form.healthTimeout.trim() || null,
          }
        : null,
    };

    createMut.mutate(req, {
      onSuccess: (resp) => {
        setSelectedId(resp.id);
        setShowForm(false);
      },
    });
  }

  return (
    <div className="flex h-full flex-col">
      <ViewHeader>
        <h1 className="eyebrow !text-primary">{t("nav.oneshots")}</h1>
        <Button
          type="button"
          variant="iris"
          size="sm"
          onClick={() => setShowForm((s) => !s)}
        >
          {showForm ? t("oneshots.cancel") : t("oneshots.newOneshot")}
        </Button>
        {oneshots.data && (
          <span className="ml-auto font-mono text-xs text-tertiary">
            {oneshots.data.length} total
          </span>
        )}
      </ViewHeader>

      <div className="flex-1 overflow-auto p-4">
        {showForm && (
          <OneshotForm
            onSubmit={handleSubmit}
            isPending={createMut.isPending}
            error={createMut.error?.message}
          />
        )}

        {oneshots.isPending && !oneshots.data ? (
          <div className="flex h-full items-center justify-center">
            <Spinner />
          </div>
        ) : sorted.length > 0 ? (
          <div className="space-y-4">
            <Card header={t("oneshots.jobs")} bodyClassName="p-0">
              <div className="divide-y divide-border-default">
                {sorted.map((o) => (
                  <OneshotRow
                    key={o.id}
                    oneshot={o}
                    selected={o.id === selectedId}
                    onSelect={() => setSelectedId(o.id)}
                    onDelete={() => {
                      deleteMut.mutate(o.id);
                      if (selectedId === o.id) setSelectedId(null);
                    }}
                    deletePending={
                      deleteMut.isPending && deleteMut.variables === o.id
                    }
                  />
                ))}
              </div>
            </Card>

            {selected && <OneshotDetail oneshot={selected} />}
          </div>
        ) : (
          <EmptyState message={t("oneshots.emptyState")} />
        )}
      </div>
    </div>
  );
}
