// Computed launch-command card for ServiceDetailView: shows the argv
// llama.cpp would be started with, both standalone and under current
// placement conditions.

import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useServiceCommand } from "../../api/hooks.ts";
import type { LaunchCommand } from "../../api/client.ts";
import { Spinner } from "../ui/Spinner.tsx";
import { CopyButton } from "../ui/CopyButton.tsx";
import { renderCommand } from "./launchCommandFormat.ts";

export function LaunchCommandSection({ name }: { name: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const { data, error, isPending } = useServiceCommand(name, open);

  return (
    <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer select-none text-xs text-tertiary hover:text-secondary">
        {t("serviceDetail.expandToCompute")}
      </summary>
      <div className="mt-2">
        {open && isPending && <Spinner />}
        {error && (
          <span className="text-sm text-danger">
            {t("serviceDetail.cannotCompute", { error: error.message })}
          </span>
        )}
        {data && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <CommandPanel
              label={t("serviceDetail.standalone")}
              command={data.on_empty}
            />
            <CommandPanel
              label={t("serviceDetail.currentConditions")}
              command={data.active ?? null}
            />
          </div>
        )}
      </div>
    </details>
  );
}

function CommandPanel({
  label,
  command,
}: {
  label: string;
  command: LaunchCommand | null;
}) {
  const { t } = useTranslation();
  return (
    <div>
      <div className="mb-1 flex items-center gap-2">
        <span className="text-xs text-tertiary">{label}</span>
        {command && (
          <>
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                command.env_inherit
                  ? "bg-emerald-500/15 text-emerald-400"
                  : "bg-amber-500/15 text-amber-400"
              }`}
              title={
                command.env_inherit
                  ? t("serviceDetail.envInheritOnHint")
                  : t("serviceDetail.envInheritOffHint")
              }
            >
              {command.env_inherit
                ? t("serviceDetail.envInheritOn")
                : t("serviceDetail.envInheritOff")}
            </span>
            <CopyButton value={renderCommand(command)} />
          </>
        )}
      </div>
      {command ? (
        <pre className="overflow-x-auto whitespace-pre-wrap break-all rounded-sm bg-base p-2 font-mono text-xs text-primary">
          {renderCommand(command)}
        </pre>
      ) : (
        <div className="flex items-center justify-center rounded-sm bg-base p-4 text-xs text-danger">
          {t("serviceDetail.doesNotFitCurrent")}
        </div>
      )}
    </div>
  );
}
