// Filterable model picker used in ChatView's composer.

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { type ServiceSummary } from "../../api/client.ts";
import { Badge } from "../ui/Badge.tsx";
import { StatusDot } from "../ui/StatusDot.tsx";

export function ModelDropdown({
  models,
  selected,
  onSelect,
  className = "",
}: {
  models: ServiceSummary[];
  selected: string | null;
  onSelect: (name: string) => void;
  className?: string;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        close();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
    }
  }, [open]);

  function close() {
    setOpen(false);
    setFilter("");
  }

  const selectedSvc = models.find((s) => s.name === selected);
  const filtered = filter
    ? models.filter((s) => s.name.toLowerCase().includes(filter.toLowerCase()))
    : models;

  return (
    <div ref={ref} className={`relative ${className}`}>
      {open ? (
        <input
          ref={inputRef}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              close();
            } else if (e.key === "Enter" && filtered.length > 0) {
              onSelect(filtered[0].name);
              close();
            }
          }}
          placeholder={t("chat.filterModels")}
          className="h-7 w-full rounded-sm border border-border-default bg-surface px-2 text-sm text-primary placeholder:text-tertiary focus:border-accent focus:outline-none"
        />
      ) : (
        <button
          onClick={() => setOpen(true)}
          className="flex h-7 w-full min-w-0 items-center gap-2 rounded-sm border border-border-default bg-surface px-2 text-sm text-primary hover:bg-elevated"
        >
          {selectedSvc ? (
            <>
              <StatusDot state={selectedSvc.state} />
              <span className="min-w-0 truncate font-mono">
                {selectedSvc.name}
              </span>
              {selectedSvc.has_mmproj && (
                <Badge variant="vision" className="shrink-0">
                  vision
                </Badge>
              )}
            </>
          ) : (
            <span className="text-tertiary">
              {t("chat.selectModelPlaceholder")}
            </span>
          )}
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="ml-auto shrink-0 text-tertiary"
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
      )}
      {open && (
        <div className="absolute bottom-full left-0 z-20 mb-1 max-h-72 w-full overflow-auto rounded-md border border-border-default bg-surface shadow-lg">
          {filtered.length === 0 ? (
            <div className="px-3 py-2 text-sm text-tertiary">
              {t("chat.noMatchingModels")}
            </div>
          ) : (
            filtered.map((s) => (
              <button
                key={s.name}
                onClick={() => {
                  onSelect(s.name);
                  close();
                }}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-elevated ${
                  s.name === selected ? "bg-elevated" : ""
                }`}
              >
                <StatusDot state={s.state} />
                <span className="font-mono text-sm text-primary">{s.name}</span>
                {s.has_mmproj && <Badge variant="vision">vision</Badge>}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
