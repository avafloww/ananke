// Pure filtering rules for the logs viewer, kept out of LogsViewer.tsx so
// that file exports components only and its HMR boundary stays clean.

import type { LogLine } from "../../api/client.ts";

/** The stream vocabulary the viewer filters on. */
export type StreamFilter = "both" | "stdout" | "stderr" | "combined";

/**
 * Apply the viewer's three filters to a line buffer.
 *
 * `both` deliberately means every stream, container `combined` output
 * included — and asking for `stdout` must not sweep it up, because a
 * container's merged output is not the child's stdout.
 */
export function filterLines(
  lines: readonly LogLine[],
  streamFilter: StreamFilter,
  runFilter: number | null,
  search: string,
): readonly LogLine[] {
  let result = lines;
  if (streamFilter !== "both")
    result = result.filter((l) => l.stream === streamFilter);
  if (runFilter !== null) result = result.filter((l) => l.run_id === runFilter);
  if (search) {
    const lower = search.toLowerCase();
    result = result.filter((l) => l.line.toLowerCase().includes(lower));
  }
  return result;
}
