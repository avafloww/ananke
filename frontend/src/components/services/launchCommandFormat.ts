// Pure formatting helpers for rendering a LaunchCommand as a
// copy-pasteable shell command, used by ServiceLaunchCommand.tsx.

import type { LaunchCommand } from "../../api/client.ts";

export function renderCommand(cmd: LaunchCommand): string {
  const lines: string[] = [];
  if (cmd.env.length > 0) {
    if (cmd.env_inherit) {
      lines.push("# inherits daemon env + overrides below");
    } else {
      lines.push("# clean env — only vars below");
    }
  }
  for (const e of cmd.env) lines.push(`${e.key}=${shellQuote(e.value)}`);
  const [binary, ...rest] = cmd.argv;
  if (binary !== undefined) lines.push(shellQuote(binary));
  for (let i = 0; i < rest.length; i++) {
    const tok = rest[i]!;
    const next = rest[i + 1];
    if (tok.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      lines.push(`  ${shellQuote(tok)} ${shellQuote(next)}`);
      i++;
    } else {
      lines.push(`  ${shellQuote(tok)}`);
    }
  }
  return lines.join(" \\\n");
}

function shellQuote(s: string): string {
  if (s.length === 0) return "''";
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s;
  return `'${s.replace(/'/g, "'\\''")}'`;
}
