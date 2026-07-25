// Markdown rendering pipeline shared by ChatView's message bubbles
// (both the answer and the reasoning trace).

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

// Renders assistant markdown (both the answer and the reasoning trace)
// through one pipeline. react-markdown escapes raw HTML by default, so no
// pre-sanitization is needed or wanted (see #24).
export function MarkdownContent({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeHighlight, rehypeKatex]}
      components={{
        // Fenced blocks arrive wrapped in a <pre> from react-markdown; this
        // styles that box. Inline code is a bare <code> (no <pre>), so it must
        // not be promoted to a block.
        pre: ({ children }) => (
          <pre className="my-2 overflow-x-auto bg-base p-2 font-mono text-xs">
            {children}
          </pre>
        ),
        code: ({ children, className }) => {
          // rehype-highlight tags fenced blocks with a language/hljs class;
          // fall back to a newline check for language-less fences. Inline code
          // gets a subtle chip; block code is left to the <pre> above.
          const isBlock =
            !!className ||
            (typeof children === "string" && children.includes("\n"));
          if (isBlock) {
            const lang = className?.replace("language-", "");
            return (
              <code data-lang={lang} className={className}>
                {children}
              </code>
            );
          }
          return (
            <code className="rounded bg-base px-1 py-0.5 font-mono text-[0.9em]">
              {children}
            </code>
          );
        },
        table: ({ children }) => (
          <div className="my-2 overflow-x-auto">
            <table className="min-w-full border-collapse text-xs">
              {children}
            </table>
          </div>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
