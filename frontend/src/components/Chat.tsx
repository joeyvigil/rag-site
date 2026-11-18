import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { streamChat, type AnswerMode, type ChatMessage, type Source, type Timings } from "../api";

type Turn = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timings?: Timings;
  error?: string;
  streaming?: boolean;
};

/** Render "[1]" style citations as small badges with the source as a tooltip. */
function WithCitations({ text, sources }: { text: string; sources?: Source[] }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (!m) return <span key={i}>{part}</span>;
        const src = sources?.find((s) => s.n === Number(m[1]));
        const title = src ? `${src.filename}${src.page ? `, p.${src.page}` : ""}` : undefined;
        return (
          <sup key={i} className="cite" title={title}>
            {m[1]}
          </sup>
        );
      })}
    </>
  );
}

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<AnswerMode>("rag");
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const updateLast = (fn: (t: Turn) => Turn) =>
    setTurns((ts) => [...ts.slice(0, -1), fn(ts[ts.length - 1])]);

  const send = async (e?: FormEvent) => {
    e?.preventDefault();
    const question = input.trim();
    if (!question || busy) return;

    const history: ChatMessage[] = turns
      .filter((t) => !t.error && t.content)
      .map((t) => ({ role: t.role, content: t.content }));

    setInput("");
    setBusy(true);
    setTurns((ts) => [
      ...ts,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        { question, history, answer_mode: mode },
        {
          onSources: (sources) => updateLast((t) => ({ ...t, sources })),
          onToken: (text) => updateLast((t) => ({ ...t, content: t.content + text })),
          onDone: (timings) => updateLast((t) => ({ ...t, timings })),
          onError: (error) => updateLast((t) => ({ ...t, error })),
        },
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") updateLast((t) => ({ ...t, error: (err as Error).message }));
    } finally {
      updateLast((t) => ({ ...t, streaming: false }));
      setBusy(false);
      abortRef.current = null;
    }
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat">
      <div className="messages">
        {turns.length === 0 && (
          <div className="empty muted">
            Upload documents on the left, wait for them to be <em>ready</em>, then ask a question.
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`msg ${t.role}`}>
            <div className="bubble">
              {t.role === "assistant" ? (
                <WithCitations text={t.content} sources={t.sources} />
              ) : (
                t.content
              )}
              {t.streaming && !t.content && <span className="muted">Thinking…</span>}
              {t.error && <div className="error">{t.error}</div>}
            </div>
            {t.sources && t.sources.length > 0 && !t.streaming && (
              <details className="sources">
                <summary>
                  {t.sources.length} sources
                  {t.timings && (
                    <span className="muted">
                      {" "}
                      · {t.timings.cached ? "cached answer" : (t.timings.mode ?? "rag")}
                      {t.timings.agent_steps ? ` (${t.timings.agent_steps} steps)` : ""} · prep {t.timings.retrieval_ms} ms ·
                      first token {t.timings.first_token_ms ?? "–"} ms · total {t.timings.total_ms} ms · {t.timings.model}
                    </span>
                  )}
                  {t.timings?.search_query && t.timings.search_query !== turns[i - 1]?.content && (
                    <div className="muted small">searched for: “{t.timings.search_query}”</div>
                  )}
                </summary>
                <ol>
                  {t.sources.map((s) => (
                    <li key={s.n}>
                      <div>
                        <strong>[{s.n}]</strong> {s.filename}
                        {s.page ? `, p.${s.page}` : ""} <span className="muted">score {s.score.toFixed(3)}</span>
                      </div>
                      <div className="snippet">{s.snippet}</div>
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={send}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about your documents…  (Enter to send, Shift+Enter for newline)"
          rows={2}
        />
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as AnswerMode)}
          disabled={busy}
          title="How the answer is found. RAG: one search (fast). Agent: the model searches and reads documents itself (most accurate, slower). Long context: whole documents in the prompt (no search, costly)."
        >
          <option value="rag">RAG</option>
          <option value="agent">Agent</option>
          <option value="long_context">Long context</option>
        </select>
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
