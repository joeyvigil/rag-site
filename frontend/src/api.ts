export type User = { id: string; email: string; created_at: string };

export type DocStatus = "pending" | "processing" | "ready" | "failed";
export type Doc = {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocStatus;
  error: string | null;
  num_chunks: number;
  num_pages: number | null;
  created_at: string;
};

export type Source = {
  n: number;
  document_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  score: number;
  snippet: string;
};

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type Timings = {
  retrieval_ms: number;
  first_token_ms: number | null;
  total_ms: number;
  model: string;
};

// ---------------------------------------------------------------- token

const TOKEN_KEY = "rag_token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable; session just won't persist */
  }
}

// ---------------------------------------------------------------- http

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");

  const res = await fetch(`/api${path}`, { ...init, headers });
  if (res.status === 401 && token) {
    setToken(null);
    window.dispatchEvent(new Event("auth-expired"));
  }
  if (!res.ok) throw new Error(await errorMessage(res));
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  register: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<User>("/auth/me"),
  listDocuments: () => request<Doc[]>("/documents"),
  uploadDocuments: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return request<Doc[]>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  reprocessDocument: (id: string) => request<Doc>(`/documents/${id}/reprocess`, { method: "POST" }),
};

// ---------------------------------------------------------------- chat (SSE)

type ChatHandlers = {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (timings: Timings) => void;
  onError: (message: string) => void;
};

/**
 * POST /api/chat and parse the Server-Sent Events stream by hand.
 * (The browser's EventSource only supports GET without custom headers.)
 */
export async function streamChat(
  body: { question: string; history: ChatMessage[] },
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch("/api/chat", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError(await errorMessage(res));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      dispatch(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
    }
  }
}

function dispatch(raw: string, h: ChatHandlers) {
  let event = "message";
  const data: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "sources") h.onSources(payload);
  else if (event === "token") h.onToken(payload.text);
  else if (event === "done") h.onDone(payload);
  else if (event === "error") h.onError(payload.message);
}
