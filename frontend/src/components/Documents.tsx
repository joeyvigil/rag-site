import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { api, type Doc } from "../api";

const ACCEPT = ".pdf,.docx,.txt,.md,.markdown,.csv,.json,.html,.htm,.png,.jpg,.jpeg,.webp";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function Documents() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.listDocuments());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while anything is still being ingested.
  const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [busy, refresh]);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setError(null);
    setUploading(true);
    try {
      await api.uploadDocuments(files);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    upload(Array.from(e.dataTransfer.files));
  };

  const remove = async (doc: Doc) => {
    if (!confirm(`Delete ${doc.filename}?`)) return;
    try {
      await api.deleteDocument(doc.id);
      setDocs((ds) => ds.filter((d) => d.id !== doc.id));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const retry = async (doc: Doc) => {
    try {
      await api.reprocessDocument(doc.id);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="docs">
      <h2>Documents</h2>
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {uploading ? "Uploading…" : "Drop files or click to upload"}
        <div className="muted small">PDF (incl. scanned), DOCX, TXT, MD, CSV, JSON, HTML, images</div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => upload(Array.from(e.target.files ?? []))}
        />
      </div>
      {error && <div className="error">{error}</div>}

      {docs.length === 0 ? (
        <p className="muted small">No documents yet.</p>
      ) : (
        <ul className="doc-list">
          {docs.map((d) => (
            <li key={d.id} className="doc">
              <div className="doc-main">
                <span className="doc-name" title={d.filename}>
                  {d.filename}
                </span>
                <span className={`badge ${d.status}`}>{d.status}</span>
              </div>
              <div className="muted small">
                {formatSize(d.size_bytes)}
                {d.num_pages ? ` · ${d.num_pages} pages` : ""}
                {d.status === "ready" ? ` · ${d.num_chunks} chunks` : ""}
              </div>
              {d.error && <div className="error small">{d.error}</div>}
              <div className="doc-actions">
                {d.status === "failed" && (
                  <button className="ghost small" onClick={() => retry(d)}>
                    Retry
                  </button>
                )}
                <button className="ghost small" onClick={() => remove(d)}>
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
