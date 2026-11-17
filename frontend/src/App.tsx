import { useEffect, useState } from "react";
import { api, getToken, setToken, type User } from "./api";
import AuthForm from "./components/AuthForm";
import Chat from "./components/Chat";
import Documents from "./components/Documents";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(Boolean(getToken()));

  useEffect(() => {
    if (getToken()) {
      api
        .me()
        .then(setUser)
        .catch(() => setToken(null))
        .finally(() => setChecking(false));
    }
    const onExpired = () => setUser(null);
    window.addEventListener("auth-expired", onExpired);
    return () => window.removeEventListener("auth-expired", onExpired);
  }, []);

  const logout = () => {
    setToken(null);
    setUser(null);
  };

  if (checking) return <div className="center muted">Loading…</div>;
  if (!user) return <AuthForm onAuthed={setUser} />;

  return (
    <div className="app">
      <header className="topbar">
        <strong>Doc Q&amp;A</strong>
        <div className="topbar-right">
          <span className="muted">{user.email}</span>
          <button className="ghost" onClick={logout}>
            Log out
          </button>
        </div>
      </header>
      <div className="layout">
        <aside className="sidebar">
          <Documents />
        </aside>
        <main className="main">
          <Chat />
        </main>
      </div>
    </div>
  );
}
