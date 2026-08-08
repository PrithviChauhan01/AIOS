"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useToken } from "./token-context";
import { API_BASE } from "./api";

const SESSION_KEY = "aios_session_id";

interface SessionContextValue {
  sessionId: string;
  newChat: () => void;
}

const SessionContext = createContext<SessionContextValue | undefined>(undefined);

// Chat turns are keyed by session_id on the backend ("main" until now, since
// the chat page hardcoded it). This context owns which session is "current" —
// New Chat archives the outgoing session server-side (POST /conversations/archive,
// see core/memory.py:archive_session) then mints a fresh id, so the previous
// session's turns stay in SQLite untouched and only show up under the
// dashboard's Conversations list (tagged archived), never mixed into the
// live chat again.
export function SessionProvider({ children }: { children: ReactNode }) {
  const { token } = useToken();
  const [sessionId, setSessionId] = useState("main");
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  useEffect(() => {
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) setSessionId(stored);
  }, []);

  const newChat = useCallback(() => {
    const outgoing = sessionIdRef.current;
    const id = `session_${Date.now()}`;
    localStorage.setItem(SESSION_KEY, id);
    setSessionId(id);

    if (token) {
      fetch(`${API_BASE}/conversations/archive`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ session_id: outgoing }),
      }).catch(() => {}); // best-effort tag — history is never deleted either way
    }
  }, [token]);

  return (
    <SessionContext.Provider value={{ sessionId, newChat }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
}
