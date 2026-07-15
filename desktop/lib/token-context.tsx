"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { API_BASE } from "./api";

const TOKEN_KEY = "aios_token";

interface TokenContextValue {
  token: string;
  setToken: (t: string) => void;
}

const TokenContext = createContext<TokenContextValue | undefined>(undefined);

// ONE place that owns the JWT lifecycle for the whole app: read a cached token
// from localStorage, or fetch a fresh one from GET /token (backend shape is
// {token: string} — every consumer reads .token, matching what chat already
// expected). Both the chat page and the dashboard consume this same context,
// so there is exactly one fetch, one cache, one shape.
export function TokenProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (stored) {
      setTokenState(stored);
      return;
    }
    let cancelled = false;
    fetch(`${API_BASE}/token`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled && d.token) {
          localStorage.setItem(TOKEN_KEY, d.token);
          setTokenState(d.token);
        }
      })
      .catch(() => {}); // manual entry (dashboard's paste-token field) still available
    return () => {
      cancelled = true;
    };
  }, []);

  const setToken = useCallback((t: string) => {
    localStorage.setItem(TOKEN_KEY, t);
    setTokenState(t);
  }, []);

  return (
    <TokenContext.Provider value={{ token, setToken }}>
      {children}
    </TokenContext.Provider>
  );
}

export function useToken() {
  const ctx = useContext(TokenContext);
  if (!ctx) throw new Error("useToken must be used within TokenProvider");
  return ctx;
}
