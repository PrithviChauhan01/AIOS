"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { loginFetch } from "./api";

// sessionStorage, never localStorage: the owner JWT dies with the tab.
const TOKEN_KEY = "aios_owner_token";
// Pre-P3 builds cached a 365-day token here. Swept on load so it can't linger.
const LEGACY_KEY = "aios_token";

type LoginResult = { ok: true } | { ok: false; error: string };

interface TokenContextValue {
  token: string; // "" → anonymous / public mode
  login: (password: string, code: string) => Promise<LoginResult>;
  logout: () => void;
}

const TokenContext = createContext<TokenContextValue | undefined>(undefined);

// Seconds-since-epoch expiry from the JWT payload, or 0 if unreadable. Only used
// to drop a stale token client-side — the backend remains the authority.
function expiryOf(token: string): number {
  try {
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return Number(JSON.parse(atob(payload)).exp) || 0;
  } catch {
    return 0;
  }
}

function readStored(): string {
  try {
    localStorage.removeItem(LEGACY_KEY);
    const t = sessionStorage.getItem(TOKEN_KEY) || "";
    if (t && expiryOf(t) * 1000 > Date.now()) return t;
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // storage blocked → stay anonymous
  }
  return "";
}

// Owns the owner JWT for the whole app. No token is ever fetched automatically:
// the app starts anonymous (public mode) and only an explicit login switches it
// to the personal backend.
export function TokenProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState("");

  useEffect(() => {
    setTokenState(readStored());
  }, []);

  const logout = useCallback(() => {
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {}
    setTokenState("");
  }, []);

  // Auto-logout at expiry (12h) so the UI never sits "logged in" on a dead token.
  useEffect(() => {
    if (!token) return;
    const ms = expiryOf(token) * 1000 - Date.now();
    const id = setTimeout(logout, Math.max(0, ms));
    return () => clearTimeout(id);
  }, [token, logout]);

  const login = useCallback(async (password: string, code: string): Promise<LoginResult> => {
    let res: Response;
    try {
      res = await loginFetch(password, code);
    } catch {
      return { ok: false, error: "Couldn't reach the server." };
    }
    if (res.ok) {
      const { token: t } = (await res.json()) as { token: string };
      try {
        sessionStorage.setItem(TOKEN_KEY, t);
      } catch {}
      setTokenState(t);
      return { ok: true };
    }
    // The backend's messages are deliberately generic; pass them through.
    const detail = await res
      .json()
      .then((d: { detail?: string }) => d.detail)
      .catch(() => undefined);
    return { ok: false, error: detail || "Login failed." };
  }, []);

  return (
    <TokenContext.Provider value={{ token, login, logout }}>
      {children}
    </TokenContext.Provider>
  );
}

export function useToken() {
  const ctx = useContext(TokenContext);
  if (!ctx) throw new Error("useToken must be used within TokenProvider");
  return ctx;
}
