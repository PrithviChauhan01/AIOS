// Every backend call goes through one of these two helpers — never a raw fetch.
// Paths are RELATIVE: the static export is served from the same origin as the
// API, and the reverse proxy routes them (next.config.ts mirrors this in dev):
//   /api/public/*  → the public-mode backend   (anonymous)
//   /api/*         → the personal backend      (owner JWT required)

const PUBLIC_BASE = "/api/public";
const OWNER_BASE = "/api";

export function publicFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${PUBLIC_BASE}${path}`, init);
}

// The personal backend is only ever called with a token. Refusing here (rather
// than trusting each caller to check) is what keeps the public UI from touching
// /api/* at all — a logged-out page can't reach it by accident.
export function ownerFetch(path: string, token: string, init: RequestInit = {}): Promise<Response> {
  if (!token) return Promise.reject(new Error("owner login required"));
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${OWNER_BASE}${path}`, { ...init, headers });
}

// The one unauthenticated call to the personal backend: the login itself.
export function loginFetch(password: string, code: string): Promise<Response> {
  return fetch(`${OWNER_BASE}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password, code }),
  });
}
