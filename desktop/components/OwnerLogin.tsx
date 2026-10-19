"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";
import { useToken } from "@/lib/token-context";

// Password + 6-digit authenticator code → POST /api/token. Errors are shown as
// the backend words them (generic by design — never which factor was wrong).
export default function OwnerLogin({ onClose }: { onClose: () => void }) {
  const { login } = useToken();
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const passwordRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    passwordRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy || !password || code.length !== 6) return;
    setBusy(true);
    setError("");
    const res = await login(password, code);
    setBusy(false);
    if (res.ok) {
      onClose();
    } else {
      setError(res.error);
      setCode(""); // a TOTP code is single-use either way
    }
  };

  // Portalled to <body>: the nav's backdrop-filter makes it the containing block
  // for position:fixed, which would trap the overlay inside the nav bar.
  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm px-4"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <form
        onSubmit={submit}
        role="dialog"
        aria-modal="true"
        aria-labelledby="owner-login-title"
        className="glass-card w-full max-w-[320px] p-6 flex flex-col gap-3.5"
        style={{ background: "var(--bg)" }}
      >
        <h2 id="owner-login-title" className="font-mono text-[10px] tracking-[0.3em] uppercase text-fg-dim">
          Owner login
        </h2>
        <input
          ref={passwordRef}
          type="password"
          autoComplete="current-password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="bg-glass border border-border rounded-full px-4 py-2 text-fg text-[13px] outline-none focus:border-border-hi"
        />
        <input
          type="text"
          inputMode="numeric"
          autoComplete="one-time-code"
          pattern="\d{6}"
          maxLength={6}
          placeholder="6-digit code"
          value={code}
          onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
          className="bg-glass border border-border rounded-full px-4 py-2 text-fg text-[13px] tracking-[0.3em] outline-none focus:border-border-hi"
        />
        {error && (
          <p role="alert" className="text-[11px] text-red-400 px-1">
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} className="pill-btn hover:text-fg transition cursor-pointer">
            Cancel
          </button>
          <button
            type="submit"
            disabled={busy || !password || code.length !== 6}
            className="pill-btn hover:text-accent hover:border-border-hi transition cursor-pointer disabled:opacity-40 disabled:cursor-default"
          >
            {busy ? "…" : "Log in"}
          </button>
        </div>
      </form>
    </div>,
    document.body
  );
}
