"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "@/lib/theme-context";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function Nav() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
      <div className="flex items-center gap-6">
        <span className="text-xs tracking-widest text-fg-dim uppercase">
          <span className="text-accent">{">"} </span>AIOS
        </span>
        <div className="flex items-center gap-4">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`text-xs uppercase tracking-wide transition ${
                  active ? "text-fg" : "text-fg-dim hover:text-fg"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      </div>

      <button
        type="button"
        onClick={toggleTheme}
        aria-pressed={theme === "light"}
        title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        className="text-fg-dim hover:text-fg transition"
      >
        {theme === "dark" ? (
          // sun (click to go light)
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
          </svg>
        ) : (
          // moon (click to go dark)
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        )}
      </button>
    </nav>
  );
}
