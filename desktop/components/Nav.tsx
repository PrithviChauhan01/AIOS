"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "@/lib/theme-context";
import { useSession } from "@/lib/session-context";

const LINKS = [
  { href: "/", label: "Chat" },
  { href: "/dashboard", label: "Dashboard" },
];

export default function Nav() {
  const pathname = usePathname();
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();
  const { newChat } = useSession();

  const activeIndex = Math.max(
    0,
    LINKS.findIndex((l) => l.href === pathname)
  );

  const handleNewChat = () => {
    newChat();
    if (pathname !== "/") router.push("/");
  };

  return (
    <nav
      className="flex items-center justify-between shrink-0 px-6 py-4 border-b border-border backdrop-blur-2xl"
      style={{ background: "rgba(255,255,255,0.025)" }}
    >
      <div className="flex items-center gap-2.5">
        <span className="brand-dot" />
        <span className="text-sm font-semibold tracking-[0.14em] text-fg">AIOS</span>
      </div>

      <div className="nav-pill" data-active={activeIndex}>
        <span className="nav-pill-indicator" />
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <Link key={l.href} href={l.href} className={`nav-link ${active ? "on" : ""}`}>
              {l.label}
            </Link>
          );
        })}
      </div>

      <div className="flex items-center gap-2.5">
        {/* Always mounted (never conditionally unmounted) so its width is
            reserved on every route — <nav> is `justify-content: between`,
            so unmounting this would change the right-hand group's width and
            shift the centered nav-pill sideways whenever the route changes.
            `invisible` keeps the box in flow with zero visual/interactive
            presence, which is what "no New Chat button on Dashboard" means. */}
        <button
          type="button"
          onClick={handleNewChat}
          tabIndex={pathname === "/" ? 0 : -1}
          aria-hidden={pathname !== "/"}
          title="Start a new chat — the current one archives into the dashboard's Conversations list"
          className={`pill-btn hover:text-accent hover:border-border-hi transition cursor-pointer ${
            pathname === "/" ? "" : "invisible pointer-events-none"
          }`}
        >
          New Chat
        </button>

        <button
          type="button"
          onClick={toggleTheme}
          aria-pressed={theme === "light"}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          className="icon-btn"
        >
          {theme === "dark" ? (
            // sun (click to go light)
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
          ) : (
            // moon (click to go dark)
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
      </div>
    </nav>
  );
}
