import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// Production: a static export (next build → out/), served from the same origin as
// the API by the reverse proxy, which owns the /api routing:
//   /api/public/*  → public-mode backend   (prefix stripped: /api/public/chat → /chat)
//   /api/*         → personal backend      (prefix stripped: /api/token → /token)
//
// Dev: rewrites mirror that proxy so the app's relative paths work under `next dev`.
// Rewrites are unsupported with output: "export", so each phase gets only its half.
// Order matters — the more specific /api/public rule must come first.
const PERSONAL_API = process.env.AIOS_PERSONAL_API ?? "http://127.0.0.1:5000";
const PUBLIC_API = process.env.AIOS_PUBLIC_API ?? "http://127.0.0.1:5001";

export default function config(phase: string): NextConfig {
  const base: NextConfig = {
    // Disable the Next.js dev tools indicator (the "N" badge / circle)
    devIndicators: false,
  };

  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      ...base,
      async rewrites() {
        return [
          { source: "/api/public/:path*", destination: `${PUBLIC_API}/:path*` },
          { source: "/api/:path*", destination: `${PERSONAL_API}/:path*` },
        ];
      },
    };
  }

  return { ...base, output: "export" };
}
