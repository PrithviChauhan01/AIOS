"use client";

import { useCallback, useEffect, useState } from "react";
import { useToken } from "@/lib/token-context";
import { API_BASE } from "@/lib/api";

interface Column {
  key: string;
  label: string;
  fmt?: (v: unknown) => string;
}

interface Resource {
  key: string;
  title: string;
  path: string;
  columns: Column[];
}

const RESOURCES: Resource[] = [
  {
    key: "jobs", title: "Jobs", path: "/dashboard/jobs", columns: [
      { key: "company", label: "Company" },
      { key: "role", label: "Role" },
      { key: "resume_used", label: "Resume", fmt: (v) => (v as string) || "—" },
      { key: "status", label: "Status" },
      { key: "applied_at", label: "Applied" },
      { key: "follow_up_at", label: "Follow-up", fmt: (v) => (v as string) || "—" },
    ],
  },
  {
    key: "leads", title: "Leads", path: "/dashboard/leads", columns: [
      { key: "studio_name", label: "Studio" },
      { key: "location", label: "Location", fmt: (v) => (v as string) || "—" },
      { key: "outreach_sent", label: "Outreach", fmt: (v) => (v ? "✓" : "—") },
      { key: "response", label: "Response", fmt: (v) => (v as string) || "—" },
      { key: "created_at", label: "Created" },
    ],
  },
  {
    key: "fitness", title: "Fitness", path: "/dashboard/fitness", columns: [
      { key: "activity", label: "Activity" },
      { key: "duration_min", label: "Duration (min)", fmt: (v) => (v == null ? "—" : String(v)) },
      { key: "notes", label: "Notes", fmt: (v) => (v as string) || "—" },
      { key: "logged_at", label: "Logged" },
    ],
  },
  {
    key: "study", title: "Study", path: "/dashboard/study", columns: [
      { key: "topic", label: "Topic" },
      { key: "duration_min", label: "Duration (min)", fmt: (v) => (v == null ? "—" : String(v)) },
      { key: "notes", label: "Notes", fmt: (v) => (v as string) || "—" },
      { key: "logged_at", label: "Logged" },
    ],
  },
  {
    key: "habits", title: "Habits", path: "/dashboard/habits", columns: [
      { key: "name", label: "Habit" },
      { key: "logged_at", label: "Date" },
      { key: "done", label: "Done", fmt: (v) => (v ? "✓" : "—") },
    ],
  },
];

const STAT_CARDS = [
  { key: "jobs_applied", label: "Jobs Applied" },
  { key: "leads", label: "Leads" },
  { key: "study_sessions_this_week", label: "Study (7d)" },
  { key: "habit_streak", label: "Habit Streak" },
  { key: "pending_reminders", label: "Pending Reminders" },
];

type Row = Record<string, unknown>;
type ResourceState = Row[] | null | { error: string };
type OverviewState = Record<string, number> | null | { error: string };

function Table({ columns, rows }: { columns: Column[]; rows: ResourceState }) {
  if (rows === null) return <div className="p-6 text-fg-dim text-sm">Loading…</div>;
  if (!Array.isArray(rows)) return <div className="p-6 text-red-500 text-sm">Error: {rows.error}</div>;
  if (rows.length === 0) return <div className="p-6 text-fg-dim text-sm">Nothing here yet.</div>;
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className="text-left px-3.5 py-2.5 text-[11px] uppercase tracking-wide text-fg-dim font-medium border-b border-border whitespace-nowrap"
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row.id ?? i)} className="hover:bg-fg/5">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className="px-3.5 py-2.5 border-b border-border last:border-b-0 whitespace-nowrap"
                >
                  {c.fmt ? c.fmt(row[c.key]) : String(row[c.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DashboardPage() {
  const { token, setToken } = useToken();
  const [tokenInput, setTokenInput] = useState("");
  const [overview, setOverview] = useState<OverviewState>(null);
  const [data, setData] = useState<Record<string, ResourceState>>({});
  const [reloadTick, setReloadTick] = useState(0);

  const authedFetch = useCallback(
    async (path: string) => {
      const res = await fetch(`${API_BASE}${path}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    [token]
  );

  useEffect(() => {
    if (!token) return;

    setOverview(null);
    authedFetch("/dashboard/overview")
      .then(setOverview)
      .catch((e) => setOverview({ error: e.message }));

    for (const r of RESOURCES) {
      setData((prev) => ({ ...prev, [r.key]: null }));
      authedFetch(r.path)
        .then((rows) => setData((prev) => ({ ...prev, [r.key]: rows })))
        .catch((e) => setData((prev) => ({ ...prev, [r.key]: { error: e.message } })));
    }
  }, [token, reloadTick, authedFetch]);

  return (
    <main className="flex-1 bg-bg text-fg">
      <div className="max-w-[1100px] mx-auto px-6 py-8">
        <div className="flex items-baseline justify-between gap-4 border-b border-border pb-4 mb-7 flex-wrap">
          <h1 className="text-sm font-semibold tracking-wide text-fg-dim uppercase">Overview</h1>
          {token && (
            <button
              type="button"
              onClick={() => setReloadTick((t) => t + 1)}
              className="text-xs border border-border rounded px-2.5 py-1.5 text-fg-dim hover:text-accent hover:border-accent transition"
            >
              Refresh
            </button>
          )}
        </div>

        {!token && (
          <div className="flex items-center gap-2 text-xs text-fg-dim mb-6">
            <span>Fetching a token automatically — or paste one:</span>
            <input
              placeholder="paste JWT token"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              className="bg-panel border border-border rounded px-2 py-1.5 text-fg text-xs w-56"
            />
            <button
              type="button"
              onClick={() => setToken(tokenInput.trim())}
              className="border border-border rounded px-2.5 py-1.5 hover:border-accent hover:text-accent transition"
            >
              Use token
            </button>
          </div>
        )}

        {token && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-9">
              {STAT_CARDS.map((s) => (
                <div key={s.key} className="bg-panel border border-border rounded-md p-4">
                  <div className="text-[11px] uppercase tracking-wide text-fg-dim mb-2">{s.label}</div>
                  <div className="text-2xl font-semibold text-accent">
                    {overview === null
                      ? "…"
                      : "error" in overview
                        ? "—"
                        : overview[s.key]}
                  </div>
                </div>
              ))}
            </div>

            {RESOURCES.map((r) => (
              <section key={r.key} className="mb-8">
                <h2 className="text-[13px] uppercase tracking-wide text-fg-dim font-semibold mb-2.5">
                  {r.title}
                </h2>
                <div className="bg-panel border border-border rounded-md overflow-hidden">
                  <Table columns={r.columns} rows={data[r.key] ?? null} />
                </div>
              </section>
            ))}
          </>
        )}
      </div>
    </main>
  );
}
