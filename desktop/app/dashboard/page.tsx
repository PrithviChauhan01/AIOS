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
];
// NOTE: habits is deliberately NOT in this list — it has its own section below, which
// shows the same rows as a per-habit view (today / streak / last 7 days) instead of a
// flat one-row-per-day dump.

const STAT_CARDS = [
  { key: "jobs_applied", label: "Jobs Applied" },
  { key: "leads", label: "Leads" },
  { key: "study_sessions_this_week", label: "Study (7d)" },
  { key: "habit_streak", label: "Habit Streak" },
  { key: "pending_reminders", label: "Pending Reminders" },
];

type Row = Record<string, unknown>;

// A panel's data is in exactly ONE of three states, and they are not the same thing:
// null = still loading, {error} = the fetch FAILED, [] = it succeeded and the table is
// genuinely empty. Collapsing the last two into "Nothing here yet" is what made a dead
// backend look like an empty database.
type Fetched<T> = T[] | null | { error: string };
type ResourceState = Fetched<Row>;
type OverviewState = Record<string, number> | null | { error: string };

function isError(state: unknown): state is { error: string } {
  return typeof state === "object" && state !== null && "error" in state;
}

/** The loading / error / empty body for a panel, or null when there are rows to render.
 *  Error is the only state that offers an action, because it's the only one the user
 *  can do anything about. */
function panelState<T>(state: Fetched<T>, emptyText: string, onRetry: () => void) {
  if (state === null) return <div className="empty">Loading…</div>;
  if (isError(state))
    return (
      <div className="empty flex flex-col items-center gap-2.5">
        <span className="text-red-400">Couldn&apos;t load — {state.error}</span>
        <button
          type="button"
          onClick={onRetry}
          className="pill-btn hover:text-accent hover:border-border-hi transition cursor-pointer"
        >
          Retry
        </button>
      </div>
    );
  if (state.length === 0) return <div className="empty">{emptyText}</div>;
  return null;
}

interface ConversationSession {
  session_id: string;
  preview: string | null;
  last_timestamp: string;
  turn_count: number;
  archived: boolean;
}

interface ConversationTurn {
  role: string;
  content: string;
  mood: string | null;
  timestamp: string;
}

interface Reminder {
  id: number;
  title: string;
  due_at: string;
  repeat: string | null;
  done: number;
  created_at: string;
}

interface HabitRow {
  id: number;
  name: string;
  logged_at: string;
  done: number;
}

type SessionsState = Fetched<ConversationSession>;
type TranscriptState = Fetched<ConversationTurn>;
type RemindersState = Fetched<Reminder>;
type HabitsState = Fetched<HabitRow>;

function Table({
  columns,
  rows,
  onRetry,
}: {
  columns: Column[];
  rows: ResourceState;
  onRetry: () => void;
}) {
  const status = panelState(rows, "Nothing here yet.", onRetry);
  if (status || !Array.isArray(rows)) return status;
  return (
    <div className="overflow-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className="text-left px-3.5 py-2.5 text-[9px] font-mono uppercase tracking-[0.14em] text-fg-dim font-medium border-b border-border whitespace-nowrap"
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row.id ?? i)} className="hover:bg-glass-2 transition-colors">
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

function ConversationsSection({
  authedFetch,
  reloadTick,
}: {
  authedFetch: (path: string) => Promise<unknown>;
  reloadTick: number;
}) {
  const [sessions, setSessions] = useState<SessionsState>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptState>(null);
  const [expanded, setExpanded] = useState(false);

  const load = useCallback(() => {
    setSessions(null);
    setExpanded(false);
    authedFetch("/conversations/sessions")
      .then((rows) => setSessions(rows as ConversationSession[]))
      .catch((e) => setSessions({ error: (e as Error).message }));
  }, [authedFetch]);

  useEffect(() => {
    load();
  }, [load, reloadTick]);

  const loadTranscript = useCallback(() => {
    if (!selected) return;
    setTranscript(null);
    authedFetch(`/conversations?session_id=${encodeURIComponent(selected)}`)
      .then((rows) => setTranscript(rows as ConversationTurn[]))
      .catch((e) => setTranscript({ error: (e as Error).message }));
  }, [authedFetch, selected]);

  useEffect(() => {
    loadTranscript();
  }, [loadTranscript]);

  return (
    <section className="mb-8">
      <h3 className="font-mono text-[10px] tracking-[0.16em] uppercase text-fg-dim font-medium mb-3.5">
        Conversations
      </h3>
      <div className="glass-card px-4 overflow-hidden">
        {!Array.isArray(sessions) || sessions.length === 0 ? (
          panelState(sessions, "Nothing here yet.", load)
        ) : (
          <>
            <div className={expanded ? "max-h-[400px] overflow-y-auto no-scrollbar" : ""}>
              {(expanded ? sessions : sessions.slice(0, 3)).map((s) => (
                <button
                  key={s.session_id}
                  type="button"
                  onClick={() => setSelected(s.session_id === selected ? null : s.session_id)}
                  className={`convo-row w-full text-left transition ${
                    selected === s.session_id ? "bg-glass-2" : ""
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="flex items-baseline gap-2">
                      <span className="text-[12.5px] text-fg">{s.session_id}</span>
                      {s.archived && <span className="glass-pill">Archived</span>}
                    </span>
                    <span className="font-mono text-[9px] text-fg-dim whitespace-nowrap">
                      {s.turn_count} turns · {s.last_timestamp}
                    </span>
                  </div>
                  <div className="font-mono text-[9px] text-fg-dim mt-1 truncate">
                    {s.preview || "—"}
                  </div>
                </button>
              ))}
            </div>
            {sessions.length > 3 && (
              <button
                type="button"
                onClick={() => setExpanded((e) => !e)}
                className="w-full text-center py-2.5 font-mono text-[9px] uppercase tracking-[0.14em] text-fg-dim hover:text-fg transition cursor-pointer border-t border-border"
              >
                {expanded ? "Show less" : `View all (${sessions.length})`}
              </button>
            )}
          </>
        )}
      </div>

      {selected && (
        <div className="glass-card overflow-hidden mt-3 p-4 max-h-[480px] overflow-y-auto no-scrollbar space-y-3">
          {!Array.isArray(transcript) || transcript.length === 0 ? (
            panelState(transcript, "No turns in this session.", loadTranscript)
          ) : (
            transcript.map((t, i) => (
              <div
                key={i}
                className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={t.role === "user" ? "bubble bubble-user max-w-xl" : "bubble bubble-aios max-w-xl"}>
                  {t.role !== "user" && (
                    <span className="text-[10px] text-fg-dim mr-2 uppercase tracking-widest">
                      {t.role}
                    </span>
                  )}
                  {t.content}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
}

function RemindersSection({
  authedFetch,
  authedPost,
  reloadTick,
  onMutate,
}: {
  authedFetch: (path: string) => Promise<unknown>;
  authedPost: (path: string) => Promise<unknown>;
  reloadTick: number;
  onMutate: () => void;
}) {
  const [items, setItems] = useState<RemindersState>(null);
  const [showDone, setShowDone] = useState(false);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // NOTE: this deliberately does not clear actionError — a failed action calls
  // onMutate, which re-runs this load, and clearing here would wipe the message
  // in the same tick it was set. It clears when the next action starts instead.
  const load = useCallback(() => {
    setItems(null);
    authedFetch("/dashboard/reminders")
      .then((rows) => setItems(rows as Reminder[]))
      .catch((e) => setItems({ error: (e as Error).message }));
  }, [authedFetch]);

  useEffect(() => {
    load();
  }, [load, reloadTick]);

  // Both actions go to the backend routes that delegate to the reminders TOOL — the
  // panel never touches the table itself. onMutate re-runs every panel so the
  // "Pending Reminders" stat card can't drift out of step with this list.
  const act = async (id: number, verb: "complete" | "delete") => {
    setBusyId(id);
    setActionError(null);
    try {
      const res = (await authedPost(`/dashboard/reminders/${id}/${verb}`)) as { ok?: boolean };
      if (!res?.ok) setActionError(`Couldn't ${verb} #${id} — it may already be gone.`);
      onMutate();
    } catch (e) {
      setActionError(`Couldn't ${verb} #${id} — ${(e as Error).message}`);
    } finally {
      setBusyId(null);
    }
  };

  const pending = Array.isArray(items) ? items.filter((r) => !r.done) : [];
  const done = Array.isArray(items) ? items.filter((r) => r.done) : [];
  const visible = showDone ? [...pending, ...done] : pending;

  return (
    <section className="mb-8">
      <h3 className="font-mono text-[10px] tracking-[0.16em] uppercase text-fg-dim font-medium mb-3.5">
        Reminders
      </h3>
      <div className="glass-card overflow-hidden">
        {!Array.isArray(items) || items.length === 0 ? (
          panelState(items, "Nothing here yet.", load)
        ) : (
          <>
            {actionError && (
              <div className="px-3.5 py-2.5 text-[11px] text-red-400 border-b border-border">
                {actionError}
              </div>
            )}
            <div className="overflow-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr>
                    {["Title", "Due", "Repeat", "Status", ""].map((label, i) => (
                      <th
                        key={i}
                        className="text-left px-3.5 py-2.5 text-[9px] font-mono uppercase tracking-[0.14em] text-fg-dim font-medium border-b border-border whitespace-nowrap"
                      >
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visible.map((r) => (
                    <tr
                      key={r.id}
                      className={`hover:bg-glass-2 transition-colors ${r.done ? "opacity-45" : ""}`}
                    >
                      <td className="px-3.5 py-2.5 border-b border-border last:border-b-0">
                        {r.title}
                      </td>
                      <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                        {r.due_at}
                      </td>
                      <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                        {r.repeat || "—"}
                      </td>
                      <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                        <span className={`glass-pill ${r.done ? "" : "ok"}`}>
                          {r.done ? "Done" : "Pending"}
                        </span>
                      </td>
                      <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap text-right">
                        <span className="inline-flex gap-3.5">
                          {!r.done && (
                            <button
                              type="button"
                              disabled={busyId === r.id}
                              onClick={() => act(r.id, "complete")}
                              className="font-mono text-[9px] uppercase tracking-[0.14em] text-fg-dim hover:text-fg transition cursor-pointer disabled:opacity-40 disabled:cursor-default"
                            >
                              Complete
                            </button>
                          )}
                          <button
                            type="button"
                            disabled={busyId === r.id}
                            onClick={() => act(r.id, "delete")}
                            className="font-mono text-[9px] uppercase tracking-[0.14em] text-fg-dim hover:text-red-400 transition cursor-pointer disabled:opacity-40 disabled:cursor-default"
                          >
                            Delete
                          </button>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {pending.length === 0 && !showDone && (
              <div className="empty">Nothing pending.</div>
            )}
            {done.length > 0 && (
              <button
                type="button"
                onClick={() => setShowDone((v) => !v)}
                className="w-full text-center py-2.5 font-mono text-[9px] uppercase tracking-[0.14em] text-fg-dim hover:text-fg transition cursor-pointer border-t border-border"
              >
                {showDone ? "Hide completed" : `Show completed (${done.length})`}
              </button>
            )}
          </>
        )}
      </div>
    </section>
  );
}

// ── Habit day grid ──
// habits rows are one per habit per day (db/sqlite_init.py), so the per-habit view is
// built here from the same read endpoint rather than adding a shaped one. Local date
// parts, never toISOString: logged_at is written from the backend's LOCAL clock
// (tools/habits.py), and a UTC conversion would shift the whole row by a day.
const HABIT_WINDOW = 7;

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate()
  ).padStart(2, "0")}`;
}

function lastDays(n: number): string[] {
  const today = new Date();
  return Array.from({ length: n }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() - (n - 1 - i));
    return dayKey(d);
  });
}

interface HabitView {
  name: string;
  today: "done" | "missed" | "unlogged";
  streak: number;
  week: ("done" | "missed" | "unlogged")[];
}

/** Group raw rows into one view per habit. `streak` counts back from today — or from
 *  yesterday when today isn't logged yet, matching core/dashboard.py:_habit_streak so
 *  the panel and the stat card can never disagree. */
function toHabitViews(rows: HabitRow[]): HabitView[] {
  const byName = new Map<string, Map<string, boolean>>();
  for (const r of rows) {
    const day = String(r.logged_at).slice(0, 10);
    if (!byName.has(r.name)) byName.set(r.name, new Map());
    byName.get(r.name)!.set(day, Boolean(r.done));
  }

  const week = lastDays(HABIT_WINDOW);
  const today = dayKey(new Date());

  return [...byName.entries()]
    .map(([name, days]) => {
      const state = (day: string) =>
        !days.has(day) ? "unlogged" : days.get(day) ? "done" : "missed";

      const cursor = new Date();
      if (days.get(today) !== true) cursor.setDate(cursor.getDate() - 1);
      let streak = 0;
      while (days.get(dayKey(cursor)) === true) {
        streak += 1;
        cursor.setDate(cursor.getDate() - 1);
      }

      return {
        name,
        today: state(today) as HabitView["today"],
        streak,
        week: week.map(state) as HabitView["week"],
      };
    })
    .sort((a, b) => b.streak - a.streak || a.name.localeCompare(b.name));
}

const DOT_CLASS: Record<HabitView["today"], string> = {
  done: "bg-fg",
  missed: "bg-fg-dim",
  unlogged: "border border-border",
};

function HabitsSection({
  authedFetch,
  reloadTick,
}: {
  authedFetch: (path: string) => Promise<unknown>;
  reloadTick: number;
}) {
  const [rows, setRows] = useState<HabitsState>(null);

  const load = useCallback(() => {
    setRows(null);
    authedFetch("/dashboard/habits")
      .then((r) => setRows(r as HabitRow[]))
      .catch((e) => setRows({ error: (e as Error).message }));
  }, [authedFetch]);

  useEffect(() => {
    load();
  }, [load, reloadTick]);

  const views = Array.isArray(rows) ? toHabitViews(rows) : [];
  const week = lastDays(HABIT_WINDOW);

  return (
    <section className="mb-8">
      <h3 className="font-mono text-[10px] tracking-[0.16em] uppercase text-fg-dim font-medium mb-3.5">
        Habits
      </h3>
      <div className="glass-card overflow-hidden">
        {!Array.isArray(rows) || views.length === 0 ? (
          panelState(rows, "Nothing here yet.", load)
        ) : (
          <div className="overflow-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr>
                  {["Habit", "Today", "Streak", `Last ${HABIT_WINDOW} days`].map((label) => (
                    <th
                      key={label}
                      className="text-left px-3.5 py-2.5 text-[9px] font-mono uppercase tracking-[0.14em] text-fg-dim font-medium border-b border-border whitespace-nowrap"
                    >
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {views.map((h) => (
                  <tr key={h.name} className="hover:bg-glass-2 transition-colors">
                    <td className="px-3.5 py-2.5 border-b border-border last:border-b-0">
                      {h.name}
                    </td>
                    <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                      <span className={`glass-pill ${h.today === "done" ? "ok" : ""}`}>
                        {h.today === "done" ? "Done" : h.today === "missed" ? "Not done" : "Not logged"}
                      </span>
                    </td>
                    <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                      {h.streak > 0 ? `${h.streak} day${h.streak === 1 ? "" : "s"}` : "—"}
                    </td>
                    <td className="px-3.5 py-2.5 border-b border-border whitespace-nowrap">
                      <span className="inline-flex items-center gap-1.5">
                        {h.week.map((state, i) => (
                          <span
                            key={week[i]}
                            title={`${week[i]} — ${
                              state === "done" ? "done" : state === "missed" ? "not done" : "not logged"
                            }`}
                            className={`w-2 h-2 rounded-full ${DOT_CLASS[state]}`}
                          />
                        ))}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
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

  const authedPost = useCallback(
    async (path: string) => {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },
    [token]
  );

  // Each loader is its own callback so a failed panel can retry ITSELF, without
  // re-fetching (and re-flashing) every other panel on the page.
  const loadOverview = useCallback(() => {
    setOverview(null);
    authedFetch("/dashboard/overview")
      .then((o) => setOverview(o as Record<string, number>))
      .catch((e) => setOverview({ error: (e as Error).message }));
  }, [authedFetch]);

  const loadResource = useCallback(
    (r: Resource) => {
      setData((prev) => ({ ...prev, [r.key]: null }));
      authedFetch(r.path)
        .then((rows) => setData((prev) => ({ ...prev, [r.key]: rows as Row[] })))
        .catch((e) => setData((prev) => ({ ...prev, [r.key]: { error: (e as Error).message } })));
    },
    [authedFetch]
  );

  useEffect(() => {
    if (!token) return;
    loadOverview();
    for (const r of RESOURCES) loadResource(r);
  }, [token, reloadTick, loadOverview, loadResource]);

  return (
    <main className="flex-1 min-h-0 overflow-y-auto no-scrollbar bg-bg text-fg">
      <div className="max-w-[1100px] mx-auto px-6 py-8">
        <div className="flex items-baseline justify-between gap-4 border-b border-border pb-4 mb-7 flex-wrap">
          <h1 className="font-mono text-[10px] tracking-[0.3em] uppercase text-fg-dim">Overview</h1>
          {token && (
            <button
              type="button"
              onClick={() => setReloadTick((t) => t + 1)}
              className="pill-btn hover:text-accent hover:border-border-hi transition cursor-pointer"
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
              className="bg-glass border border-border rounded-full px-3 py-1.5 text-fg text-xs w-56 outline-none"
            />
            <button
              type="button"
              onClick={() => setToken(tokenInput.trim())}
              className="pill-btn hover:text-accent hover:border-border-hi transition cursor-pointer"
            >
              Use token
            </button>
          </div>
        )}

        {token && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-3">
              {STAT_CARDS.map((s) => (
                <div key={s.key} className="glass-card p-4">
                  <div className="font-mono text-[9px] tracking-[0.14em] uppercase text-fg-dim">{s.label}</div>
                  <div className="text-[30px] font-medium mt-2 tracking-[-0.02em] text-fg">
                    {overview === null
                      ? "…"
                      : isError(overview)
                        ? "—"
                        : overview[s.key]}
                  </div>
                </div>
              ))}
            </div>
            {/* A dash in every card means one of two very different things — say which. */}
            <div className="mb-9">
              {isError(overview) && (
                <div className="flex items-center gap-3 text-[11px] text-red-400">
                  <span>Couldn&apos;t load the counts — {overview.error}</span>
                  <button
                    type="button"
                    onClick={loadOverview}
                    className="font-mono text-[9px] uppercase tracking-[0.14em] text-fg-dim hover:text-fg transition cursor-pointer"
                  >
                    Retry
                  </button>
                </div>
              )}
            </div>

            <ConversationsSection authedFetch={authedFetch} reloadTick={reloadTick} />

            <RemindersSection
              authedFetch={authedFetch}
              authedPost={authedPost}
              reloadTick={reloadTick}
              onMutate={() => setReloadTick((t) => t + 1)}
            />

            <HabitsSection authedFetch={authedFetch} reloadTick={reloadTick} />

            {RESOURCES.map((r) => (
              <section key={r.key} className="mb-8">
                <h3 className="font-mono text-[10px] tracking-[0.16em] uppercase text-fg-dim font-medium mb-3.5">
                  {r.title}
                </h3>
                <div className="glass-card overflow-hidden">
                  <Table
                    columns={r.columns}
                    rows={data[r.key] ?? null}
                    onRetry={() => loadResource(r)}
                  />
                </div>
              </section>
            ))}
          </>
        )}
      </div>
    </main>
  );
}
