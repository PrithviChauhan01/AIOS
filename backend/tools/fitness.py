"""Fitness log — persists PRs/workouts Sir reports conversationally.

A WRITE tool like reminders/jobs: it persists to the existing `fitness_logs`
table (id, activity, duration_min, notes, logged_at). Before this tool existed,
a reported PR only ever lived in the chat transcript + cognition's in-context
reasoning — the FitnessTeacher (agents/fitness.py) is a pure research/advice
teacher and never wrote state, so the dashboard's Fitness view stayed
permanently empty. This tool is what closes that gap, through the same
gate -> extract -> execute -> real-result action path reminders/jobs use.

`weight`/`reps`/`sets` have no dedicated columns on fitness_logs — the existing
schema (and the dashboard table reading it) only has activity/duration_min/
notes/logged_at, matching study_sessions' shape. Rather than a migration, the
lift's structured detail is packed into `notes` as a readable string
("225 lbs x 5 reps"), and `activity` holds the lift name — so a logged PR
renders correctly in the CURRENT dashboard table with no frontend change.
"""

import os
import sqlite3
from datetime import datetime, timedelta

from config import Config
from tools.base import Tool
# Same stored datetime format reminders/jobs use, so ORDER BY logged_at sorts
# correctly and the dashboard's raw string renders consistently.
from tools.reminders import _DT_FMT, parse_minutes


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_logged_at(date_phrase: str) -> datetime:
    """Best-effort PAST-oriented date resolution for a PR/workout report.
    Deliberately narrow — a PR is always today or in the recent past, so this
    is NOT reminders.parse_due (which biases toward the FUTURE for scheduling
    and would misdate 'yesterday' as next week). Only the phrasings Sir
    actually says are handled; anything else defaults to now rather than
    guessing wrong."""
    now = datetime.now()
    t = (date_phrase or "").strip().lower()
    if not t or t in ("today", "this morning", "just now", "now"):
        return now
    if t == "yesterday":
        return now - timedelta(days=1)
    return now


def _format_detail(weight, reps, sets) -> str | None:
    """'225 lbs x 5 reps x 3 sets' from whichever of weight/reps/sets were
    actually given. None if nothing usable came through."""
    parts = []
    weight = (weight or "").strip() if isinstance(weight, str) else weight
    if weight:
        parts.append(str(weight))
    if reps:
        try:
            n = int(reps)
            parts.append(f"{n} rep" + ("" if n == 1 else "s"))
        except (TypeError, ValueError):
            pass
    if sets:
        try:
            n = int(sets)
            parts.append(f"{n} set" + ("" if n == 1 else "s"))
        except (TypeError, ValueError):
            pass
    return " x ".join(parts) if parts else None


# ── Action API (writes state) ──
def log_fitness(lift: str, weight=None, reps=None, sets=None, date: str = None,
                duration=None) -> dict:
    """Record ONE PR/workout result in the `fitness_logs` table.

    Validates before writing: a lift name is required, plus at least one real
    detail — weight/reps/sets, or a duration for work that is timed rather than
    loaded ("ran 30 minutes"). A bare "logged a workout" with no number at all is
    not a usable row. Returns either:
      success  → {"ok": True, "id", "activity", "notes", "duration_min", "logged_at"}
      clarify  → {"ok": False, "needs": "lift"|"detail", "ask": "..."}
    A clarifying ask is returned (not raised) so the caller can put it to Sir
    rather than store an empty row."""
    lift = (lift or "").strip()
    if not lift:
        return {"ok": False, "needs": "lift", "ask": "Which lift was that, Sir?"}

    notes = _format_detail(weight, reps, sets)
    duration_min = parse_minutes(duration)
    if not notes and not duration_min:
        return {"ok": False, "needs": "detail",
                "ask": f"What did you hit on {lift}, Sir — weight or reps?"}

    logged_str = _resolve_logged_at(date).strftime(_DT_FMT)

    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO fitness_logs (activity, duration_min, notes, logged_at) "
            "VALUES (?, ?, ?, ?)",
            (lift, duration_min, notes, logged_str),
        )
        conn.commit()
        fid = cur.lastrowid
        print(f"[fitness] WRITE db={_db_path()} id={fid} activity={lift!r} "
              f"notes={notes!r} duration_min={duration_min}")
    finally:
        conn.close()
    return {"ok": True, "id": fid, "activity": lift, "notes": notes,
            "duration_min": duration_min, "logged_at": logged_str}


def log_fitness_batch(exercises: list, date: str = None) -> dict:
    """Record EVERY exercise reported in one message — "benched 225x5 and squatted
    315x3" is two rows, not one. Each entry goes through log_fitness above, so
    validation and the row shape are identical to the single case; this only owns
    the fan-out and the honest per-entry outcome.

    Returns {"ok": bool, "items": [row, ...], "count": N, "failed": [ask, ...]}
    where ok is True if AT LEAST ONE row landed. `failed` carries the clarifying
    asks for entries that were too thin to store, so a partial success is
    reported as exactly that — never as a clean sweep."""
    if not exercises:
        return {"ok": False, "needs": "lift", "ask": "Which lift was that, Sir?"}

    items, failed = [], []
    for ex in exercises:
        if not isinstance(ex, dict):
            continue
        res = log_fitness(ex.get("lift"), ex.get("weight"), ex.get("reps"),
                          ex.get("sets"), ex.get("date") or date, ex.get("duration"))
        (items if res.get("ok") else failed).append(res if res.get("ok") else res.get("ask"))

    if not items:
        # Nothing usable at all — relay the first clarify so Sir gets one clean ask.
        return {"ok": False, "needs": "detail", "count": 0, "failed": failed,
                "ask": failed[0] if failed else "Which lift was that, Sir?"}
    return {"ok": True, "items": items, "count": len(items), "failed": failed}


def list_fitness(limit: int = None) -> list:
    """Return logged PRs/workouts, most recent first."""
    conn = _conn()
    try:
        sql = "SELECT id, activity, duration_min, notes, logged_at FROM fitness_logs ORDER BY logged_at DESC"
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Pool wrapper ──
class FitnessTool(Tool):
    """Shared-pool handle for the fitness action tool. is_action=True marks it
    as state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): it lists recent PRs/workouts, fail-soft. The write action is exposed
    as log() which delegates to the module function above."""

    name = "fitness"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_fitness(limit=20)
            results = [
                f"#{r['id']} {r['activity']} — {r['notes'] or 'no detail'} ({r['logged_at']})"
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:fitness] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def log(self, lift: str, weight=None, reps=None, sets=None, date: str = None,
            duration=None) -> dict:
        return log_fitness(lift, weight, reps, sets, date, duration)

    def log_many(self, exercises: list, date: str = None) -> dict:
        return log_fitness_batch(exercises, date)

    def list(self, limit: int = None) -> list:
        return list_fitness(limit)
