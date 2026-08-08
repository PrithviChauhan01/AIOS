"""Study log — persists the study sessions Sir reports conversationally.

A WRITE tool like reminders/jobs/fitness: it persists to the existing
`study_sessions` table (id, topic, duration_min, notes, logged_at). Before this
tool existed, "did two hours of DP today" was answered by the StudyTeacher
(agents/study.py) — a pure research/teaching agent that never writes state — so
the dashboard's Study view stayed permanently empty. This closes that gap through
the same gate -> extract -> execute -> real-result action path reminders use.

Column mapping: the spoken SUBJECT is the row's `topic` (the schema's name for
it); the specific topics covered and any free notes are folded into `notes`,
since the existing schema has no separate column for them and no migration is
wanted here.
"""

import os
import sqlite3

from config import Config
from tools.base import Tool
# Same stored datetime format + past-oriented date resolution the fitness log uses,
# so ORDER BY logged_at sorts correctly across both logs and 'yesterday' means
# yesterday (never next week — parse_due is future-biased and is NOT used here).
from tools.reminders import _DT_FMT, parse_minutes
from tools.fitness import _resolve_logged_at


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _format_notes(topics, notes) -> str | None:
    """'binary search, DP — felt slow on the recursion' from whichever of topics/
    notes was actually given. None when neither was."""
    parts = [str(p).strip() for p in (topics, notes) if p and str(p).strip()]
    return " — ".join(parts) or None


# ── Action API (writes state) ──
def log_study(subject: str, duration=None, topics: str = None, notes: str = None,
              date: str = None) -> dict:
    """Record a study session in the `study_sessions` table.

    Validates before writing: a subject is required (an empty row is not a
    session). `duration` may be a spoken phrase ('2 hours', '90 mins') or a
    number of minutes; it is optional — a session with topics but no stated
    length is still a real session.

    Returns either:
      success  → {"ok": True, "id", "topic", "duration_min", "notes", "logged_at"}
      clarify  → {"ok": False, "needs": "subject", "ask": "..."}
    A clarifying ask is returned (not raised) so the caller can put it to Sir."""
    subject = (subject or "").strip()
    if not subject:
        return {"ok": False, "needs": "subject", "ask": "What were you studying, Sir?"}

    duration_min = parse_minutes(duration)
    note_text = _format_notes(topics, notes)
    logged_str = _resolve_logged_at(date).strftime(_DT_FMT)

    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO study_sessions (topic, duration_min, notes, logged_at) "
            "VALUES (?, ?, ?, ?)",
            (subject, duration_min, note_text, logged_str),
        )
        conn.commit()
        sid = cur.lastrowid
        print(f"[study] WRITE db={_db_path()} id={sid} topic={subject!r} "
              f"duration_min={duration_min}")
    finally:
        conn.close()
    return {"ok": True, "id": sid, "topic": subject, "duration_min": duration_min,
            "notes": note_text, "logged_at": logged_str}


def list_study(limit: int = None) -> list:
    """Return logged study sessions, most recent first."""
    conn = _conn()
    try:
        sql = ("SELECT id, topic, duration_min, notes, logged_at FROM study_sessions "
               "ORDER BY logged_at DESC, id DESC")
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Pool wrapper ──
class StudyTool(Tool):
    """Shared-pool handle for the study action tool. is_action=True marks it as
    state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): it lists recent sessions, fail-soft. The write action is exposed as
    log() which delegates to the module function above."""

    name = "study"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_study(limit=20)
            results = [
                f"#{r['id']} {r['topic']}"
                + (f" — {r['duration_min']} min" if r.get("duration_min") else "")
                + f" ({r['logged_at']})"
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:study] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def log(self, subject: str, duration=None, topics: str = None,
            notes: str = None, date: str = None) -> dict:
        return log_study(subject, duration, topics, notes, date)

    def list(self, limit: int = None) -> list:
        return list_study(limit)
