"""Habits — the daily habit log.

A WRITE tool like reminders/jobs/fitness: it persists to the existing `habits`
table (id, name, logged_at DATE, done). Before this tool existed "mark meditation
done" was answered conversationally and nothing was written, so the dashboard's
Habits view (and any future streak count) stayed permanently empty. This closes
that gap through the same gate -> extract -> execute -> real-result action path
reminders use.

The table is one row PER HABIT PER DAY, so marking is an upsert on
(name, logged_at): marking the same habit twice in a day updates that day's row
instead of stacking duplicates, which is what makes "did I do X today" a single
honest lookup rather than a count.
"""

import os
import sqlite3

from config import Config
from tools.base import Tool
# Past-oriented date resolution shared with the fitness log ('yesterday' means
# yesterday). habits.logged_at is a DATE, so only the day part is stored.
from tools.fitness import _resolve_logged_at

_DAY_FMT = "%Y-%m-%d"


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _day(date_phrase: str = None) -> str:
    """The day a mark/check applies to, as stored ('YYYY-MM-DD'). Defaults today."""
    return _resolve_logged_at(date_phrase).strftime(_DAY_FMT)


def _find(conn, name: str, day: str):
    """That habit's row for that day, or None. Name match is case-insensitive so
    'Meditation' and 'meditation' are the same habit."""
    return conn.execute(
        "SELECT id, name, logged_at, done FROM habits "
        "WHERE lower(name) = lower(?) AND logged_at = ?",
        (name, day),
    ).fetchone()


# ── Action API (writes state) ──
def mark_habit(name: str, done: bool = True, date: str = None) -> dict:
    """Mark a habit done (or explicitly not done) for a day. Upserts that day's
    row, so marking twice never creates a second row.

    Returns either:
      success  → {"ok": True, "id", "name", "done", "logged_at", "already": bool}
      clarify  → {"ok": False, "needs": "name", "ask": "..."}
    `already` is True when that day's row was ALREADY in this state — cognition
    needs it to say "already logged" instead of claiming a fresh write."""
    name = (name or "").strip()
    if not name:
        return {"ok": False, "needs": "name", "ask": "Which habit, Sir?"}

    day = _day(date)
    flag = 1 if done else 0
    conn = _conn()
    try:
        row = _find(conn, name, day)
        if row is not None:
            already = bool(row["done"]) == bool(done)
            if not already:
                conn.execute("UPDATE habits SET done = ? WHERE id = ?", (flag, row["id"]))
                conn.commit()
            hid, stored_name = row["id"], row["name"]
        else:
            already = False
            cur = conn.execute(
                "INSERT INTO habits (name, logged_at, done) VALUES (?, ?, ?)",
                (name, day, flag),
            )
            conn.commit()
            hid, stored_name = cur.lastrowid, name
        print(f"[habits] WRITE db={_db_path()} id={hid} name={stored_name!r} "
              f"day={day} done={flag} already={already}")
    finally:
        conn.close()
    return {"ok": True, "id": hid, "name": stored_name, "done": bool(done),
            "logged_at": day, "already": already}


def check_habit(name: str, date: str = None) -> dict:
    """Answer "did I do X today". Reads only — never writes a row just to answer.

    Returns {"ok": True, "name", "done", "logged_at", "found"} where found=False
    means there is no row at all for that day (never logged), which is different
    from a row saying done=0."""
    name = (name or "").strip()
    if not name:
        return {"ok": False, "needs": "name", "ask": "Which habit, Sir?"}

    day = _day(date)
    conn = _conn()
    try:
        row = _find(conn, name, day)
    finally:
        conn.close()
    return {"ok": True, "name": (row["name"] if row is not None else name),
            "done": bool(row["done"]) if row is not None else False,
            "logged_at": day, "found": row is not None}


def list_habits(date: str = None, limit: int = None) -> list:
    """Habit rows for a day (today by default), most recently added first."""
    day = _day(date)
    conn = _conn()
    try:
        sql = ("SELECT id, name, logged_at, done FROM habits WHERE logged_at = ? "
               "ORDER BY id DESC")
        params = (day,)
        if limit:
            sql += " LIMIT ?"
            params = (day, limit)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Pool wrapper ──
class HabitsTool(Tool):
    """Shared-pool handle for the habits action tool. is_action=True marks it as
    state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): today's habit rows, fail-soft. The write/read actions delegate to the
    module functions above, which stay the source of truth."""

    name = "habits"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_habits()
            results = [
                f"#{r['id']} {r['name']} — {'done' if r['done'] else 'not done'} ({r['logged_at']})"
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:habits] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def mark(self, name: str, done: bool = True, date: str = None) -> dict:
        return mark_habit(name, done, date)

    def check(self, name: str, date: str = None) -> dict:
        return check_habit(name, date)

    def list(self, date: str = None, limit: int = None) -> list:
        return list_habits(date, limit)
