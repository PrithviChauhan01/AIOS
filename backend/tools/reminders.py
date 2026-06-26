"""Reminders — AIOS's first ACTION tool. Unlike read-only tools (wikipedia),
this one WRITES state: it persists to the existing `reminders` table
(id, title, due_at, repeat, done, created_at).

Scope of this slice: storage + retrieval + natural-language time parsing only.
There is NO firing/scheduler here — nothing reads due_at and acts on it yet;
that's the next sub-slice. set/list/complete/delete + parse_due, nothing more.

The public API is the four module-level functions plus parse_due. RemindersTool
wraps them so the tool lives in the shared pool (registry) like any other tool,
and is tagged is_action=True so actions can later route through a confirm gate.
"""

import re
import sqlite3
from datetime import datetime, timedelta

from config import Config
from tools.base import Tool

# dateparser handles relative offsets and date+time combos well, but it isn't on
# the critical path: if the wheel is missing on this Python, we fall back to the
# hand-rolled parser below, which covers the phrases AIOS actually uses.
try:
    import dateparser
except Exception:  # pragma: no cover - absence is the whole point of the guard
    dateparser = None

# Stored format: lexicographic order == chronological order, so ORDER BY due_at
# is a correct time sort, and it matches SQLite's DATETIME convention.
_DT_FMT = "%Y-%m-%d %H:%M:%S"

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Fuzzy parts of day → a concrete clock time. Applied before any date parsing so
# "tomorrow morning" becomes "tomorrow 9am" for both parsers.
_PARTS = {
    "morning": "9am",
    "afternoon": "3pm",
    "evening": "6pm",
    "tonight": "8pm",
    "night": "9pm",
    "noon": "12pm",
    "midnight": "12am",
}

# Default time when a date is given but no clock time (e.g. "tomorrow").
_DEFAULT_HOUR = 9

# Junk-title guard. A reminder title must name a concrete task. LLM-driven flows
# sometimes hand set_reminder a *definition* of a deadline ("the target to
# achieve by a specific date") or a bare placeholder ("reminder", "todo")
# instead of a real task — that's exactly the phantom row this rejects.
_PLACEHOLDER_TITLES = {
    "reminder", "a reminder", "set a reminder", "set reminder", "your reminder",
    "untitled", "none", "n/a", "na", "tbd", "todo", "to do", "task", "something",
    "target", "the target", "goal", "deadline",
}
_PLACEHOLDER_PATTERNS = (
    "target to achieve",
    "specific date",
    "achieve by a",
)


def _is_placeholder_title(title) -> bool:
    """True if the title is empty or a generic non-task placeholder/definition
    rather than a real thing to be reminded about."""
    if not isinstance(title, str):
        return True
    t = title.strip().lower()
    if len(t) < 2:                      # empty or single-char → not a task
        return True
    if t in _PLACEHOLDER_TITLES:
        return True
    return any(p in t for p in _PLACEHOLDER_PATTERNS)


def _due_is_valid(due_at) -> bool:
    """True if due_at is a non-empty string in the stored datetime format."""
    if not isinstance(due_at, str) or not due_at.strip():
        return False
    try:
        datetime.strptime(due_at.strip(), _DT_FMT)
        return True
    except ValueError:
        return False


# ── Connection ──
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Natural-language time parsing ──
def _normalize(text: str) -> str:
    t = text.strip().lower()
    for word, clock in _PARTS.items():
        t = re.sub(rf"\b{word}\b", clock, t)
    return t


def _parse_clock(text: str):
    """Pull a clock time out of text → (hour, minute) or None.
    Handles '6pm', '9:30am', 'at 6 pm', and bare '9' (treated as 24h hour)."""
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    mer = m.group(3)
    if mer == "pm" and hour != 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _at(date, hour, minute):
    return datetime(date.year, date.month, date.day, hour, minute, 0)


def _fallback_parse(text: str, now: datetime):
    """Deterministic parser for the phrases AIOS uses. Used as the primary path
    (predictable for known phrasings) and as the sole parser when dateparser is
    absent. Returns a datetime or None."""
    t = text

    # "in N minutes/hours/days/weeks"
    m = re.search(r"\bin\s+(\d+)\s+(minute|min|hour|hr|day|week)s?\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {
            "minute": timedelta(minutes=n), "min": timedelta(minutes=n),
            "hour": timedelta(hours=n), "hr": timedelta(hours=n),
            "day": timedelta(days=n), "week": timedelta(weeks=n),
        }[unit]
        return (now + delta).replace(second=0, microsecond=0)

    clock = _parse_clock(t)

    # weekday, optionally prefixed with "next" (force a full week ahead)
    for name, idx in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t):
            force_next = bool(re.search(rf"\bnext\s+{name}\b", t))
            ahead = (idx - now.weekday()) % 7
            if force_next and ahead == 0:
                ahead = 7
            date = (now + timedelta(days=ahead)).date()
            hour, minute = clock or (_DEFAULT_HOUR, 0)
            dt = _at(date, hour, minute)
            # same-day weekday with a time already passed → next week
            if ahead == 0 and dt <= now:
                dt += timedelta(days=7)
            return dt

    # tomorrow / today
    if re.search(r"\btomorrow\b", t):
        date = (now + timedelta(days=1)).date()
        hour, minute = clock or (_DEFAULT_HOUR, 0)
        return _at(date, hour, minute)
    if re.search(r"\btoday\b", t):
        hour, minute = clock or (_DEFAULT_HOUR, 0)
        return _at(now.date(), hour, minute)

    # bare time only → today, rolling to tomorrow if already past
    if clock:
        dt = _at(now.date(), *clock)
        if dt <= now:
            dt += timedelta(days=1)
        return dt

    return None


def parse_due(text, now: datetime = None) -> datetime:
    """Parse a natural-language time into a concrete datetime, anchored to `now`
    (defaults to datetime.now()). Accepts an existing datetime untouched.
    Raises ValueError if nothing parseable is found — a reminder with no time
    is useless, so the caller must handle it rather than store garbage."""
    if isinstance(text, datetime):
        return text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("no due time given")

    now = now or datetime.now()
    norm = _normalize(text)

    dt = _fallback_parse(norm, now)
    if dt is None and dateparser is not None:
        dt = dateparser.parse(norm, settings={
            "RELATIVE_BASE": now,
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        })
    if dt is None:
        raise ValueError(f"could not parse a time from: {text!r}")
    return dt.replace(microsecond=0)


# ── Action API (writes state) ──
def set_reminder(title: str, due_at, repeat: str = None) -> dict:
    """Create a reminder. `due_at` may be a datetime or a natural-language string
    ('at 6pm', 'in 30 minutes', 'tomorrow morning', 'next Monday 9am').

    Validates before writing — junk in is the phantom-row bug. Returns either:
      success  → {"ok": True, "id", "title", "due_at", "repeat", "done"}
      clarify  → {"ok": False, "needs": "title"|"due_at", "ask": "..."}
    A clarifying ask is returned (not raised) so the caller can put it to Sir
    rather than store a placeholder."""
    # Title must name a real task — reject empty/placeholder before touching the DB.
    if _is_placeholder_title(title):
        return {"ok": False, "needs": "title", "ask": "What should I remind you about, Sir?"}

    # Time must be concrete. parse_due raises on anything vague; turn that into a
    # clarifying ask rather than a stored or thrown error.
    try:
        when = parse_due(due_at)
    except ValueError:
        return {"ok": False, "needs": "due_at", "ask": "When should I set that for, Sir?"}

    due_str = when.strftime(_DT_FMT)
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO reminders (title, due_at, repeat) VALUES (?, ?, ?)",
            (title.strip(), due_str, repeat),
        )
        conn.commit()
        rid = cur.lastrowid
    finally:
        conn.close()
    return {"ok": True, "id": rid, "title": title.strip(), "due_at": due_str,
            "repeat": repeat, "done": 0}


def list_reminders(pending_only: bool = True) -> list:
    """Return reminders sorted by due_at ascending (soonest first).
    pending_only=True excludes ones already marked done."""
    conn = _conn()
    try:
        sql = "SELECT id, title, due_at, repeat, done, created_at FROM reminders"
        if pending_only:
            sql += " WHERE done = 0"
        sql += " ORDER BY due_at ASC"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def complete_reminder(id: int) -> bool:
    """Mark a reminder done. Returns True if a row was updated."""
    conn = _conn()
    try:
        cur = conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_reminder(id: int) -> bool:
    """Remove a reminder. Returns True if a row was deleted."""
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM reminders WHERE id = ?", (id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def cleanup_reminders() -> list:
    """One-off hygiene pass: delete rows that were never valid reminders — null/
    empty/malformed due_at, or empty/placeholder titles (the phantom row).
    Returns the deleted rows so the caller can report what was purged.
    Safe to re-run; clean rows are left untouched."""
    conn = _conn()
    deleted = []
    try:
        rows = conn.execute("SELECT id, title, due_at FROM reminders").fetchall()
        for r in rows:
            if _is_placeholder_title(r["title"]) or not _due_is_valid(r["due_at"]):
                conn.execute("DELETE FROM reminders WHERE id = ?", (r["id"],))
                deleted.append(dict(r))
        conn.commit()
    finally:
        conn.close()
    return deleted


# Run directly for the one-off purge: `python -m tools.reminders`
if __name__ == "__main__":
    purged = cleanup_reminders()
    if purged:
        print(f"[reminders] purged {len(purged)} junk row(s):")
        for r in purged:
            print(f"  - #{r['id']} title={r['title']!r} due_at={r['due_at']!r}")
    else:
        print("[reminders] nothing to clean — list is already clean")


# ── Pool wrapper ──
class RemindersTool(Tool):
    """Shared-pool handle for the reminders action tool. is_action=True marks it
    as state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): it lists pending reminders, fail-soft. The write actions are exposed as
    methods that delegate to the module functions above."""

    name = "reminders"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_reminders(pending_only=True)
            results = [
                f"#{r['id']} {r['title']} — due {r['due_at']}"
                + (f" (repeats {r['repeat']})" if r["repeat"] else "")
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:reminders] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def set(self, title: str, due_at, repeat: str = None) -> dict:
        return set_reminder(title, due_at, repeat)

    def list(self, pending_only: bool = True) -> list:
        return list_reminders(pending_only)

    def complete(self, id: int) -> bool:
        return complete_reminder(id)

    def delete(self, id: int) -> bool:
        return delete_reminder(id)
