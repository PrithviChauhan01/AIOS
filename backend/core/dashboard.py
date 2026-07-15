import sqlite3
import datetime

from config import Config


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(query: str, params: tuple = ()) -> list:
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def get_jobs() -> list:
    return _rows(
        "SELECT id, company, role, resume_used, status, applied_at, follow_up_at "
        "FROM jobs ORDER BY applied_at DESC"
    )


def get_leads() -> list:
    return _rows(
        "SELECT id, studio_name, location, outreach_sent, response, created_at "
        "FROM leads ORDER BY created_at DESC"
    )


def get_fitness_logs() -> list:
    return _rows(
        "SELECT id, activity, duration_min, notes, logged_at "
        "FROM fitness_logs ORDER BY logged_at DESC"
    )


def get_study_sessions() -> list:
    return _rows(
        "SELECT id, topic, duration_min, notes, logged_at "
        "FROM study_sessions ORDER BY logged_at DESC"
    )


def get_habits() -> list:
    # Raw per-day rows, most recent first — the caller groups by name if it wants
    # a per-habit day grid; nothing here assumes a shape the (currently empty,
    # unwritten) table hasn't earned yet.
    return _rows(
        "SELECT id, name, logged_at, done FROM habits ORDER BY name ASC, logged_at DESC"
    )


def _habit_streak() -> int:
    """Longest current consecutive-day streak (done=1, ending today or yesterday)
    across all habit names. 0 if the table is empty — no habit has ever been logged."""
    rows = _rows("SELECT name, logged_at, done FROM habits WHERE done = 1")
    if not rows:
        return 0
    by_name = {}
    for r in rows:
        try:
            d = datetime.date.fromisoformat(str(r["logged_at"])[:10])
        except ValueError:
            continue
        by_name.setdefault(r["name"], set()).add(d)

    today = datetime.date.today()
    best = 0
    for days in by_name.values():
        cursor = today
        if cursor not in days:
            cursor -= datetime.timedelta(days=1)  # allow "not logged yet today"
        streak = 0
        while cursor in days:
            streak += 1
            cursor -= datetime.timedelta(days=1)
        best = max(best, streak)
    return best


def get_overview() -> dict:
    conn = _conn()
    try:
        week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        jobs_applied = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        leads_count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        study_this_week = conn.execute(
            "SELECT COUNT(*) FROM study_sessions WHERE logged_at >= ?", (week_ago,)
        ).fetchone()[0]
        pending_reminders = conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE done = 0"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "jobs_applied": jobs_applied,
        "leads": leads_count,
        "study_sessions_this_week": study_this_week,
        "habit_streak": _habit_streak(),
        "pending_reminders": pending_reminders,
    }
