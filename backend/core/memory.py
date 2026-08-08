import sqlite3
from typing import Optional
from config import Config

def save_message(session_id: str, role: str, content: str, mode: str = "default"):
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.execute(
        "INSERT INTO conversations (session_id, role, content, mode) VALUES (?, ?, ?, ?)",
        (session_id, role, content, mode)
    )
    conn.commit()
    conn.close()

def get_history(session_id: str, limit: int = 20) -> list:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    cursor = conn.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    # Reverse so oldest is first
    rows.reverse()
    return [{"role": row[0], "content": row[1]} for row in rows]


# cognition.py never writes a real mood into `mode` (save_message's mode arg is
# never passed a mood — it stays the "default" column default), so there is no
# genuine historical mood to surface. Only pass through a value if it happens to
# be one of the frontend's known moods; otherwise report None so the UI's own
# "neutral" fallback applies instead of a bogus tint.
_KNOWN_MOODS = {"warm", "neutral", "sharp", "soft"}


def get_conversation(session_id: str, limit: Optional[int] = None) -> list:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    if limit:
        # Most recent `limit` turns, then flip back to chat order (oldest first) —
        # same DESC-then-reverse trick as get_history above.
        cursor = conn.execute(
            "SELECT role, content, mode, timestamp FROM conversations "
            "WHERE session_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        rows.reverse()
    else:
        cursor = conn.execute(
            "SELECT role, content, mode, timestamp FROM conversations "
            "WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
    conn.close()
    return [
        {
            "role": role,
            "content": content,
            "mood": mode if mode in _KNOWN_MOODS else None,
            "timestamp": timestamp,
        }
        for role, content, mode, timestamp in rows
    ]


def get_sessions() -> list:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    cursor = conn.execute(
        """
        SELECT c1.session_id,
               (SELECT content FROM conversations c2
                WHERE c2.session_id = c1.session_id AND c2.role = 'user'
                ORDER BY c2.timestamp ASC, c2.id ASC LIMIT 1) AS preview,
               MAX(c1.timestamp) AS last_timestamp,
               COUNT(*) AS turn_count,
               s.ended_at IS NOT NULL AS archived
        FROM conversations c1
        LEFT JOIN sessions s ON s.id = c1.session_id
        GROUP BY c1.session_id
        ORDER BY last_timestamp DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "session_id": session_id,
            "preview": preview,
            "last_timestamp": last_timestamp,
            "turn_count": turn_count,
            "archived": bool(archived),
        }
        for session_id, preview, last_timestamp, turn_count, archived in rows
    ]


# The `sessions` table (schema in db/sqlite_init.py) was defined but never
# written to by any code — `ended_at` already means exactly "this session is
# no longer live," so New Chat reuses it as the archive marker instead of
# adding a new column/table. Upsert because a session_id may not have a row
# here yet (conversations rows can exist with no matching sessions row).
def archive_session(session_id: str):
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.execute(
        "INSERT INTO sessions (id, ended_at) VALUES (?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(id) DO UPDATE SET ended_at = CURRENT_TIMESTAMP",
        (session_id,)
    )
    conn.commit()
    conn.close()