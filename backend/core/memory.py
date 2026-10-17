import sqlite3
import threading
from typing import Optional
from config import Config
from db.sqlite_init import ensure_conversation_schema

# ── Secret-tier redaction ──
# A secret turn was being written to `conversations.content` VERBATIM. The vault held
# a correctly encrypted copy, and traces/trace_stages were clean, but the plaintext
# still sat in this table — a real password among them.
#
# Two things were leaking, not one:
#   AT REST  — the raw value in a plaintext SQLite column, readable by anything that
#              opens the file, and served by GET /conversations.
#   IN FLIGHT— get_history() feeds this column straight into the prompt. A secret
#              turn is answered locally, but the NEXT public turn in the same session
#              gets the full 20-message window on a CLOUD book. So a redacted-at-rest
#              row is also the thing that stops the secret being replayed to Groq an
#              hour later. Redacting here fixes both, because both read this column.
#
# What is redacted, precisely:
#   * role='user' AND tier='secret' — the message Sir typed, which is the only place
#     the raw value ever appears.
#   * NOT the assistant reply. On this path it is the acknowledge-and-hold line
#     ("Noted, Sir. It's in the vault.") — it contains no value by construction, and
#     blanking it would leave the transcript reading as though she never answered.
#   * NOT private tier. Private is a cloud tier by design; redacting it would gut the
#     history of ordinary personal conversation for no privacy gain.
#
# The placeholder is what history loads and what she sees. She does not need the raw
# value — the acknowledge-and-hold path already answers this turn without it.
SECRET_PLACEHOLDER = "[secret handover - vaulted]"

_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn) -> None:
    """Add the redaction columns once per process. Without them the INSERT below
    raises and a secret turn fails outright — so this runs before the first write
    rather than trusting that init_sqlite() ran in this process."""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if not _schema_ready:
            ensure_conversation_schema(conn)
            conn.commit()
            _schema_ready = True


def save_message(session_id: str, role: str, content: str, mode: str = "default",
                 tier: str = None, vault_id: str = None):
    """Persist one turn. `tier` is the turn's sensitivity as cognition resolved it —
    'secret' redacts a user message to SECRET_PLACEHOLDER and records where the real
    content went. Callers that pass no tier (the greeting path, which runs before
    triage and cannot be secret) are unaffected."""
    if Config.PUBLIC_MODE:
        return  # public visitors' turns live only in core.public_session, never on disk
    redact = tier == "secret" and role == "user"
    stored = SECRET_PLACEHOLDER if redact else content
    conn = sqlite3.connect(Config.SQLITE_PATH)
    try:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, mode, redacted, "
            "vault_id) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, stored, mode, int(redact),
             vault_id if redact else None)
        )
        conn.commit()
    finally:
        conn.close()
    if redact:
        # Never the value, never its length — just that the swap happened.
        print(f"[memory] secret turn redacted at rest "
              f"(vault_id={'set' if vault_id else 'none'})")

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
            "SELECT role, content, mode, timestamp, redacted FROM conversations "
            "WHERE session_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        rows.reverse()
    else:
        cursor = conn.execute(
            "SELECT role, content, mode, timestamp, redacted FROM conversations "
            "WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
    conn.close()
    # `redacted` is surfaced so a reader can tell a placeholder from Sir literally
    # having typed one. `content` is already the placeholder for those rows — the raw
    # value is not fetched here and could not be, it is not in this table.
    return [
        {
            "role": role,
            "content": content,
            "mood": mode if mode in _KNOWN_MOODS else None,
            "timestamp": timestamp,
            "redacted": bool(redacted),
        }
        for role, content, mode, timestamp, redacted in rows
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