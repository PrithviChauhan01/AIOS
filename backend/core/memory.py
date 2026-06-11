import sqlite3
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