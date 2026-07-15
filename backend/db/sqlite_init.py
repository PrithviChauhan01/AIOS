import sqlite3
import os
from config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    mode        TEXT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    started_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at        DATETIME,
    mode            TEXT DEFAULT 'default',
    vision_active   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    due_at      DATETIME NOT NULL,
    repeat      TEXT,
    done        INTEGER DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company         TEXT NOT NULL,
    role            TEXT NOT NULL,
    resume_used     TEXT,
    status          TEXT DEFAULT 'applied',
    applied_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    follow_up_at    DATETIME,
    notes           TEXT,
    url             TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    doc_type      TEXT,
    tags          TEXT,
    source_path   TEXT,
    tier          TEXT DEFAULT 'private',
    storage       TEXT NOT NULL,            -- 'chroma' (embedded) | 'vault' (encrypted, local-only)
    vault_id      TEXT,                     -- set when storage='vault'
    chunk_count   INTEGER DEFAULT 0,
    char_count    INTEGER DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    studio_name     TEXT NOT NULL,
    location        TEXT,
    contact         TEXT,
    research        TEXT,
    outreach_sent   INTEGER DEFAULT 0,
    response        TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    follow_up_at    DATETIME
);

CREATE TABLE IF NOT EXISTS habits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    logged_at   DATE DEFAULT CURRENT_DATE,
    done        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fitness_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    activity        TEXT NOT NULL,
    duration_min    INTEGER,
    notes           TEXT,
    logged_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS study_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT NOT NULL,
    duration_min    INTEGER,
    notes           TEXT,
    logged_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS onboarding (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question_asked  TEXT,
    answer          TEXT,
    asked_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    week            INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS vault (
    id          TEXT PRIMARY KEY,
    content     BLOB NOT NULL,
    meta        TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# ── Routing outcomes (Slice A — passive outcome log) ──
# One row per completed teacher task: which domain, which book(s), whether the
# self-check passed, how many attempts, tokens, and wall-clock latency. Write-only for
# now — a later slice reads it to drive experience-based routing. Kept as a standalone
# constant so the writer (core.outcomes) can defensively ensure the table exists without
# re-running the whole schema, while this file stays the single source of truth.
ROUTING_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS routing_outcomes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 DATETIME DEFAULT CURRENT_TIMESTAMP,
    domain             TEXT,
    book_used          TEXT,
    ensemble           INTEGER DEFAULT 0,   -- bool: two books ran in parallel
    self_check_passed  INTEGER DEFAULT 0,   -- bool
    attempts           INTEGER DEFAULT 1,
    tokens             INTEGER DEFAULT 0,
    latency_ms         INTEGER DEFAULT 0,
    complexity         TEXT,
    session_id         TEXT
);
"""

SCHEMA += ROUTING_OUTCOMES_DDL

def init_sqlite():
    os.makedirs(os.path.dirname(Config.SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print("[db] SQLite ready")