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

# ── Trace spine ──
# One `traces` row per request + N `trace_stages` rows, written in a SINGLE
# transaction from core.trace.Trace.flush() at the end of the turn. Purely
# observational: nothing in routing, triage, privacy or cognition reads these.
#
# METADATA ONLY. No message content, retrieved material, tool payload or vault data
# is ever written here — `meta` holds a whitelisted JSON object of labels/counts
# (core.trace.ALLOWED_META_KEYS), so a secret-tier turn stores the same SHAPE of
# metadata as a public one and none of its substance.
#
# Kept as a standalone constant, like ROUTING_OUTCOMES_DDL above, so the writer can
# defensively ensure the tables exist without re-running the whole schema while this
# file stays the single source of truth.
TRACE_DDL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id         TEXT PRIMARY KEY,
    session_id       TEXT,
    ts_start         TEXT,
    ts_end           TEXT,
    total_ms         INTEGER,
    sensitivity      TEXT,
    complexity       TEXT,
    domain           TEXT,
    teachers         TEXT,               -- comma-joined domains that actually ran
    deliverable      INTEGER,            -- bool, nullable: NULL = never determined
    fast_lane        INTEGER,            -- bool, nullable
    secret_mode      INTEGER,            -- bool, nullable
    final_book       TEXT,
    tokens_in        INTEGER,            -- provider-reported only; NULL if unreported
    tokens_out       INTEGER,
    loop_attempts    INTEGER,
    tools_used       TEXT,               -- comma-joined tool NAMES ('' = none), never args
    idempotency_hit  INTEGER,            -- bool, always written (0 = checked, no hit)
    error            TEXT
);

CREATE TABLE IF NOT EXISTS trace_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,           -- entry | triage | action_dispatch |
                                         -- teacher:<domain> | book | looper:<n> | cognition
    seq         INTEGER NOT NULL,        -- assigned at stage OPEN → true start order
    parent_seq  INTEGER,                 -- enclosing stage's seq; NULL at top level.
                                         -- `book` nests under `cognition`/`teacher:*`/
                                         -- `looper:<n>` — it runs INSIDE them, not after.
    ts_start    TEXT,
    ts_end      TEXT,
    ms          INTEGER,
    book        TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    meta        TEXT,                    -- whitelisted JSON metadata, never content
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_trace_stages_trace_id ON trace_stages (trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_session_id     ON traces (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_ts_start       ON traces (ts_start);
"""

SCHEMA += TRACE_DDL

# Columns added to the trace tables AFTER they first shipped. CREATE TABLE IF NOT
# EXISTS is a no-op on a database that already has the older table, so each one needs
# an explicit ALTER. Idempotent and additive by construction: a missing column is
# added, an existing one is left alone, and nothing is ever dropped or rewritten.
_TRACE_ADDED_COLUMNS = (
    ("trace_stages", "parent_seq", "INTEGER"),
)


def ensure_trace_schema(conn) -> None:
    """Create the trace tables if absent, then add any column a pre-existing
    database is missing. Safe to call on every write path — core.trace.flush() does,
    so a trace can be written by a process that never ran init_sqlite()."""
    conn.executescript(TRACE_DDL)
    for table, column, decl in _TRACE_ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            print(f"[db] migrated: {table}.{column} added")


def init_sqlite():
    os.makedirs(os.path.dirname(Config.SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.executescript(SCHEMA)
    ensure_trace_schema(conn)
    conn.commit()
    conn.close()
    print("[db] SQLite ready")