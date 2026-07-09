import sqlite3

from config import Config
from db.sqlite_init import ROUTING_OUTCOMES_DDL

# ── Slice A: passive outcome logging ──
# Record ONE row per completed teacher task into routing_outcomes. This is write-only:
# it takes NO routing decision and is never read this slice — it just accumulates data so
# a later slice can drive experience-based routing. Everything here is FAIL-SOFT: a
# logging error must never break the response, so the whole write is wrapped and only a
# warning is printed. Never raises.


def log_outcome(*, domain: str, book_used: str, ensemble: bool,
                self_check_passed: bool, attempts: int, tokens: int,
                latency_ms: int, complexity: str, session_id: str) -> None:
    """Append one routing outcome. Best-effort — swallows every error with a warning.

    book_used is comma-joined for an ensemble ("nemotron_super,groq"); ensemble booleans
    are stored as 0/1. ts defaults to CURRENT_TIMESTAMP in the table."""
    try:
        conn = sqlite3.connect(Config.SQLITE_PATH)
        try:
            # Ensure the table exists even if init_sqlite() hasn't run in this process —
            # keeps the logger self-contained without duplicating the DDL.
            conn.execute(ROUTING_OUTCOMES_DDL)
            conn.execute(
                "INSERT INTO routing_outcomes "
                "(domain, book_used, ensemble, self_check_passed, attempts, tokens, "
                " latency_ms, complexity, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    domain,
                    book_used,
                    int(bool(ensemble)),
                    int(bool(self_check_passed)),
                    int(attempts or 0),
                    int(tokens or 0),
                    int(latency_ms or 0),
                    complexity,
                    session_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        # Passive logging is best-effort — never let it break the turn.
        print(f"[outcomes] warning: failed to log routing outcome ({e})")
