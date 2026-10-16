"""One-off cleanup: redact secret-tier plaintext already sitting in `conversations`.

Secret turns were written to conversations.content verbatim before core.memory
started redacting them. The vault has a correctly encrypted copy of each; this
script removes the plaintext duplicate. It is EXPLICITLY INVOKED — importing it does
nothing, and nothing in the request path calls it.

    python -m db.redact_secrets                  # dry run: report, change nothing
    python -m db.redact_secrets --apply          # redact the trace-confirmed rows
    python -m db.redact_secrets --scan           # review candidates traces can't reach
    python -m db.redact_secrets --purge-ids 1,2  # redact exactly these rows

Two passes, because the evidence for them is not the same strength:

  CONFIRMED — the `traces` table records each request's resolved sensitivity along
  with its session and its start/end timestamps. A conversations row inside a
  secret-tier trace's window, in that trace's session, WAS that turn. That is a
  join on recorded facts, not an inference from content, so --apply acts on it.

  CANDIDATES — turns that predate the trace spine have no such record. Their tier
  was never stored, so it cannot be recovered. --scan reports them for review and
  stops; the operator decides and passes ids to --purge-ids. Nothing is guessed,
  nothing is deleted, and no row is touched on the strength of what it looks like.

Rows are never DELETED. content is overwritten with the placeholder and redacted=1
is set, so the turn stays in the transcript and history stays coherent — exactly
what a live secret turn now writes.

This script never prints message content. Not in a dry run, not in a scan, not in an
error. Ids, roles, timestamps and lengths only — printing the plaintext to a terminal
or a log while removing it from the database would defeat the entire exercise.
"""

import argparse
import re
import sqlite3
import sys

from config import Config
from core.memory import SECRET_PLACEHOLDER
from db.sqlite_init import ensure_conversation_schema

# Only the human's own message is redacted. The assistant reply on a secret turn is
# the acknowledgement — no value in it — and blanking it would leave a transcript
# where she appears never to have answered. Same rule core.memory applies live.
_ROLE = "user"

# A conversations row sits inside a secret trace's window, in the same session.
# conversations.timestamp is second-resolution ('YYYY-MM-DD HH:MM:SS'); traces carry
# milliseconds, so the window is truncated to seconds on both ends before comparing —
# string comparison is a valid time comparison in this format. Rows already redacted
# are skipped so a re-run is a no-op.
_CONFIRMED_SQL = """
SELECT DISTINCT cv.id, cv.session_id, cv.role, cv.timestamp, LENGTH(cv.content), t.trace_id
FROM conversations cv
JOIN traces t
  ON t.session_id = cv.session_id
 AND cv.timestamp >= substr(t.ts_start, 1, 19)
 AND cv.timestamp <= substr(t.ts_end, 1, 19)
WHERE t.sensitivity = 'secret'
  AND cv.role = ?
  AND COALESCE(cv.redacted, 0) = 0
ORDER BY cv.id
"""


def _connect():
    conn = sqlite3.connect(Config.SQLITE_PATH)
    ensure_conversation_schema(conn)
    conn.commit()
    return conn


def find_confirmed(conn) -> list:
    """Rows a secret-tier trace positively identifies. Metadata only."""
    return [
        {"id": rid, "session_id": sid, "role": role, "timestamp": ts,
         "chars": length, "trace_id": tid}
        for rid, sid, role, ts, length, tid in conn.execute(_CONFIRMED_SQL, (_ROLE,))
    ]


def find_candidates(conn) -> list:
    """Un-redacted user rows mentioning a privacy label from privacy.local, that no
    secret trace confirms. Review material — a label in a message means the message
    is ABOUT that subject, not that the turn was classified secret ("explain how PAN
    validation works" matches too). Reported, never acted on."""
    try:
        from core.triage import _RULES
        labels = _RULES.get("patterns") or []
    except Exception as e:
        print(f"  (privacy labels unavailable: {e})")
        return []
    if not labels:
        return []

    confirmed = {r["id"] for r in find_confirmed(conn)}
    patterns = [(w, re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)) for w in labels]
    out = []
    for rid, sid, ts, content in conn.execute(
        "SELECT id, session_id, timestamp, content FROM conversations "
        "WHERE role = ? AND COALESCE(redacted, 0) = 0 ORDER BY id", (_ROLE,)
    ):
        if rid in confirmed:
            continue
        hits = [w for w, rx in patterns if rx.search(content or "")]
        if hits:
            out.append({"id": rid, "session_id": sid, "timestamp": ts,
                        "chars": len(content or ""), "labels": ",".join(hits)})
    return out


def check_recoverable(conn, ids) -> tuple:
    """Split `ids` into (recoverable, orphaned) by whether the vault still holds that
    row's exact content.

    Redaction is irreversible, so this is the gate in front of it: the vault copy is
    what makes overwriting the plaintext safe rather than destructive. A row whose
    content is NOT in the vault (vault write failed at the time, or an ephemeral
    VAULT_KEY was lost to a restart) would be destroyed, not redacted — so --apply
    skips it and says so.

    Plaintext is compared in memory and never returned, printed or logged; only the
    ids come back out."""
    if not ids:
        return [], []
    try:
        from core.vault import vault_get, vault_list
        vaulted = set()
        for entry in vault_list():
            text = vault_get(entry["id"])
            if text is not None:
                vaulted.add(text.strip())
    except Exception as e:
        print(f"  (vault unreadable: {type(e).__name__} — treating every row as orphaned)")
        return [], list(ids)

    marks = ",".join("?" * len(ids))
    recoverable, orphaned = [], []
    for rid, content in conn.execute(
        f"SELECT id, content FROM conversations WHERE id IN ({marks})", tuple(ids)
    ):
        (recoverable if (content or "").strip() in vaulted else orphaned).append(rid)
    return sorted(recoverable), sorted(orphaned)


def redact_ids(conn, ids) -> int:
    """Overwrite content with the placeholder and mark the rows redacted. Returns the
    number of rows changed. Idempotent — an already-redacted row is left alone."""
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    cur = conn.execute(
        f"UPDATE conversations SET content = ?, redacted = 1 "
        f"WHERE id IN ({marks}) AND COALESCE(redacted, 0) = 0",
        (SECRET_PLACEHOLDER, *ids),
    )
    conn.commit()
    return cur.rowcount


def _report(title, rows, columns):
    print(f"\n{title}: {len(rows)}")
    for r in rows:
        print("  " + "  ".join(f"{c}={r[c]}" for c in columns))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="redact the trace-confirmed rows (default is a dry run)")
    ap.add_argument("--scan", action="store_true",
                    help="also list unconfirmed candidates for manual review")
    ap.add_argument("--purge-ids", default="",
                    help="comma-separated conversations.id values to redact outright")
    ap.add_argument("--force", action="store_true",
                    help="redact even rows the vault cannot recover (destructive)")
    args = ap.parse_args(argv)

    conn = _connect()
    try:
        if args.purge_ids:
            try:
                ids = [int(x) for x in args.purge_ids.split(",") if x.strip()]
            except ValueError:
                print("--purge-ids takes integers, e.g. --purge-ids 165,191")
                return 2
            changed = redact_ids(conn, ids)
            print(f"redacted {changed} row(s) by id: {ids}")
            return 0

        confirmed = find_confirmed(conn)
        _report("Confirmed secret-tier rows (matched to a secret trace)", confirmed,
                ("id", "session_id", "timestamp", "chars", "trace_id"))

        recoverable, orphaned = check_recoverable(conn, [r["id"] for r in confirmed])
        print(f"\n  vault holds a copy of: {len(recoverable)} — safe to redact")
        if orphaned:
            print(f"  NOT in the vault:      {orphaned} — redacting these DESTROYS "
                  f"them.\n  They are skipped. Recover or accept the loss, then "
                  f"re-run with --force.")

        if args.scan:
            candidates = find_candidates(conn)
            _report("Unconfirmed candidates — REVIEW, nothing will be done to these",
                    candidates, ("id", "session_id", "timestamp", "chars", "labels"))
            if candidates:
                print("\n  A label match means the message mentions the subject, not "
                      "that it was secret.\n  Inspect them yourself, then redact the "
                      "real ones:\n    python -m db.redact_secrets --purge-ids "
                      + ",".join(str(c["id"]) for c in candidates))

        targets = recoverable + (orphaned if args.force else [])
        if not args.apply:
            print(f"\nDRY RUN — nothing written. Re-run with --apply to redact "
                  f"{len(targets)} row(s).")
            return 0

        changed = redact_ids(conn, targets)
        print(f"\nredacted {changed} row(s); content replaced with "
              f"{SECRET_PLACEHOLDER!r}, redacted=1. No rows deleted.")
        if orphaned and not args.force:
            print(f"skipped {len(orphaned)} unrecoverable row(s): {orphaned}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
