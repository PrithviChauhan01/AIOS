"""Encrypted local vault for SECRET-tier data only.

This is the sink for anything triage flags `secret`. Content is Fernet-encrypted
and written to the local SQLite `vault` table — NEVER to ChromaDB, NEVER embedded,
NEVER returned into a cloud-bound prompt. Retrieval (vault_get) decrypts for
local/ollama use only.

Fail-safe posture: if the key or the crypto library is unavailable we refuse to
store rather than store in the clear, and we never fall back to a cloud path.
"""

import json
import sqlite3
import uuid

from config import Config

# Cached Fernet instance + the key actually in use this process.
_fernet = None
_key_warned = False

_VAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS vault (
    id          TEXT PRIMARY KEY,
    content     BLOB NOT NULL,
    meta        TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_fernet():
    """Build (once) the Fernet cipher.

    Key comes from Config.VAULT_KEY (.env). If it is missing we generate an
    ephemeral one and print a clear one-time instruction so the user can persist
    it — we do NOT crash, and we never route secret data anywhere but local.
    """
    global _fernet, _key_warned
    if _fernet is not None:
        return _fernet

    # Lazy import so a missing `cryptography` install can't break server import.
    from cryptography.fernet import Fernet

    key = (Config.VAULT_KEY or "").strip()
    if not key:
        key = Fernet.generate_key().decode()
        if not _key_warned:
            _key_warned = True
            print(
                "\n[vault] No VAULT_KEY set. Generated an EPHEMERAL key for this "
                "session only.\n[vault] Secrets stored now will be UNRECOVERABLE "
                "after restart. To persist them, add this line to backend/.env:\n"
                f"\n    VAULT_KEY={key}\n"
            )

    _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def _connect():
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.execute(_VAULT_SCHEMA)  # defensive: table exists even if init didn't run
    return conn


def vault_store(content: str, meta: dict = None) -> str:
    """Encrypt `content` and persist it locally. Returns the new row id.

    Raises if encryption fails — callers in the privacy path must treat a failure
    as "do not proceed / do not leak", never as "send to cloud instead".
    """
    cipher = _get_fernet()
    token = cipher.encrypt(content.encode("utf-8"))
    vid = str(uuid.uuid4())
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO vault (id, content, meta) VALUES (?, ?, ?)",
            (vid, token, json.dumps(meta or {})),
        )
        conn.commit()
    finally:
        conn.close()
    return vid


def vault_list() -> list:
    """List vault entries WITHOUT decrypting content (id + meta + created_at).

    Decryption is deliberately not done here so a listing can never spill plaintext.
    Use vault_get(id) to decrypt a single entry for local use.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, meta, created_at FROM vault ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for vid, meta, created_at in rows:
        try:
            parsed = json.loads(meta) if meta else {}
        except Exception:
            parsed = {}
        out.append({"id": vid, "meta": parsed, "created_at": created_at})
    return out


def vault_delete(vault_id: str) -> bool:
    """Remove one vault entry. Returns True if a row was deleted. Used when a
    secret-tier document is deleted from the document store."""
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM vault WHERE id = ?", (vault_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def vault_get(vault_id: str) -> str | None:
    """Decrypt and return one entry's content. LOCAL USE ONLY — the returned
    plaintext must never be placed in a cloud-bound prompt. Returns None if the
    id is unknown or decryption fails (e.g. wrong/rotated key)."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT content FROM vault WHERE id = ?", (vault_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    from cryptography.fernet import InvalidToken
    try:
        return _get_fernet().decrypt(row[0]).decode("utf-8")
    except InvalidToken:
        print(f"[vault] decrypt failed for {vault_id} — key mismatch?")
        return None
