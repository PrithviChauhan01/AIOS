"""Documents — general local document store (Slice D).

A WRITE tool like reminders/jobs. It ingests ANY document Sir asks to save —
resumes, certs, notes, IDs, contracts — parses the text, chunks it, and stores
it with privacy tiering decided AT SAVE TIME:

  • public / private  → embedded to the `mem_docs` ChromaDB namespace for semantic
                        recall (tier tagged on every chunk so a future cloud-bound
                        consumer can filter the same way core/profile does).
  • secret (IDs/financials, anything the privacy rules flag) → the encrypted local
                        VAULT only. NEVER embedded, NEVER placed in a cloud prompt.

Metadata for BOTH tiers lives in the SQLite `documents` table. Retrieval of a
secret doc returns its content for LOCAL use only; the orchestrator forces such a
turn onto the local model so the plaintext can't reach a cloud provider.

Scope of this slice: storage + retrieval + privacy tiering. No resume matching,
no outreach — those are later consumers of this store. is_action=True so a confirm
gate can later sit in front of writes, same as reminders/jobs.
"""

import os
import sqlite3

from config import Config
from tools.base import Tool
from core.vault import vault_store, vault_get, vault_delete
from db.chroma_init import get_chroma_client
# Reuse the SAME privacy rules triage uses, so a doc is tiered exactly like a turn.
from core.triage import _rule_sensitivity, _max_tier

# Chroma namespace for embedded document chunks (public/private only).
_DOCS_NS = "mem_docs"

# Doc types that are sensitive by nature even if the parsed text doesn't trip a
# rule (e.g. a sparse scan). Forces the encrypted vault.
_SECRET_DOC_TYPES = {
    "id", "ids", "identity", "aadhaar", "pan", "passport", "license", "licence",
    "ssn", "financial", "finance", "bank", "tax", "payslip", "salary slip",
}

# How much content a single get() hands to cognition — enough to be useful,
# bounded so a huge doc can't blow the prompt window.
_GET_CONTENT_CAP = 6000

_DOCS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    doc_type      TEXT,
    tags          TEXT,
    source_path   TEXT,
    tier          TEXT DEFAULT 'private',
    storage       TEXT NOT NULL,
    vault_id      TEXT,
    chunk_count   INTEGER DEFAULT 0,
    char_count    INTEGER DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


# ── Connection (same convention as tools/reminders.py / tools/jobs.py) ──
def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.execute(_DOCS_SCHEMA)  # defensive: table exists even if init didn't run
    conn.row_factory = sqlite3.Row
    return conn


# ── Parsing (PDF / txt / docx) ──
def _parse_file(path: str) -> str:
    """Extract plain text from a file. PDF via pypdf, docx via python-docx, plain
    text read directly. Raises ValueError for an unsupported type or a missing
    parser so save_document can turn it into a clarify, not a crash."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise ValueError(f"PDF support needs pypdf ({e})")
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in (".txt", ".md", ".markdown", ".text", ".csv", ".log", ""):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".docx":
        try:
            import docx
        except Exception as e:
            raise ValueError(f"DOCX support needs python-docx ({e})")
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    raise ValueError(f"unsupported file type {ext or '(none)'}")


def _chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list:
    """Char-based sliding-window chunking with overlap so a fact split across a
    boundary still surfaces in retrieval. Cheap and parser-agnostic."""
    text = text.strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


# ── Tiering ──
def _norm_tags(tags) -> str:
    """Normalize tags (str 'a,b' or list) to a clean comma-joined string."""
    if not tags:
        return ""
    if isinstance(tags, (list, tuple)):
        items = [str(t).strip() for t in tags]
    else:
        items = [t.strip() for t in str(tags).split(",")]
    return ",".join(t for t in items if t)


def _decide_tier(name: str, doc_type: str, tags: str, text: str) -> str:
    """Decide public/private/secret from the SAME rules triage uses, over both the
    metadata and the parsed content. doc_type in _SECRET_DOC_TYPES forces secret."""
    meta = " ".join(x for x in (name, doc_type or "", tags or "") if x)
    tier = _max_tier(_rule_sensitivity(meta), _rule_sensitivity(text))
    if (doc_type or "").strip().lower() in _SECRET_DOC_TYPES:
        tier = "secret"
    return tier


def _embed_chunks(doc_id: int, chunks: list, name: str, doc_type: str,
                  tags: str, tier: str) -> int:
    """Embed chunks into the mem_docs namespace, tier-tagged. Public/private only —
    a secret doc never reaches here. Returns the number of chunks embedded."""
    if not chunks:
        return 0
    col = get_chroma_client().get_or_create_collection(_DOCS_NS)
    col.add(
        documents=chunks,
        ids=[f"doc{doc_id}_chunk{i}" for i in range(len(chunks))],
        metadatas=[{
            "doc_id": doc_id, "name": name, "doc_type": doc_type or "",
            "tags": tags or "", "tier": tier, "chunk_index": i,
        } for i in range(len(chunks))],
    )
    return len(chunks)


# ── Action API (writes state) ──
def save_document(path: str, name: str = None, doc_type: str = None, tags=None) -> dict:
    """Ingest a document from a file path: parse → chunk → tier → store.

    Returns success → {"ok": True, "id", "name", "doc_type", "tier", "storage",
                       "chunk_count", "char_count"}
    or a clarify/error → {"ok": False, "needs"/"error", "ask"} so the caller can
    put it to Sir rather than fabricate a save."""
    path = (path or "").strip()
    if not path:
        return {"ok": False, "needs": "path",
                "ask": "Which file should I save, Sir? Give me the path."}
    if not os.path.isfile(path):
        return {"ok": False, "error": "file not found",
                "ask": f"I can't find a file at {path}, Sir."}
    try:
        text = _parse_file(path)
    except Exception as e:
        return {"ok": False, "error": str(e),
                "ask": f"I couldn't read that file, Sir ({e})."}
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "no extractable text",
                "ask": "That file has no extractable text, Sir — is it a scan or image?"}

    name = (name or os.path.basename(path)).strip()
    doc_type = (doc_type or "").strip().lower() or None
    tags = _norm_tags(tags)
    tier = _decide_tier(name, doc_type, tags, text)
    storage = "vault" if tier == "secret" else "chroma"

    # Insert metadata first to get the id, then embed/vault against it.
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO documents (name, doc_type, tags, source_path, tier, storage, "
            "vault_id, chunk_count, char_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, doc_type, tags, path, tier, storage, None, 0, len(text)),
        )
        conn.commit()
        doc_id = cur.lastrowid

        vault_id = None
        chunk_count = 0
        if storage == "vault":
            # SECRET → encrypt the full text locally; never embed.
            vault_id = vault_store(text, {"kind": "document", "name": name,
                                          "doc_type": doc_type, "tier": "secret"})
        else:
            chunk_count = _embed_chunks(doc_id, _chunk_text(text), name, doc_type, tags, tier)

        conn.execute(
            "UPDATE documents SET vault_id = ?, chunk_count = ? WHERE id = ?",
            (vault_id, chunk_count, doc_id),
        )
        conn.commit()
    finally:
        conn.close()

    print(f"[documents] WRITE db={_db_path()} id={doc_id} name={name!r} "
          f"tier={tier} storage={storage} chunks={chunk_count}")
    return {"ok": True, "id": doc_id, "name": name, "doc_type": doc_type,
            "tier": tier, "storage": storage, "chunk_count": chunk_count,
            "char_count": len(text)}


def list_documents(doc_type: str = None, tag: str = None) -> list:
    """List stored documents (metadata only — never content), newest first.
    Optional filters by doc_type and/or tag."""
    conn = _conn()
    try:
        sql = ("SELECT id, name, doc_type, tags, tier, storage, chunk_count, "
               "char_count, created_at FROM documents")
        clauses, params = [], []
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type.strip().lower())
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag.strip()}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _resolve_doc(identifier) -> dict | None:
    """Find a document row by numeric id or by name (exact, else unique substring).
    Returns the row dict, the sentinel {'_ambiguous': [...]} when a name matches
    several, or None if nothing matches."""
    conn = _conn()
    try:
        # numeric id?
        try:
            rid = int(identifier)
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (rid,)).fetchone()
            return dict(row) if row else None
        except (TypeError, ValueError):
            pass
        name = (identifier or "").strip().lower()
        if not name:
            return None
        rows = [dict(r) for r in conn.execute("SELECT * FROM documents").fetchall()]
    finally:
        conn.close()
    exact = [r for r in rows if r["name"].lower() == name]
    if len(exact) == 1:
        return exact[0]
    partial = [r for r in rows if name in r["name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        return {"_ambiguous": [{"id": r["id"], "name": r["name"]} for r in partial]}
    return None


def get_document(identifier) -> dict:
    """Retrieve a document's metadata + content by id or name. For a vault (secret)
    doc the content is decrypted for LOCAL use only and tier='secret' is returned so
    the caller keeps it off any cloud path. Chroma docs are reassembled from their
    chunks in order."""
    row = _resolve_doc(identifier)
    if row is None:
        return {"ok": False, "needs": "which", "ask": "I don't have a document matching that, Sir."}
    if "_ambiguous" in row:
        names = ", ".join(d["name"] for d in row["_ambiguous"])
        return {"ok": False, "needs": "which",
                "ask": f"Which one, Sir? I have: {names}."}

    if row["storage"] == "vault":
        content = vault_get(row["vault_id"]) if row["vault_id"] else None
        if content is None:
            return {"ok": False, "error": "vault read failed",
                    "ask": "I couldn't decrypt that one, Sir — the vault key may have changed."}
    else:
        col = get_chroma_client().get_or_create_collection(_DOCS_NS)
        got = col.get(where={"doc_id": row["id"]}, include=["documents", "metadatas"])
        pairs = sorted(
            zip(got.get("documents", []), got.get("metadatas", [])),
            key=lambda p: p[1].get("chunk_index", 0),
        )
        content = "".join(doc for doc, _ in pairs)

    return {"ok": True, "id": row["id"], "name": row["name"], "doc_type": row["doc_type"],
            "tags": row["tags"], "tier": row["tier"], "storage": row["storage"],
            "content": content or ""}


def delete_document(id: int) -> dict:
    """Delete a document everywhere it lives: its chunks in mem_docs, its vault
    entry if secret, and its metadata row. Returns the real outcome."""
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (id,)).fetchone()
        if row is None:
            conn.close()
            return {"ok": False, "found": False,
                    "ask": "I couldn't find that document, Sir."}
        row = dict(row)
        conn.execute("DELETE FROM documents WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()

    # Best-effort cleanup of the content stores; the metadata row is already gone.
    if row["storage"] == "vault" and row["vault_id"]:
        try:
            vault_delete(row["vault_id"])
        except Exception as e:
            print(f"[documents] vault cleanup failed for {id}: {e}")
    else:
        try:
            col = get_chroma_client().get_or_create_collection(_DOCS_NS)
            col.delete(where={"doc_id": id})
        except Exception as e:
            print(f"[documents] chroma cleanup failed for {id}: {e}")

    print(f"[documents] DELETE db={_db_path()} id={id} name={row['name']!r}")
    return {"ok": True, "found": True, "id": id, "name": row["name"]}


# ── Pool wrapper ──
class DocumentsTool(Tool):
    """Shared-pool handle for the document store. is_action=True marks it as
    state-writing. fetch() is the read-only list view (the ABC requires it)."""

    name = "documents"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_documents()
            results = [
                f"#{r['id']} {r['name']} [{r['doc_type'] or 'doc'}] "
                f"({r['tier']}/{r['storage']})"
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:documents] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — module functions stay the source of truth.
    def save(self, path: str, name: str = None, doc_type: str = None, tags=None) -> dict:
        return save_document(path, name, doc_type, tags)

    def list(self, doc_type: str = None, tag: str = None) -> list:
        return list_documents(doc_type, tag)

    def get(self, identifier) -> dict:
        return get_document(identifier)

    def delete(self, id: int) -> dict:
        return delete_document(id)
