"""Jobs — the job-application log (income track, Slice C).

A WRITE tool like reminders: it persists to the existing `jobs` table
(company, role, resume_used, status, applied_at, follow_up_at, url, notes).

Scope of this slice: manual logging + listing of applications only. There is NO
auto-apply (that needs the confirm-gate slice) and NO job-board scraping (needs
API keys). The JobsTeacher (agents/jobs.py) produces the role shortlist; this
tool records the ones Sir actually applies to, through the same gate → extract →
execute → real-result action path reminders use.

The public API is the module-level functions plus pick_resume. JobsTool wraps
them so the tool lives in the shared pool (registry) like any other tool, and is
tagged is_action=True so actions can later route through a confirm gate.
"""

import os
import re
import sqlite3
from collections import Counter

from config import Config
from tools.base import Tool
# Reuse the reminders time parser for follow_up_at — same natural-language phrasing
# ("in a week", "next Monday", "tomorrow"), same stored datetime format.
from tools.reminders import parse_due, _DT_FMT
# Resume matching CONSUMES the Slice-D document store: real resume content drives
# the engineering/design choice instead of a hardcoded keyword guess.
from tools.documents import list_documents, get_document

# ── Resume routing — engineering vs design ──
# The teacher routes per role via the LLM (see agents/jobs.py). This deterministic
# version is the default used when Sir logs an application WITHOUT naming which
# resume he sent, inferred from the role/company text.
_ENG_SIGNALS = (
    "engineer", "engineering", "developer", "software", "swe", "backend",
    "frontend", "front-end", "full stack", "fullstack", "full-stack", "data",
    "ml", "machine learning", "ai", "devops", "infra", "platform", "sre",
    "embedded", "programmer", "qa", "security",
)
_DES_SIGNALS = (
    "design", "designer", "ux", "ui", "product design", "visual", "brand",
    "creative", "figma", "graphic", "motion", "illustrat", "art director",
)


def pick_resume(text: str) -> str:
    """Infer which resume fits a role/company string: 'engineering' or 'design'.
    Defaults to engineering on a tie or no signal. Used only as a tie-break /
    variant fallback now that matching runs against the ACTUAL stored resume."""
    low = (text or "").lower()
    eng = sum(1 for s in _ENG_SIGNALS if s in low)
    des = sum(1 for s in _DES_SIGNALS if s in low)
    return "design" if des > eng else "engineering"


# ── Resume matching against the real document store (Slice E) ──
# Generic words carry no signal for matching a JD to a resume; drop them so the
# overlap reflects real skills/experience, not filler.
_SKILL_STOP = {
    "and", "the", "for", "with", "you", "your", "our", "are", "will", "have",
    "this", "that", "from", "role", "roles", "work", "team", "teams", "years",
    "year", "experience", "experienced", "strong", "ability", "including",
    "etc", "able", "across", "into", "using", "use", "used", "build", "building",
    "company", "companies", "looking", "join", "based", "remote", "hybrid",
    "responsibilities", "requirements", "preferred", "plus", "skills", "resume",
    "job", "position", "candidate", "we", "a", "an", "to", "of", "in", "on",
    "as", "at", "is", "or", "be", "by",
}


def _skill_tokens(text: str) -> set:
    """Meaningful skill/keyword tokens from JD or resume text (lowercased, filler
    and bare numbers removed). Keeps tech-ish tokens like c++, node.js, ui/ux."""
    toks = re.findall(r"[a-z][a-z0-9+#.]{1,}", (text or "").lower())
    return {t.strip(".") for t in toks if t not in _SKILL_STOP and len(t) > 1}


def _resume_variant(doc: dict) -> str | None:
    """The resume's variant from its metadata (name/type/tags), or None if not
    stated there — caller then infers from content."""
    blob = f"{doc.get('name','')} {doc.get('doc_type','')} {doc.get('tags','')}".lower()
    if "design" in blob:
        return "design"
    if any(k in blob for k in ("engineering", "engineer", "swe", "software", "eng")):
        return "engineering"
    return None


def list_resumes() -> list:
    """Document rows that are resumes (by type, tag, or name). Consumes the store."""
    out = []
    for d in list_documents():
        blob = f"{d.get('name','')} {d.get('doc_type','')} {d.get('tags','')}".lower()
        if (d.get("doc_type") or "").lower() == "resume" or "resume" in blob \
                or re.search(r"\bcv\b", blob):
            out.append(d)
    return out


def _load_resume_docs() -> list:
    """Each resume on file with its variant + full content pulled from the store."""
    docs = []
    for d in list_resumes():
        got = get_document(d["id"])
        content = got.get("content", "") if got.get("ok") else ""
        docs.append({"id": d["id"], "name": d["name"],
                     "variant": _resume_variant(d) or pick_resume(content),
                     "content": content})
    return docs


def resume_profiles() -> list:
    """Compact per-resume profiles (variant + top skills) for the teacher prompt —
    enough to ground the LLM's per-role routing in REAL skills without sending raw
    resume content (PII) to a cloud book."""
    profs = []
    for r in _load_resume_docs():
        freq = Counter(t for t in _skill_tokens(r["content"]))
        skills = [w for w, _ in freq.most_common(15)]
        profs.append({"id": r["id"], "name": r["name"], "variant": r["variant"], "skills": skills})
    return profs


def match_resume(text: str) -> dict:
    """Choose the resume variant that best fits a role/JD by overlapping the JD's
    skill tokens with each stored resume's ACTUAL content.

    Returns {"ok": True, "variant", "matched": [skills], "score", "candidates"} or
    {"ok": False, "reason": "no_resume"} when nothing is on file — caller then says
    so plainly instead of guessing."""
    docs = _load_resume_docs()
    if not docs:
        return {"ok": False, "reason": "no_resume"}
    want = _skill_tokens(text)
    scored = []
    for r in docs:
        matched = sorted(want & _skill_tokens(r["content"]))
        scored.append({"variant": r["variant"], "name": r["name"],
                       "matched": matched, "score": len(matched)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    # Tie across different variants → fall back to the keyword heuristic on the JD.
    if len(scored) > 1 and scored[1]["score"] == best["score"] \
            and scored[1]["variant"] != best["variant"]:
        pref = pick_resume(text)
        best = next((s for s in scored if s["variant"] == pref), best)
    return {"ok": True, "variant": best["variant"], "matched": best["matched"],
            "score": best["score"], "candidates": scored}


# ── Connection (same convention as tools/reminders.py) ──
def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Action API (writes state) ──
def log_application(company: str, role: str, resume_used: str = None,
                    status: str = "applied", url: str = None,
                    follow_up_at=None, notes: str = None) -> dict:
    """Record a job application in the `jobs` table.

    Validates before writing: company and role are required. `resume_used` is
    normalized to engineering|design, inferred from the role when not given.
    `follow_up_at` may be a natural-language time ('in a week') or empty.

    Returns either:
      success  → {"ok": True, "id", "company", "role", "resume_used", "status",
                  "follow_up_at", "url"}
      clarify  → {"ok": False, "needs": "company"|"role", "ask": "..."}
    A clarifying ask is returned (not raised) so the caller can put it to Sir."""
    company = (company or "").strip()
    role = (role or "").strip()
    if not company:
        return {"ok": False, "needs": "company",
                "ask": "Which company should I log that application under, Sir?"}
    if not role:
        return {"ok": False, "needs": "role", "ask": "What role was it, Sir?"}

    # Resume choice: honor an explicit one; otherwise MATCH the role against Sir's
    # actual stored resumes — not a hardcoded label. If none on file, say so (store
    # NULL) rather than guess.
    resume = (resume_used or "").strip().lower()
    resume_note = None
    if resume not in ("engineering", "design"):
        m = match_resume(f"{role} {company}")
        if m.get("ok"):
            resume = m["variant"]
            print(f"[jobs] resume match: role={role!r} -> chose={resume} "
                  f"(matched skills: {', '.join(m['matched'][:8]) or 'none'})")
        else:
            resume = None
            resume_note = "no resume on file"
            print(f"[jobs] resume match: role={role!r} -> no resume on file")

    status = (status or "applied").strip().lower() or "applied"

    follow_up = None
    if follow_up_at:
        try:
            follow_up = parse_due(follow_up_at).strftime(_DT_FMT)
        except Exception:
            follow_up = None  # unparseable follow-up is non-fatal; just store none

    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO jobs (company, role, resume_used, status, follow_up_at, url, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (company, role, resume, status, follow_up, (url or None), (notes or None)),
        )
        conn.commit()
        jid = cur.lastrowid
        print(f"[jobs] WRITE db={_db_path()} id={jid} company={company!r} "
              f"role={role!r} resume={resume}")
    finally:
        conn.close()
    return {"ok": True, "id": jid, "company": company, "role": role,
            "resume_used": resume, "resume_note": resume_note, "status": status,
            "follow_up_at": follow_up, "url": url or None}


def list_applications(status: str = None) -> list:
    """Return logged applications, most recent first. Optional status filter."""
    conn = _conn()
    try:
        sql = ("SELECT id, company, role, resume_used, status, applied_at, "
               "follow_up_at, url FROM jobs")
        params = ()
        if status:
            sql += " WHERE status = ?"
            params = (status.strip().lower(),)
        sql += " ORDER BY applied_at DESC"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Pool wrapper ──
class JobsTool(Tool):
    """Shared-pool handle for the jobs action tool. is_action=True marks it as
    state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): it lists logged applications, fail-soft. The write action is exposed as
    log() which delegates to the module function above."""

    name = "jobs"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_applications()
            results = [
                f"#{r['id']} {r['company']} — {r['role']} "
                f"[{r['resume_used']}] ({r['status']})"
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:jobs] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def log(self, company: str, role: str, resume_used: str = None,
            status: str = "applied", url: str = None, follow_up_at=None,
            notes: str = None) -> dict:
        return log_application(company, role, resume_used, status, url, follow_up_at, notes)

    def list(self, status: str = None) -> list:
        return list_applications(status)
