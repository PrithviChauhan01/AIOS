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
import sqlite3

from config import Config
from tools.base import Tool
# Reuse the reminders time parser for follow_up_at — same natural-language phrasing
# ("in a week", "next Monday", "tomorrow"), same stored datetime format.
from tools.reminders import parse_due, _DT_FMT

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
    Defaults to engineering on a tie or no signal."""
    low = (text or "").lower()
    eng = sum(1 for s in _ENG_SIGNALS if s in low)
    des = sum(1 for s in _DES_SIGNALS if s in low)
    return "design" if des > eng else "engineering"


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

    resume = (resume_used or "").strip().lower()
    if resume not in ("engineering", "design"):
        resume = pick_resume(f"{role} {company}")

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
            "resume_used": resume, "status": status,
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
