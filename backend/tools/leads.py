"""Leads — persists the prospecting rows the LeadgenTeacher finds.

A WRITE tool like reminders/jobs: it persists to the existing `leads` table
(studio_name, location, contact, research, outreach_sent, response, created_at,
follow_up_at). Before this tool existed a prospecting run was VERIFIED Places
data that only ever lived in the chat transcript and the in-process result cache
(agents/leadgen.py:_RESULT_CACHE, which dies with the process) — so the
dashboard's Leads view stayed permanently empty even though Sir had real,
researched leads on screen. This tool is what closes that gap, through the same
gate -> extract -> execute -> real-result action path reminders/jobs use.

BULK is the primary shape here, and that is the difference from every other
action tool: one research turn returns N businesses and ALL of them must land,
so save() takes the whole row set in a single transaction. A single manually
named lead is just the N=1 case.

Re-saving the same set is a normal thing for Sir to do ("save those" twice, or
a repeat search) — so an already-stored studio_name is SKIPPED, not duplicated,
and the real saved/skipped split is returned so cognition confirms what actually
happened rather than claiming N new rows.
"""

import os
import sqlite3

from config import Config
from tools.base import Tool


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _contact_of(row: dict) -> str | None:
    """Phone + website into the single `contact` column, in that order — the two
    fields Sir actually calls/opens from a call sheet."""
    parts = [str(row.get(k)).strip() for k in ("phone", "website")
             if row.get(k) and str(row.get(k)).strip() not in ("", "—")]
    return " · ".join(parts) or None


def _research_of(row: dict) -> str | None:
    """The verified signal Places returned (rating + review count), or an explicit
    research note if the caller passed one. Never invented."""
    if row.get("research"):
        return str(row["research"]).strip() or None
    rating = row.get("rating")
    reviews = row.get("reviews")
    if rating is None and not reviews:
        return None
    head = f"{rating}" if rating is not None else "unrated"
    return f"Rating {head} ({reviews or 0} reviews)"


# ── Action API (writes state) ──
def save_leads(rows: list, location: str = None) -> dict:
    """Bulk-insert prospecting rows into the `leads` table, one row per business.

    `rows` are the leadgen/Places dicts (name, phone, website, address, rating,
    reviews) or already-shaped lead dicts (studio_name, location, contact,
    research). A row with no name is skipped — a nameless lead is not a lead.
    ONE connection/transaction for the whole set, so a 10-result research turn
    either lands as 10 rows or fails as a unit.

    Returns either:
      success  → {"ok": True, "items": [row, ...], "count": N, "skipped": [name, ...]}
      clarify  → {"ok": False, "needs": "rows", "ask": "..."}
    `items` are the REAL inserted rows (with their assigned ids); `skipped` are
    names already on file. Never raises."""
    if not rows:
        return {"ok": False, "needs": "rows",
                "ask": "I have no leads in hand to save, Sir — run the search first."}

    saved, skipped = [], []
    conn = _conn()
    try:
        # Existing names, lowercased — the dedup key. Loaded once, then kept in sync
        # as we insert, so duplicates WITHIN one batch are caught too.
        existing = {(r["studio_name"] or "").strip().lower()
                    for r in conn.execute("SELECT studio_name FROM leads")}
        for row in rows:
            name = str(row.get("name") or row.get("studio_name") or "").strip()
            if not name:
                continue
            if name.lower() in existing:
                skipped.append(name)
                continue
            loc = str(row.get("location") or row.get("address") or location or "").strip() or None
            contact = _contact_of(row)
            research = _research_of(row)
            cur = conn.execute(
                "INSERT INTO leads (studio_name, location, contact, research) "
                "VALUES (?, ?, ?, ?)",
                (name, loc, contact, research),
            )
            existing.add(name.lower())
            saved.append({"id": cur.lastrowid, "studio_name": name, "location": loc,
                          "contact": contact, "research": research})
        conn.commit()
        print(f"[leads] WRITE db={_db_path()} saved={len(saved)} skipped={len(skipped)}")
    finally:
        conn.close()

    return {"ok": True, "items": saved, "count": len(saved), "skipped": skipped}


def save_lead(name: str, location: str = None, contact: str = None,
              research: str = None) -> dict:
    """Save ONE manually named lead. The N=1 case of save_leads — same validation,
    same row shape, so there is one write path, not two."""
    name = (name or "").strip()
    if not name:
        return {"ok": False, "needs": "name", "ask": "Which studio should I save, Sir?"}
    return save_leads([{"studio_name": name, "location": location,
                        "contact": contact, "research": research}])


def list_leads(limit: int = None) -> list:
    """Return saved leads, most recent first."""
    conn = _conn()
    try:
        sql = ("SELECT id, studio_name, location, contact, research, outreach_sent, "
               "response, created_at FROM leads ORDER BY created_at DESC, id DESC")
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Pool wrapper ──
class LeadsTool(Tool):
    """Shared-pool handle for the leads action tool. is_action=True marks it as
    state-writing so a confirm gate can later sit in front of its actions.

    fetch() is the read-only view (the ABC requires it and pool plumbing calls
    it): it lists saved leads, fail-soft. The write actions delegate to the
    module functions above, which stay the source of truth."""

    name = "leads"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            items = list_leads(limit=20)
            results = [
                f"#{r['id']} {r['studio_name']} — {r['location'] or 'no location'}"
                + (f" ({r['contact']})" if r.get("contact") else "")
                for r in items
            ]
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:leads] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — kept thin so the module functions stay the source of truth.
    def save_many(self, rows: list, location: str = None) -> dict:
        return save_leads(rows, location)

    def save(self, name: str, location: str = None, contact: str = None,
             research: str = None) -> dict:
        return save_lead(name, location, contact, research)

    def list(self, limit: int = None) -> list:
        return list_leads(limit)
