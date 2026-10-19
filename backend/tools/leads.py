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
a repeat search) — so an already-stored studio_name + location is SKIPPED, not duplicated,
and the real saved/skipped split is returned so cognition confirms what actually
happened rather than claiming N new rows.
"""

import os
import re
import sqlite3
from datetime import date, datetime, timedelta

from config import Config
from tools.base import Tool


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


_SCHEMA_READY: set = set()  # db paths already migrated this process


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    # Pipeline columns arrive by ALTER on an older table — make sure they're there
    # before any read/write here, once per database per process.
    path = _db_path()
    if path not in _SCHEMA_READY:
        from db.sqlite_init import ensure_leads_schema
        ensure_leads_schema(conn)
        conn.commit()
        _SCHEMA_READY.add(path)
    return conn


def _dedup_key(name, location) -> tuple:
    return ((name or "").strip().lower(), (location or "").strip().lower())


def _contact_of(row: dict) -> str | None:
    """Phone + email + website into the single `contact` column, in that order — the
    fields Sir actually calls/opens from a call sheet."""
    parts = [str(row.get(k)).strip() for k in ("phone", "email", "website")
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
        # Existing (name, location) pairs, lowercased — the dedup key: the same studio
        # name in another city is a different lead. Loaded once, then kept in sync as
        # we insert, so duplicates WITHIN one batch are caught too.
        existing = {_dedup_key(r["studio_name"], r["location"])
                    for r in conn.execute("SELECT studio_name, location FROM leads")}
        for row in rows:
            name = str(row.get("name") or row.get("studio_name") or "").strip()
            if not name:
                continue
            loc = str(row.get("location") or row.get("address") or location or "").strip() or None
            key = _dedup_key(name, loc)
            if key in existing:
                skipped.append(name)
                continue
            # An already-shaped row (leadgen's web leads) carries its own contact.
            contact = row.get("contact") or _contact_of(row)
            research = _research_of(row)
            cur = conn.execute(
                "INSERT INTO leads (studio_name, location, contact, research) "
                "VALUES (?, ?, ?, ?)",
                (name, loc, contact, research),
            )
            existing.add(key)
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
        sql = (f"SELECT {_LEAD_COLS} FROM leads ORDER BY created_at DESC, id DESC")
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── Sales pipeline (Slice B2) ──
# found → contacted → replied → demo → won / lost. Every write resolves ONE lead by
# name first; an unknown or ambiguous name is a clarify result, never a guess.
STAGES = ("found", "contacted", "replied", "demo", "won", "lost")
CLOSED_STAGES = ("won", "lost")
# Stages that mean Sir actually exchanged words with the studio — moving into one
# stamps last_contact_at.
_CONTACT_STAGES = ("contacted", "replied", "demo")
_CONTACTED_FOLLOW_UP_DAYS = 3
_LEAD_COLS = ("id, studio_name, location, contact, research, outreach_sent, response, "
              "created_at, stage, last_contact_at, next_follow_up_at, notes")


def _now() -> datetime:
    return datetime.now()


def _stamp() -> str:
    return _now().strftime("%Y-%m-%d %H:%M")


def _append_note_sql(conn, lead_id: int, text: str) -> None:
    """Append-only: the log is only ever concatenated onto, never rewritten."""
    conn.execute(
        "UPDATE leads SET notes = COALESCE(notes, '') || ? WHERE id = ?",
        (f"[{_stamp()}] {text}\n", lead_id))


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def resolve_lead(conn, name: str) -> dict:
    """{"ok": True, "lead": row} for exactly one match, else a clarify result.
    Exact (case/punctuation-insensitive) name wins; otherwise a lead whose name
    starts with, or contains as whole words, what Sir said ("8K" → "8K Entertainment")."""
    want = _norm(name)
    if not want:
        return {"ok": False, "needs": "name", "ask": "Which lead, Sir?"}
    rows = [dict(r) for r in conn.execute(f"SELECT {_LEAD_COLS} FROM leads")]
    exact = [r for r in rows if _norm(r["studio_name"]) == want]
    matches = exact or [r for r in rows
                        if f" {want} " in f" {_norm(r['studio_name'])} "]
    if len(matches) == 1:
        return {"ok": True, "lead": matches[0]}
    if not matches:
        return {"ok": False, "needs": "lead", "not_found": name,
                "ask": f"I have no lead called \"{name}\" on file, Sir."}
    names = ", ".join(f"{r['studio_name']} ({r['location'] or 'no location'})"
                      for r in matches[:5])
    return {"ok": False, "needs": "lead", "ambiguous": name,
            "ask": f"\"{name}\" matches more than one lead — {names}. Which one, Sir?"}


def _fetch(conn, lead_id: int) -> dict:
    return dict(conn.execute(f"SELECT {_LEAD_COLS} FROM leads WHERE id = ?",
                             (lead_id,)).fetchone())


def _as_date(when) -> str:
    """A follow-up phrase or date → 'YYYY-MM-DD'. Raises ValueError if unparseable."""
    if isinstance(when, date) and not isinstance(when, datetime):
        return when.isoformat()
    # ISO first: parse_due reads "2026-10-15" as a clock time (tomorrow 10:00).
    try:
        return date.fromisoformat(str(when).strip()).isoformat()
    except ValueError:
        pass
    from tools.reminders import parse_due
    return parse_due(when, now=_now()).date().isoformat()


def set_stage(name: str, stage: str, follow_up=None) -> dict:
    """Move a lead to `stage`. contacted → next follow-up in 3 days unless one is
    given; won/lost → follow-up cleared; a contact stage stamps last_contact_at.
    The change is appended to the lead's notes log."""
    stage = (stage or "").strip().lower()
    if stage not in STAGES:
        return {"ok": False, "needs": "stage",
                "ask": f"Which stage, Sir — {', '.join(STAGES)}?"}
    try:
        given = _as_date(follow_up) if follow_up else None
    except ValueError:
        return {"ok": False, "needs": "date",
                "ask": f"I couldn't read \"{follow_up}\" as a date, Sir."}
    conn = _conn()
    try:
        found = resolve_lead(conn, name)
        if not found["ok"]:
            return found
        lead = found["lead"]
        before = lead["stage"] or "found"
        follow = lead["next_follow_up_at"]
        if stage in CLOSED_STAGES:
            follow = None
        elif given:
            follow = given
        elif stage == "contacted":
            follow = (_now().date() + timedelta(days=_CONTACTED_FOLLOW_UP_DAYS)).isoformat()
        last_contact = (_now().strftime("%Y-%m-%d %H:%M:%S") if stage in _CONTACT_STAGES
                        else lead["last_contact_at"])
        conn.execute(
            "UPDATE leads SET stage = ?, next_follow_up_at = ?, last_contact_at = ? "
            "WHERE id = ?", (stage, follow, last_contact, lead["id"]))
        _append_note_sql(conn, lead["id"], f"stage: {before} → {stage}")
        conn.commit()
        updated = _fetch(conn, lead["id"])
        print(f"[leads] WRITE stage #{lead['id']} {before}->{stage} follow_up={follow}")
    finally:
        conn.close()
    return {"ok": True, "lead": updated, "from_stage": before}


def add_note(name: str, text: str) -> dict:
    """Append one timestamped line to the lead's notes log."""
    text = " ".join((text or "").split())
    if not text:
        return {"ok": False, "needs": "note", "ask": "What should the note say, Sir?"}
    conn = _conn()
    try:
        found = resolve_lead(conn, name)
        if not found["ok"]:
            return found
        _append_note_sql(conn, found["lead"]["id"], text)
        conn.commit()
        updated = _fetch(conn, found["lead"]["id"])
        print(f"[leads] WRITE note #{updated['id']}")
    finally:
        conn.close()
    return {"ok": True, "lead": updated, "note": text}


def set_follow_up(name: str, when) -> dict:
    """Set the lead's next follow-up date. Refused on a won/lost lead — a closed
    deal has nothing to follow up."""
    try:
        follow = _as_date(when)
    except ValueError:
        return {"ok": False, "needs": "date",
                "ask": "When should I set the follow-up for, Sir?"
                       if not when else f"I couldn't read \"{when}\" as a date, Sir."}
    conn = _conn()
    try:
        found = resolve_lead(conn, name)
        if not found["ok"]:
            return found
        lead = found["lead"]
        if lead["stage"] in CLOSED_STAGES:
            return {"ok": False, "needs": "stage", "lead": lead,
                    "ask": f"{lead['studio_name']} is marked {lead['stage']} — "
                           "reopen it first if you want a follow-up, Sir."}
        conn.execute("UPDATE leads SET next_follow_up_at = ? WHERE id = ?",
                     (follow, lead["id"]))
        _append_note_sql(conn, lead["id"], f"follow-up set for {follow}")
        conn.commit()
        updated = _fetch(conn, lead["id"])
        print(f"[leads] WRITE follow-up #{lead['id']} -> {follow}")
    finally:
        conn.close()
    return {"ok": True, "lead": updated}


def list_by_stage(stage: str = None) -> list:
    """Leads in one stage (or all), in pipeline order, then most recently contacted."""
    order = "CASE stage " + " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(STAGES)) \
        + " ELSE 99 END"
    conn = _conn()
    try:
        if stage:
            rows = conn.execute(
                f"SELECT {_LEAD_COLS} FROM leads WHERE stage = ? "
                "ORDER BY last_contact_at DESC, id", (stage.lower(),)).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_LEAD_COLS} FROM leads "
                f"ORDER BY {order}, last_contact_at DESC, id").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def due_follow_ups(today: date = None) -> list:
    """Open leads whose next follow-up is today or overdue — most overdue first."""
    today = (today or _now().date()).isoformat()
    conn = _conn()
    try:
        rows = conn.execute(
            f"SELECT {_LEAD_COLS} FROM leads WHERE next_follow_up_at IS NOT NULL "
            "AND next_follow_up_at <= ? AND stage NOT IN ('won', 'lost') "
            "ORDER BY next_follow_up_at, studio_name", (today,)).fetchall()
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

    def set_stage(self, name: str, stage: str, follow_up=None) -> dict:
        return set_stage(name, stage, follow_up)

    def add_note(self, name: str, text: str) -> dict:
        return add_note(name, text)

    def set_follow_up(self, name: str, when) -> dict:
        return set_follow_up(name, when)

    def list_by_stage(self, stage: str = None) -> list:
        return list_by_stage(stage)

    def due_follow_ups(self, today=None) -> list:
        return due_follow_ups(today)
