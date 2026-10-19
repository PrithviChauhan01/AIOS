"""Business — clients, projects and invoices (Slice B4). An ACTION tool.

Three tables (db/sqlite_init.BUSINESS_DDL): a client may come from a won lead
(lead_id), a project belongs to a client, an invoice to a project. Every write
resolves its target by name first — an unknown or ambiguous name is a clarify
result, never a guess — and returns the REAL row it wrote, so cognition only ever
confirms what actually happened.

Overdue is computed, not typed: every call here first flips a sent invoice whose
due date has passed to 'overdue' (and back, if a due date moves out), so the
status column is always true when read.

Deadlines and invoice due dates create reminders (tools/reminders) at 10:00 on the
day. A reminder that can't be set never undoes the write — it is reported.
"""

import os
import re
import sqlite3
from datetime import date, datetime, timedelta

from config import Config
from tools.base import Tool

PROJECT_STATUSES = ("planned", "active", "review", "done")
INVOICE_STATUSES = ("draft", "sent", "paid", "overdue")
_UNPAID = ("draft", "sent", "overdue")
_REMINDER_HOUR = "10:00"
_WEEK_DAYS = 7
_PDF_DIR = "outputs"  # same folder the dossier / call-sheet exports write to

_SCHEMA_READY: set = set()


def _db_path() -> str:
    return os.path.abspath(Config.SQLITE_PATH)


def _now() -> datetime:
    return datetime.now()


def _today() -> date:
    return _now().date()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    path = _db_path()
    if path not in _SCHEMA_READY:
        from db.sqlite_init import BUSINESS_DDL
        conn.executescript(BUSINESS_DDL)
        _SCHEMA_READY.add(path)
    _refresh_overdue(conn)
    return conn


def _refresh_overdue(conn) -> None:
    today = _today().isoformat()
    conn.execute("UPDATE invoices SET status = 'overdue' "
                 "WHERE status = 'sent' AND due_at < ?", (today,))
    conn.execute("UPDATE invoices SET status = 'sent' "
                 "WHERE status = 'overdue' AND due_at >= ?", (today,))
    conn.commit()


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match(rows: list, field: str, name: str) -> list:
    """Exact (normalized) match wins; else whole-word containment ("Verve" → "Verve Media")."""
    want = _norm(name)
    if not want:
        return []
    exact = [r for r in rows if _norm(r[field]) == want]
    return exact or [r for r in rows if f" {want} " in f" {_norm(r[field])} "]


def _as_date(when) -> str:
    """A date phrase / ISO date → 'YYYY-MM-DD'. Raises ValueError if unparseable."""
    if isinstance(when, date) and not isinstance(when, datetime):
        return when.isoformat()
    from tools.reminders import parse_due
    return parse_due(when, now=_now()).date().isoformat()


def _set_reminder(title: str, day: str) -> dict | None:
    """A reminder at 10:00 on `day`, or None when the day has already passed (a past
    reminder would just fire at once)."""
    if not day or day < _today().isoformat():
        return None
    try:
        from tools.reminders import set_reminder
        return set_reminder(title, f"{day} {_REMINDER_HOUR}")
    except Exception as e:
        print(f"[business] reminder failed: {e}")
        return {"ok": False, "error": str(e)}


def money(amount, currency: str = "INR") -> str:
    return f"{currency} {amount:,.2f}" if amount is not None else "—"


# ── Clients ──
def _clients(conn) -> list:
    return [dict(r) for r in conn.execute("SELECT * FROM clients ORDER BY name")]


def resolve_client(conn, name: str) -> dict:
    hits = _match(_clients(conn), "name", name)
    if len(hits) == 1:
        return {"ok": True, "client": hits[0]}
    if not hits:
        return {"ok": False, "needs": "client", "not_found": name,
                "ask": f"I have no client called \"{name}\", Sir."}
    return {"ok": False, "needs": "client",
            "ask": f"\"{name}\" matches {', '.join(c['name'] for c in hits[:5])} — which one, Sir?"}


def add_client(name: str, contact: str = None, lead_id: int = None, notes: str = None) -> dict:
    name = " ".join((name or "").split())
    if not name:
        return {"ok": False, "needs": "name", "ask": "What's the client's name, Sir?"}
    conn = _conn()
    try:
        existing = [c for c in _clients(conn) if _norm(c["name"]) == _norm(name)]
        if existing:
            return {"ok": False, "needs": "name", "exists": True, "client": existing[0],
                    "ask": f"{existing[0]['name']} is already a client, Sir."}
        cur = conn.execute(
            "INSERT INTO clients (name, contact, lead_id, notes) VALUES (?, ?, ?, ?)",
            (name, contact or None, lead_id, notes or None))
        conn.commit()
        client = dict(conn.execute("SELECT * FROM clients WHERE id = ?",
                                   (cur.lastrowid,)).fetchone())
        print(f"[business] WRITE client #{client['id']}")
    finally:
        conn.close()
    return {"ok": True, "client": client}


def client_for_lead(lead_id: int) -> dict | None:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM clients WHERE lead_id = ?", (lead_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def client_from_lead(lead_id: int) -> dict:
    """Create the client for a won lead — name, contact and a back-reference from the
    lead row. Idempotent: a lead already turned into a client returns that client."""
    existing = client_for_lead(lead_id)
    if existing:
        return {"ok": True, "client": existing, "already": True}
    conn = _conn()
    try:
        lead = conn.execute("SELECT id, studio_name, location, contact FROM leads WHERE id = ?",
                            (lead_id,)).fetchone()
    finally:
        conn.close()
    if lead is None:
        return {"ok": False, "error": f"lead #{lead_id} no longer exists"}
    contact = lead["contact"] if lead["contact"] and lead["contact"] != "not found" else None
    where = f" ({lead['location']})" if lead["location"] else ""
    return add_client(lead["studio_name"], contact, lead_id,
                      f"From lead #{lead_id}{where}")


def list_clients() -> list:
    """Every client with its project count and unpaid total (per currency)."""
    conn = _conn()
    try:
        clients = _clients(conn)
        for c in clients:
            c["projects"] = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE client_id = ?", (c["id"],)).fetchone()[0]
            c["unpaid"] = [dict(r) for r in conn.execute(
                "SELECT i.currency, SUM(i.amount) AS total FROM invoices i "
                "JOIN projects p ON p.id = i.project_id WHERE p.client_id = ? "
                f"AND i.status IN {_UNPAID} GROUP BY i.currency", (c["id"],))]
    finally:
        conn.close()
    return clients


# ── Projects ──
def _projects(conn, client_id: int = None) -> list:
    sql = ("SELECT p.*, c.name AS client FROM projects p JOIN clients c ON c.id = p.client_id")
    params = ()
    if client_id is not None:
        sql += " WHERE p.client_id = ?"
        params = (client_id,)
    return [dict(r) for r in conn.execute(sql + " ORDER BY p.id", params)]


def resolve_project(conn, title: str) -> dict:
    hits = _match(_projects(conn), "title", title)
    if len(hits) == 1:
        return {"ok": True, "project": hits[0]}
    if not hits:
        return {"ok": False, "needs": "project", "not_found": title,
                "ask": f"I have no project called \"{title}\", Sir."}
    names = ", ".join(f"{p['title']} ({p['client']})" for p in hits[:5])
    return {"ok": False, "needs": "project", "ask": f"\"{title}\" matches {names} — which one, Sir?"}


def _deadline_reminder(project: dict) -> dict | None:
    return _set_reminder(f"Deadline: {project['title']} ({project['client']})",
                         project.get("deadline"))


def _resplit_title(conn, title: str, client: str, not_found: dict) -> tuple:
    """The parser splits "add project X for Y" at the FIRST " for ", so a title that
    itself contains "for" ("Content for socials for Verve Media") arrives as
    title="Content", client="socials for Verve Media". Rejoin and try every " for "
    split; use the one whose remainder is EXACTLY one existing client. None or
    several → the original not-found clarify, never a guess."""
    full = f"{title} for {client}"
    names = {_norm(c["name"]): c for c in _clients(conn)}
    hits = []
    for m in re.finditer(r"\s+for\s+", full, re.IGNORECASE):
        head, tail = full[:m.start()].strip(), full[m.end():].strip()
        if head and _norm(tail) in names:
            hits.append((head, names[_norm(tail)]))
    if len(hits) == 1:
        return hits[0][0], {"ok": True, "client": hits[0][1]}
    return title, not_found


def add_project(client: str, title: str, deadline=None, fee=None, notes: str = None) -> dict:
    title = " ".join((title or "").split())
    if not title:
        return {"ok": False, "needs": "title", "ask": "What's the project called, Sir?"}
    try:
        day = _as_date(deadline) if deadline else None
    except ValueError:
        return {"ok": False, "needs": "deadline",
                "ask": f"I couldn't read \"{deadline}\" as a date, Sir."}
    conn = _conn()
    try:
        found = resolve_client(conn, client)
        if not found["ok"] and found.get("not_found"):
            title, found = _resplit_title(conn, title, client, found)
        if not found["ok"]:
            return found
        cl = found["client"]
        cur = conn.execute(
            "INSERT INTO projects (client_id, title, deadline, fee, notes) VALUES (?, ?, ?, ?, ?)",
            (cl["id"], title, day, fee, notes or None))
        conn.commit()
        project = {**dict(conn.execute("SELECT * FROM projects WHERE id = ?",
                                       (cur.lastrowid,)).fetchone()), "client": cl["name"]}
        print(f"[business] WRITE project #{project['id']} client=#{cl['id']} deadline={day}")
    finally:
        conn.close()
    return {"ok": True, "project": project, "reminder": _deadline_reminder(project)}


def update_project(title: str, status: str = None, deadline=None) -> dict:
    if status is None and deadline is None:
        return {"ok": False, "needs": "change", "ask": "What should I change on it, Sir?"}
    status = (status or "").strip().lower() or None
    if status and status not in PROJECT_STATUSES:
        return {"ok": False, "needs": "status",
                "ask": f"Which status, Sir — {', '.join(PROJECT_STATUSES)}?"}
    try:
        day = _as_date(deadline) if deadline else None
    except ValueError:
        return {"ok": False, "needs": "deadline",
                "ask": f"I couldn't read \"{deadline}\" as a date, Sir."}
    conn = _conn()
    try:
        found = resolve_project(conn, title)
        if not found["ok"]:
            return found
        before = found["project"]
        conn.execute("UPDATE projects SET status = ?, deadline = ? WHERE id = ?",
                     (status or before["status"], day or before["deadline"], before["id"]))
        conn.commit()
        project = next(p for p in _projects(conn) if p["id"] == before["id"])
        print(f"[business] WRITE project #{project['id']} status={project['status']} "
              f"deadline={project['deadline']}")
    finally:
        conn.close()
    reminder = _deadline_reminder(project) if day and project["status"] != "done" else None
    return {"ok": True, "project": project, "from_status": before["status"],
            "from_deadline": before["deadline"], "reminder": reminder}


# ── Invoices ──
def _invoices(conn, where: str = "", params=()) -> list:
    sql = ("SELECT i.*, p.title AS project, c.name AS client, c.contact AS client_contact "
           "FROM invoices i JOIN projects p ON p.id = i.project_id "
           "JOIN clients c ON c.id = p.client_id")
    return [dict(r) for r in conn.execute(f"{sql} {where}", params)]


def _next_number(conn, year: int) -> str:
    rows = conn.execute("SELECT number FROM invoices WHERE number LIKE ?",
                        (f"INV-{year}-%",)).fetchall()
    seq = max((int(r[0].rsplit("-", 1)[1]) for r in rows), default=0) + 1
    return f"INV-{year}-{seq:03d}"


def _invoice_project(conn, target: str) -> dict:
    """An invoice target names a project, or a client with exactly one open project."""
    proj = resolve_project(conn, target)
    if proj["ok"]:
        return proj
    cl = resolve_client(conn, target)
    if not cl["ok"]:
        if proj.get("not_found") and cl.get("not_found"):
            return {"ok": False, "needs": "project",
                    "ask": f"I have no client or project called \"{target}\", Sir."}
        return cl if not cl.get("not_found") else proj
    projects = [p for p in _projects(conn, cl["client"]["id"]) if p["status"] != "done"] \
        or _projects(conn, cl["client"]["id"])
    if len(projects) == 1:
        return {"ok": True, "project": projects[0]}
    if not projects:
        return {"ok": False, "needs": "project",
                "ask": f"{cl['client']['name']} has no projects yet, Sir — add one first."}
    names = ", ".join(p["title"] for p in projects[:5])
    return {"ok": False, "needs": "project",
            "ask": f"{cl['client']['name']} has several projects ({names}) — which one, Sir?"}


def create_invoice(target: str, amount: float = None, due=None, currency: str = "INR") -> dict:
    """A draft invoice for a project (or a client's only open project). Amount defaults
    to the project fee; due date to INVOICE_DUE_DAYS after today."""
    try:
        due_day = _as_date(due) if due else \
            (_today() + timedelta(days=Config.INVOICE_DUE_DAYS)).isoformat()
    except ValueError:
        return {"ok": False, "needs": "due", "ask": f"I couldn't read \"{due}\" as a date, Sir."}
    conn = _conn()
    try:
        found = _invoice_project(conn, target)
        if not found["ok"]:
            return found
        project = found["project"]
        amt = amount if amount is not None else project.get("fee")
        if not amt or amt <= 0:
            return {"ok": False, "needs": "amount",
                    "ask": f"How much should the invoice for {project['title']} be, Sir?"}
        issued = _today()
        number = _next_number(conn, issued.year)
        conn.execute(
            "INSERT INTO invoices (project_id, number, amount, currency, issued_at, due_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project["id"], number, float(amt), (currency or "INR").upper(),
             issued.isoformat(), due_day))
        conn.commit()
        invoice = _invoices(conn, "WHERE i.number = ?", (number,))[0]
        print(f"[business] WRITE invoice {number} project=#{project['id']}")
    finally:
        conn.close()
    reminder = _set_reminder(
        f"Invoice {number} due — {invoice['client']} ({money(invoice['amount'], invoice['currency'])})",
        due_day)
    return {"ok": True, "invoice": invoice, "reminder": reminder}


def resolve_invoice(conn, target: str, unpaid_only: bool = False) -> dict:
    """By number (INV-2026-001), else the client's / project's most recent invoice
    (most recent unpaid one when unpaid_only)."""
    m = re.search(r"INV-\d{4}-\d{3,}", target or "", re.IGNORECASE)
    if m:
        hits = _invoices(conn, "WHERE i.number = ?", (m.group(0).upper(),))
        if hits:
            return {"ok": True, "invoice": hits[0]}
        return {"ok": False, "needs": "invoice", "ask": f"There's no invoice {m.group(0).upper()}, Sir."}
    rows = _invoices(conn, "ORDER BY i.issued_at DESC, i.id DESC")
    if unpaid_only:
        rows = [r for r in rows if r["status"] in _UNPAID]
    hits = _match(rows, "client", target) or _match(rows, "project", target)
    if not hits:
        return {"ok": False, "needs": "invoice",
                "ask": f"I have no {'unpaid ' if unpaid_only else ''}invoice for \"{target}\", Sir."}
    return {"ok": True, "invoice": hits[0]}


def mark_invoice(target: str, status: str) -> dict:
    status = (status or "").lower()
    if status not in ("sent", "paid"):
        return {"ok": False, "needs": "status", "ask": "Mark it sent or paid, Sir?"}
    conn = _conn()
    try:
        found = resolve_invoice(conn, target, unpaid_only=True)
        if not found["ok"]:
            return found
        before = found["invoice"]
        if before["status"] == status:
            return {"ok": False, "needs": "status", "invoice": before,
                    "ask": f"{before['number']} is already marked {status}, Sir."}
        new = status
        if status == "sent" and before["due_at"] < _today().isoformat():
            new = "overdue"   # sent after its due date — that IS overdue
        conn.execute("UPDATE invoices SET status = ? WHERE id = ?", (new, before["id"]))
        conn.commit()
        invoice = _invoices(conn, "WHERE i.id = ?", (before["id"],))[0]
        print(f"[business] WRITE invoice {invoice['number']} {before['status']}->{new}")
    finally:
        conn.close()
    return {"ok": True, "invoice": invoice, "from_status": before["status"]}


def unpaid_invoices() -> list:
    conn = _conn()
    try:
        return _invoices(conn, f"WHERE i.status IN {_UNPAID} ORDER BY i.due_at, i.number")
    finally:
        conn.close()


def due_this_week(today: date = None) -> dict:
    """Open project deadlines and unpaid invoice due dates up to 7 days out —
    including anything already past due — soonest first."""
    today = today or _today()
    horizon = (today + timedelta(days=_WEEK_DAYS)).isoformat()
    conn = _conn()
    try:
        projects = [p for p in _projects(conn)
                    if p["deadline"] and p["deadline"] <= horizon and p["status"] != "done"]
        invoices = _invoices(conn, f"WHERE i.status IN {_UNPAID} AND i.due_at <= ? "
                                   "ORDER BY i.due_at", (horizon,))
    finally:
        conn.close()
    projects.sort(key=lambda p: p["deadline"])
    return {"projects": projects, "invoices": invoices, "today": today.isoformat(),
            "until": horizon}


def invoice_pdf(target: str) -> dict:
    """Render an invoice (by number, else the client's latest unpaid one, else its
    latest) to PDF via the reportlab export. Returns the file path."""
    conn = _conn()
    try:
        found = resolve_invoice(conn, target, unpaid_only=True)
        if not found["ok"] and not re.search(r"INV-\d{4}", target or "", re.IGNORECASE):
            found = resolve_invoice(conn, target)
    finally:
        conn.close()
    if not found["ok"]:
        return found
    inv = found["invoice"]
    from tools.pdf_gen import make_invoice_pdf
    path = make_invoice_pdf(
        number=inv["number"], business=Config.BUSINESS_NAME,
        business_contact=Config.BUSINESS_CONTACT, client=inv["client"],
        client_contact=inv.get("client_contact") or "", item=inv["project"],
        amount=inv["amount"], currency=inv["currency"], issued=inv["issued_at"],
        due=inv["due_at"], status=inv["status"], out_dir=_PDF_DIR)
    print(f"[business] invoice PDF {inv['number']} -> {path}")
    return {"ok": True, "invoice": inv, "file_path": path}


# ── Pool wrapper ──
class BusinessTool(Tool):
    """Shared-pool handle. is_action=True: hidden in public mode by the registry."""

    name = "business"
    is_action = True

    async def fetch(self, query: str) -> dict:
        try:
            return self._ok(query, [f"{c['name']} — {c['projects']} project(s)"
                                    for c in list_clients()])
        except Exception as e:
            print(f"[tools:business] fetch failed: {e}")
            return self._empty(query)

    add_client = staticmethod(add_client)
    client_from_lead = staticmethod(client_from_lead)
    client_for_lead = staticmethod(client_for_lead)
    list_clients = staticmethod(list_clients)
    add_project = staticmethod(add_project)
    update_project = staticmethod(update_project)
    create_invoice = staticmethod(create_invoice)
    mark_invoice = staticmethod(mark_invoice)
    unpaid_invoices = staticmethod(unpaid_invoices)
    due_this_week = staticmethod(due_this_week)
    invoice_pdf = staticmethod(invoice_pdf)
