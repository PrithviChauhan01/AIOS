"""Email — AIOS's first EXTERNAL action, and the home of the confirm-before-acting
gate (Slice F).

Unlike reminders/jobs/documents (local writes), sending an email is irreversible
and leaves the machine. So this tool never sends on the first turn: it DRAFTS a
subject + body, stages it as pending for the session, and waits. Only an explicit
confirmation on a LATER turn ("yes" / "send it") actually sends. "no" / "cancel"
discards it; an edit revises it. This stage → confirm → execute → real-result
shape is the reusable gate every irreversible action will route through.

Send is real (Gmail SMTP via an App Password in config). The send result is
captured and reported truthfully — never a fabricated "sent".

Module state (`_PENDING`) is keyed by session_id, single-user/in-process, same as
the other tools' module-level state. Public API is the functions; EmailTool wraps
them for the registry with is_action=True.
"""

import json
import re
import smtplib
from email.message import EmailMessage

from config import Config
from tools.base import Tool

# session_id -> {"to", "to_name", "subject", "body"} pending an explicit confirm.
_PENDING = {}

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_client = None


def _groq():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


# ── Composition (her voice, via the cloud book) ──
_COMPOSE_PROMPT = """You draft an email on behalf of Sir (Prithvi). Write a crisp, warm, professional email.
Return ONLY a JSON object: {"subject": "...", "body": "..."}
- Subject: short and specific.
- Body: 2–5 short, natural sentences. Direct, no fluff, no invented facts, no "[placeholder]" tokens.
- Sign off as "Prithvi".
JSON only."""

_REVISE_PROMPT = """You revise a draft email per Sir's change request. Keep what works, apply the change.
Return ONLY a JSON object: {"subject": "...", "body": "..."} — sign off as "Prithvi". JSON only."""


def _parse_subject_body(text: str):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    d = json.loads(m.group(0))
    subject = (d.get("subject") or "").strip()
    body = (d.get("body") or "").strip()
    return (subject, body) if subject and body else None


def _compose(recipient: str, topic: str, instruction: str):
    """Compose subject+body for a new email. Falls back to a minimal draft if the
    book is unavailable, so a compose hiccup never blocks the gate."""
    try:
        r = _groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _COMPOSE_PROMPT},
                {"role": "user", "content":
                    f"Recipient: {recipient or 'the recipient'}\n"
                    f"What to say / intent: {topic or instruction}\n"
                    f"Full instruction from Sir: {instruction}"},
            ],
            max_tokens=400, temperature=0.4,
        )
        parsed = _parse_subject_body(r.choices[0].message.content)
        if parsed:
            return parsed
    except Exception as e:
        print(f"[email] compose fallback ({e})")
    subject = (topic or "Quick note").strip()[:78]
    greeting = f"Hi {recipient}," if recipient else "Hi,"
    body = f"{greeting}\n\n{topic or instruction}\n\nBest,\nPrithvi"
    return subject, body


def _compose_revise(current: dict, instruction: str):
    """Recompose an existing draft per a change request. On failure, leaves it as-is."""
    try:
        r = _groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _REVISE_PROMPT},
                {"role": "user", "content":
                    f"CURRENT SUBJECT: {current['subject']}\n"
                    f"CURRENT BODY:\n{current['body']}\n\n"
                    f"CHANGE REQUEST: {instruction}"},
            ],
            max_tokens=400, temperature=0.4,
        )
        parsed = _parse_subject_body(r.choices[0].message.content)
        if parsed:
            return parsed
    except Exception as e:
        print(f"[email] revise fallback ({e})")
    return current["subject"], current["body"]


# ── Real send (Gmail SMTP) ──
def _send_smtp(to: str, subject: str, body: str) -> dict:
    addr = (Config.GMAIL_ADDRESS or "").strip()
    pw = (Config.GMAIL_APP_PASSWORD or "").strip()
    if not addr or not pw:
        return {"ok": False, "error": "not_configured",
                "ask": "Email isn't set up yet, Sir — add GMAIL_ADDRESS and a "
                       "GMAIL_APP_PASSWORD to backend/.env and I'll send it."}
    msg = EmailMessage()
    msg["From"] = addr
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(addr, pw)
            s.send_message(msg)
        return {"ok": True, "to": to, "subject": subject}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Gate state machine ──
def make_draft(to: str, recipient: str, topic: str, instruction: str) -> dict:
    """Build (but do NOT send) a draft. Returns the draft, or a clarify if we don't
    have a real recipient address or anything to say."""
    to = (to or "").strip()
    if not _EMAIL_RE.fullmatch(to):
        found = _EMAIL_RE.search(instruction or "")  # address may be in the raw text
        to = found.group(0) if found else ""
    if not to:
        who = recipient or "the recipient"
        return {"ok": False, "needs": "to", "ask": f"What's {who}'s email address, Sir?"}
    if not (topic or instruction):
        return {"ok": False, "needs": "topic", "ask": "What should the email say, Sir?"}
    subject, body = _compose(recipient, topic, instruction)
    return {"ok": True, "to": to, "to_name": recipient or "", "subject": subject, "body": body}


def stage(session_id: str, draft: dict) -> None:
    _PENDING[session_id] = {"to": draft["to"], "to_name": draft.get("to_name", ""),
                            "subject": draft["subject"], "body": draft["body"]}


def has_pending(session_id: str) -> bool:
    return session_id in _PENDING


def get_pending(session_id: str) -> dict | None:
    return _PENDING.get(session_id)


def reconfirm(session_id: str) -> dict:
    d = _PENDING.get(session_id)
    if not d:
        return {"ok": False, "error": "no_pending"}
    return {"ok": True, "stage": "reconfirm", **d}


def revise(session_id: str, instruction: str) -> dict:
    d = _PENDING.get(session_id)
    if not d:
        return {"ok": False, "error": "no_pending"}
    d["subject"], d["body"] = _compose_revise(d, instruction)
    print("[email] draft revised -> awaiting confirm")
    return {"ok": True, "stage": "revise", **d}


def cancel(session_id: str) -> dict:
    d = _PENDING.pop(session_id, None)
    print("[email] cancelled")
    return {"ok": True, "cancelled": True, "had_draft": d is not None,
            "to": (d or {}).get("to"), "subject": (d or {}).get("subject")}


def confirm_send(session_id: str) -> dict:
    """Send the staged draft — ONLY called after explicit confirmation. On success
    clears pending; on failure keeps it so Sir can retry or cancel."""
    d = _PENDING.get(session_id)
    if not d:
        return {"ok": False, "error": "no_pending",
                "ask": "There's no email waiting to send, Sir."}
    res = _send_smtp(d["to"], d["subject"], d["body"])
    if res.get("ok"):
        _PENDING.pop(session_id, None)
        print(f"[email] SENT to={d['to']} subject={d['subject']!r}")
        return {"ok": True, "sent": True, "to": d["to"], "subject": d["subject"]}
    print(f"[email] send failed ({res.get('error')}) — draft kept")
    res.update({"sent": False, "pending": True, "to": d["to"], "subject": d["subject"]})
    res.setdefault("ask", "I couldn't send it, Sir — the draft's still here to retry or cancel.")
    return res


# ── Pool wrapper ──
class EmailTool(Tool):
    """Shared-pool handle for the email action tool. is_action=True; sending is the
    first irreversible/external action, gated by explicit confirmation."""

    name = "email"
    is_action = True

    async def fetch(self, query: str) -> dict:
        # Read-only view: is there a draft awaiting confirmation?
        try:
            results = []
            for sid, d in _PENDING.items():
                results.append(f"[{sid}] to {d['to']} — {d['subject']} (awaiting confirm)")
            return self._ok(query, results)
        except Exception as e:
            print(f"[tools:email] fetch failed: {e}")
            return self._empty(query)

    # Action passthroughs — module functions stay the source of truth.
    def draft(self, to, recipient, topic, instruction):
        return make_draft(to, recipient, topic, instruction)

    def stage(self, session_id, draft):
        return stage(session_id, draft)

    def has_pending(self, session_id):
        return has_pending(session_id)

    def confirm_send(self, session_id):
        return confirm_send(session_id)

    def cancel(self, session_id):
        return cancel(session_id)

    def revise(self, session_id, instruction):
        return revise(session_id, instruction)

    def reconfirm(self, session_id):
        return reconfirm(session_id)
