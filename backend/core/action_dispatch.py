"""Action dispatch — the ONE stage that actually EXECUTES a tool.

Everything else in the spine generates text: triage classifies, teachers research,
cognition speaks. None of them write state. Without this stage "remind me to
stretch in a minute" is spoken back ("On it, Sir") but set_reminder() is never
called and nothing lands in the DB.

This stage sits right after triage. It:
  1. cheaply GATES on whether the line is even a reminder command (a keyword
     pre-check, so the extractor LLM never runs on ordinary chat),
  2. EXTRACTS strict JSON intent+args with one cheap LOCAL call (ollama llama3.2,
     same model triage uses — free, private, never touches the cloud), hard
     failing safe to {"action": "none"} on any bad/again-missing JSON,
  3. EXECUTES the real RemindersTool through the registry, respecting is_action,
  4. returns the REAL result dict so cognition can confirm truthfully — never a
     fabricated "Done, Sir".

Reminders was the first action tool; the job-application log (Slice C) reuses this
exact gate → extract → execute → real-result shape. Each tool brings its own gate
(a cheap keyword pre-check) and its own strict-JSON extract prompt; the detected
action carries a "tool" key so run_action / action_material / fn_name dispatch to
the right one. email/calendar will plug in the same way.
"""

import json
import re

import ollama

from tools.registry import get_tool, is_action_tool

# Cheap gate: spoken reminder commands essentially always contain "remind"
# (remind/reminds/reminder/reminders) — set, list, complete and delete phrasings
# alike ("remind me…", "list my reminders", "delete the X reminder", "mark the X
# reminder done"). Gating here keeps the extractor LLM off every non-reminder
# line; a rare false gate (e.g. "that reminds me…") just costs one local call and
# falls through to none → normal routing.
_GATE = re.compile(r"\bremind", re.IGNORECASE)

# Jobs gate: logging/listing a job application. Requires a log/track/save/record
# verb paired with a job/application noun, OR an explicit "applied to/for/at" —
# specific enough to keep the extractor off ordinary chat. A rare false gate just
# costs one local call and falls through to none → normal routing.
_JOBS_GATE = re.compile(
    r"\b(log|track|save|record|add)\b.*\b(job|application|applied|role|position|gig)\b"
    r"|\bapplied\s+(to|for|at)\b"
    r"|\b(jobs?|applications?)\s+(i|i've|i have)\s+applied\b",
    re.IGNORECASE,
)

# Action verb → the module function name, for an honest, greppable orch log line.
_FN_NAME = {
    "set": "set_reminder",
    "list": "list_reminders",
    "complete": "complete_reminder",
    "delete": "delete_reminder",
}
_JOBS_FN_NAME = {
    "log": "log_application",
    "list": "list_applications",
}

# Documents gate: saving/listing/retrieving/deleting a stored document. Catches
# "save this document/file/pdf", "what documents do I have", "pull up my resume",
# "delete the X document". Broad-ish; the extractor returns none for non-commands.
_DOCS_GATE = re.compile(
    r"\bdocuments?\b|\bpull up\b"
    r"|\b(save|store|add|keep|file)\b.*\b(file|pdf|docx?|resume|cv|cert\w*|contract|note|notes|id|card|statement)\b"
    r"|\b(what|which|list|show)\b.*\b(files?|docs?|documents?)\b"
    r"|\b(get|open|show|find|fetch)\b.*\bmy\b.*\b(resume|cv|cert\w*|contract|notes?|file|doc\w*|card|statement)\b"
    r"|\bdelete\b.*\b(document|file|doc)\b",
    re.IGNORECASE,
)

_DOCS_FN_NAME = {
    "save": "save_document",
    "list": "list_documents",
    "get": "get_document",
    "delete": "delete_document",
}

_EXTRACT_PROMPT = """You convert a short spoken command about REMINDERS into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "set|list|complete|delete|none", "title": "<the bare task to be reminded of, or empty>", "due_at": "<the time exactly as the user said it, e.g. 'in a minute','tomorrow 9am','at 6pm', or empty>", "repeat": "<cadence as said, e.g. 'daily','every 20 minutes', or empty>", "id": <reminder number if the user gave one, else null>}

Rules:
- set      = create a NEW reminder ("remind me to X at Y", "set a reminder to X").
- list     = show reminders ("what are my reminders", "list my reminders").
- complete = mark an existing one done ("mark the X reminder done", "complete X").
- delete   = remove an existing one ("delete/cancel/remove the X reminder").
- none     = anything that is NOT actually a reminder command.
- title: the bare task ONLY. Strip "remind me to", the time, and repeat words.
  "remind me to stretch in a minute" -> title "stretch", due_at "in a minute".
- due_at: copy the user's time phrasing verbatim. Do NOT compute or invent a date.
- If unsure whether it's a command at all, use "none".
JSON only."""


def _extract(message: str) -> dict | None:
    """Local strict-JSON intent extraction. Returns the parsed dict, or None on
    anything that isn't a usable reminder command (bad JSON, action 'none', or an
    unknown action). Hard fail-safe: never raises, never guesses."""
    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT},
                {"role": "user", "content": message},
            ],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
    except Exception as e:
        print(f"[orch] action extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    return {
        "action": action,
        "title": (parsed.get("title") or "").strip(),
        "due_at": (parsed.get("due_at") or "").strip(),
        "repeat": (parsed.get("repeat") or "").strip() or None,
        "id": parsed.get("id"),
    }


_JOBS_EXTRACT_PROMPT = """You convert a short command about LOGGING A JOB APPLICATION into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "log|list|none", "company": "<employer name, or empty>", "role": "<job title, or empty>", "resume": "engineering|design|", "status": "<e.g. applied, interviewing, or empty>", "url": "<url, or empty>", "follow_up_at": "<follow-up time exactly as said, e.g. 'in a week','next Monday', or empty>"}

Rules:
- log  = record a NEW application Sir made ("log my application to X for Y", "I applied to X as Y", "track this application").
- list = show logged applications ("what jobs have I applied to", "list my applications").
- none = anything that is NOT logging or listing a job application.
- company: the employer. role: the job title. Strip filler like "my application to".
- resume: ONLY if Sir explicitly says which resume he sent (engineering or design); otherwise empty.
- follow_up_at: copy the time phrasing verbatim. Do NOT compute or invent a date.
- If unsure whether it's a job-log command at all, use "none".
JSON only."""


def _extract_jobs(message: str) -> dict | None:
    """Local strict-JSON extraction for a job-application command. Returns the
    parsed dict, or None for anything that isn't a usable log/list command. Same
    hard fail-safe contract as _extract."""
    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _JOBS_EXTRACT_PROMPT},
                {"role": "user", "content": message},
            ],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
    except Exception as e:
        print(f"[orch] jobs extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _JOBS_FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    return {
        "tool": "jobs",
        "action": action,
        "company": (parsed.get("company") or "").strip(),
        "role": (parsed.get("role") or "").strip(),
        "resume": (parsed.get("resume") or "").strip().lower() or None,
        "status": (parsed.get("status") or "").strip() or None,
        "url": (parsed.get("url") or "").strip() or None,
        "follow_up_at": (parsed.get("follow_up_at") or "").strip() or None,
    }


_DOCS_EXTRACT_PROMPT = """You convert a short command about a DOCUMENT STORE into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "save|list|get|delete|none", "path": "<file path to ingest, or empty>", "name": "<document name/title, or empty>", "doc_type": "<e.g. resume, certificate, id, contract, notes, or empty>", "tags": "<comma-separated tags, or empty>", "id": <document number if the user gave one, else null>}

Rules:
- save   = store a document FROM A FILE PATH ("save this document at C:/x.pdf", "store my resume /path/cv.pdf as Resume").
- list   = show stored documents ("what documents do I have", "list my files").
- get    = retrieve/pull up a stored document ("pull up my resume", "get my offer letter", "show my PAN doc").
- delete = remove a stored document ("delete the resume document", "remove document 3").
- none   = anything that is NOT a document-store command.
- path: the file path for save, exactly as given. name/doc_type: copy what the user said; leave empty if not stated.
- For get/delete, put the document's name in "name" (and "id" if a number was given).
- If unsure whether it's a document command at all, use "none".
JSON only."""


def _extract_docs(message: str) -> dict | None:
    """Local strict-JSON extraction for a document-store command. Returns the
    parsed dict, or None for anything that isn't a usable command. Same hard
    fail-safe contract as _extract."""
    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _DOCS_EXTRACT_PROMPT},
                {"role": "user", "content": message},
            ],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
    except Exception as e:
        print(f"[orch] docs extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _DOCS_FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    return {
        "tool": "documents",
        "action": action,
        "path": (parsed.get("path") or "").strip(),
        "name": (parsed.get("name") or "").strip(),
        "doc_type": (parsed.get("doc_type") or "").strip() or None,
        "tags": (parsed.get("tags") or "").strip() or None,
        "id": parsed.get("id"),
    }


def detect_action(message: str) -> dict | None:
    """Cheap per-tool gate, then local extraction. Returns the intent dict (tagged
    with its "tool") for a real action command, or None for ordinary chat (which
    then routes normally). Gates are tried in order; a line that gates but fails
    extraction can still fall through to the next tool's gate."""
    msg = message or ""
    if _GATE.search(msg):
        action = _extract(msg)
        if action is not None:
            action["tool"] = "reminders"
            return action
    if _JOBS_GATE.search(msg):
        action = _extract_jobs(msg)
        if action is not None:
            return action
    if _DOCS_GATE.search(msg):
        return _extract_docs(msg)
    return None


# ── Execution ──
def _resolve_id(action: dict, items: list) -> int | None:
    """Resolve which reminder a complete/delete targets. Prefer an explicit id;
    otherwise match the title against pending reminders, but only when exactly one
    matches (ambiguity → caller asks Sir which one)."""
    if action.get("id") is not None:
        try:
            return int(action["id"])
        except (TypeError, ValueError):
            pass
    title = (action.get("title") or "").strip().lower()
    if title:
        matches = [r for r in items if title in r["title"].lower()]
        if len(matches) == 1:
            return matches[0]["id"]
    return None


def run_action(action: dict) -> dict:
    """Execute the detected action against the REAL tool via the registry,
    respecting is_action. Dispatches by the action's "tool" tag. Returns the actual
    tool result dict (success, the needs/ask clarify path, or a not-found). Never
    raises."""
    if action.get("tool") == "jobs":
        return _run_jobs(action)
    if action.get("tool") == "documents":
        return _run_documents(action)
    return _run_reminders(action)


def _run_documents(action: dict) -> dict:
    """Execute a document-store action against the REAL DocumentsTool via the registry."""
    name = "documents"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "save":
            return tool.save(action.get("path"), action.get("name"),
                             action.get("doc_type"), action.get("tags"))
        if verb == "list":
            return {"ok": True, "items": tool.list(action.get("doc_type"), action.get("tags"))}
        if verb == "get":
            # Prefer the spoken NAME over an id: the local extractor sometimes
            # hallucinates an id (e.g. 1) when the user only named a document
            # ("my id document"), which silently returned the wrong row.
            ident = action.get("name") or action.get("id")
            return tool.get(ident)
        if verb == "delete":
            ident = action.get("name") or action.get("id")
            target = tool.get(ident)  # resolve to a concrete id (and surface ambiguity)
            if not target.get("ok"):
                return target  # needs/which or not-found — relayed truthfully
            return tool.delete(target["id"])
    except Exception as e:
        print(f"[orch] documents action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown documents verb {verb}"}


def _run_jobs(action: dict) -> dict:
    """Execute a job-application action against the REAL JobsTool via the registry."""
    name = "jobs"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "log":
            return tool.log(
                action.get("company"), action.get("role"), action.get("resume"),
                action.get("status") or "applied", action.get("url"),
                action.get("follow_up_at"),
            )
        if verb == "list":
            return {"ok": True, "items": tool.list()}
    except Exception as e:
        print(f"[orch] jobs action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown jobs verb {verb}"}


def _run_reminders(action: dict) -> dict:
    """Execute the detected action against the REAL RemindersTool via the registry,
    respecting is_action. Returns the actual tool result dict (success, the
    needs/ask clarify path, or a not-found). Never raises."""
    name = "reminders"
    if not is_action_tool(name):  # guard: only state-writing tools execute here
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]

    try:
        if verb == "set":
            return tool.set(action.get("title"), action.get("due_at"), action.get("repeat"))

        if verb == "list":
            return {"ok": True, "items": tool.list(pending_only=True)}

        # complete / delete need a concrete row — resolve from id or unique title.
        items = tool.list(pending_only=True)
        rid = _resolve_id(action, items)
        if rid is None:
            verbed = "complete" if verb == "complete" else "remove"
            return {"ok": False, "needs": "which",
                    "ask": f"Which reminder should I {verbed}, Sir?"}
        ok = tool.complete(rid) if verb == "complete" else tool.delete(rid)
        return {"ok": bool(ok), "id": rid, "found": bool(ok)}
    except Exception as e:
        print(f"[orch] action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}


def fn_name(action: dict) -> str:
    """Module-function name for an action, for logging ('set' → 'set_reminder',
    'log' → 'log_application'). Dispatches by the action's "tool" tag."""
    verb = action.get("action")
    if action.get("tool") == "jobs":
        return _JOBS_FN_NAME.get(verb, verb)
    if action.get("tool") == "documents":
        return _DOCS_FN_NAME.get(verb, verb)
    return _FN_NAME.get(verb, verb)


def action_material(action: dict, res: dict) -> str:
    """Render the REAL tool result into a text block for cognition. This is the
    truth she confirms from — she must not embellish past what it states.
    Dispatches by the action's "tool" tag."""
    if action.get("tool") == "jobs":
        return _jobs_material(action.get("action"), res)
    if action.get("tool") == "documents":
        return _documents_material(action.get("action"), res)
    return _reminder_material(action.get("action"), res)


def _documents_material(verb: str, res: dict) -> str:
    """Render a document-store action result for cognition. A secret/vault doc's
    content is included for HER to relay, but the orchestrator forces such a turn
    local so it never reaches a cloud provider."""
    if verb == "save":
        if res.get("ok"):
            where = ("the ENCRYPTED LOCAL VAULT (secret — never sent to the cloud)"
                     if res["storage"] == "vault"
                     else f"the document store ({res['chunk_count']} chunks, retrievable)")
            return (
                "ACTION EXECUTED — you just SAVED a document. This actually happened, "
                f"the write is done. Details: name={res['name']!r}, "
                f"type={res.get('doc_type') or 'unspecified'}, tier={res['tier']}, "
                f"stored in {where}. Confirm to Sir minimally, naming the document and — "
                "if it went to the vault — that it's kept locally/private."
            )
        return (
            "ACTION NOT DONE — the document was NOT saved "
            f"({res.get('needs') or res.get('error')}). Do NOT claim it's saved. Put "
            f"this to Sir: {res.get('ask')}"
        )

    if verb == "list":
        items = res.get("items", [])
        if not items:
            return ("ACTION EXECUTED — you checked the document store and it is EMPTY. "
                    "Tell Sir he hasn't saved any documents yet.")
        lines = "\n".join(
            f"#{r['id']} {r['name']} [{r['doc_type'] or 'doc'}] "
            f"({r['tier']}/{r['storage']})"
            + (f" — tags: {r['tags']}" if r.get("tags") else "")
            for r in items
        )
        return ("ACTION EXECUTED — these are Sir's stored documents. Present them "
                "cleanly:\n" + lines)

    if verb == "get":
        if not res.get("ok"):
            return (f"ACTION NOT DONE — {res.get('ask')} Do not invent a document.")
        secret_note = (" This document is SECRET/local — relay it to Sir but treat it as "
                       "private." if res.get("tier") == "secret" else "")
        return (
            f"ACTION EXECUTED — you pulled up Sir's document {res['name']!r} "
            f"(type={res.get('doc_type') or 'unspecified'}, tier={res['tier']}).{secret_note} "
            "Present its content usefully for what he asked — summarize or surface the "
            "relevant part, don't dump it raw. CONTENT:\n" + (res.get("content") or "")[:6000]
        )

    # delete
    if res.get("ok"):
        return (f"ACTION EXECUTED — document #{res.get('id')} ({res.get('name')!r}) was "
                "deleted everywhere it lived. This really happened. Confirm to Sir, minimally.")
    if res.get("needs"):
        return f"ACTION NOT DONE — {res.get('ask')} Do not claim it's deleted."
    return ("ACTION NOT DONE — no matching document was found, so nothing changed. "
            "Tell Sir you couldn't find that one.")


def _jobs_material(verb: str, res: dict) -> str:
    """Render a job-application action result for cognition."""
    if verb == "log":
        if res.get("ok"):
            return (
                "ACTION EXECUTED — you just LOGGED a job application to the database. "
                "This actually happened, the write is done. Details: "
                f"company={res['company']!r}, role={res['role']!r}, "
                f"resume_used={res['resume_used']!r}, status={res['status']!r}"
                + (f", follow_up={res['follow_up_at']!r}" if res.get("follow_up_at") else "")
                + ". Confirm to Sir minimally, naming the company, role and which resume."
            )
        # clarify path — nothing was written
        return (
            "ACTION NOT DONE — no application was saved because a detail is missing "
            f"({res.get('needs')}). Do NOT claim it's logged. Put this to Sir: "
            f"{res.get('ask')}"
        )

    # list
    items = res.get("items", [])
    if not items:
        return ("ACTION EXECUTED — you checked the applications log and it is EMPTY. "
                "Tell Sir he hasn't logged any applications yet.")
    lines = "\n".join(
        f"#{r['id']} {r['company']} — {r['role']} [{r['resume_used']}] ({r['status']})"
        + (f", follow-up {r['follow_up_at']}" if r.get("follow_up_at") else "")
        for r in items
    )
    return ("ACTION EXECUTED — these are Sir's logged applications. Present them "
            "cleanly:\n" + lines)


def _reminder_material(verb: str, res: dict) -> str:
    """Render a reminder action result for cognition. This is the truth she confirms
    from — she must not embellish past what it states."""
    if verb == "set":
        if res.get("ok"):
            return (
                "ACTION EXECUTED — you just SAVED a reminder to the database. This "
                "actually happened, the write is done. Details: "
                f"task={res['title']!r}, due_at={res['due_at']!r}, "
                f"repeat={res.get('repeat') or 'none'}. Confirm to Sir, naming what "
                "and roughly when in natural language."
            )
        # clarify path — nothing was written
        return (
            "ACTION NOT DONE — no reminder was saved because a detail is missing "
            f"({res.get('needs')}). Do NOT claim it's set. Put this to Sir: "
            f"{res.get('ask')}"
        )

    if verb == "list":
        items = res.get("items", [])
        if not items:
            return ("ACTION EXECUTED — you checked the reminders list and it is EMPTY. "
                    "Tell Sir there are none pending.")
        lines = "\n".join(
            f"#{r['id']} {r['title']} — due {r['due_at']}"
            + (f" (repeats {r['repeat']})" if r.get("repeat") else "")
            for r in items
        )
        return ("ACTION EXECUTED — these are Sir's current pending reminders. Present "
                "them cleanly:\n" + lines)

    # complete / delete
    done_word = "marked done" if verb == "complete" else "deleted"
    if res.get("ok"):
        return (f"ACTION EXECUTED — reminder #{res.get('id')} was {done_word}. This "
                "really happened. Confirm to Sir, minimally.")
    if res.get("needs"):
        return f"ACTION NOT DONE — {res.get('ask')} Do not claim it's {done_word}."
    return ("ACTION NOT DONE — no matching reminder was found, so nothing changed. "
            "Tell Sir you couldn't find that one.")
