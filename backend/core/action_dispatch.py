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

import hashlib
import json
import re
import threading
import time

from core.net import ollama_chat
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

# Bulk-clear gate for reminders — checked BEFORE the LLM extractor even runs
# (deterministic, no ollama round trip, so it can never misfire the way a model
# occasionally does). "empty all the reminders" / "remove all the reminders" /
# "clear my reminders" all match. Plural "reminders" (or an explicit "all") is
# required so a SINGLE-item command ("delete the reminder about calling mom",
# "cancel the reminder for the dentist") never gets read as a wipe.
_CLEAR_RE = re.compile(
    r"\b(clear|empty|wipe)\b[^.]{0,25}\breminders\b"
    r"|\b(delete|remove|cancel|clear)\b[^.]{0,10}\ball\b[^.]{0,15}\breminders?\b"
    r"|\ball\b[^.]{0,10}\b(my\s+)?reminders\b[^.]{0,20}"
    r"\b(clear(ed)?|empty|delet(e|ed)|remov(e|ed)|wipe(d)?|gone)\b",
    re.IGNORECASE,
)
# "clear my reminders" alone means PENDING only (the default everywhere else in this
# module — set/list/complete/delete all operate on pending rows). Only an explicit
# "including completed" (or a close variant) widens it to everything.
_CLEAR_ALL_SCOPE_RE = re.compile(
    r"\bincluding\b[^.]{0,15}\bcompleted\b|\binclude\b[^.]{0,15}\bcompleted\b"
    r"|\bcompleted\b[^.]{0,10}\b(too|as well)\b",
    re.IGNORECASE,
)

# Fallback safety net — a message that READS like a bulk-clear/delete-all command
# for something we have NO tool for (leads, habits, jobs, documents, or a reminders
# phrasing odd enough to slip past _CLEAR_RE above). Checked LAST, only when every
# other gate found nothing: it turns "empty my study log" from silent nothing (which
# fell through to general chat and got narrated back) into an honest decline routed
# through the SAME ACTION NOT DONE path real actions use. See _run_unsupported.
_BULK_CLEAR_INTENT_RE = re.compile(
    r"\b(clear|empty|wipe|erase)\b[^.]{0,25}\b(my\s+)?"
    r"(reminders?|leads?|habits?|jobs?|applications?|documents?|files?|"
    r"study(\s+(log|sessions?))?|logs?|history|everything|all of (it|them))\b"
    r"|\b(delete|remove)\b[^.]{0,10}\ball\b[^.]{0,20}\b(my\s+)?"
    r"(reminders?|leads?|habits?|jobs?|applications?|documents?|files?|"
    r"study(\s+(log|sessions?))?|logs?)\b",
    re.IGNORECASE,
)


# Action verb → the module function name, for an honest, greppable orch log line.
_FN_NAME = {
    "set": "set_reminder",
    "list": "list_reminders",
    "complete": "complete_reminder",
    "delete": "delete_reminder",
    "clear": "clear_reminders",
}
_JOBS_FN_NAME = {
    "log": "log_application",
    "list": "list_applications",
}

# Fitness gate: reporting/logging a lift PR or workout result. Requires an
# explicit PR/max signal, OR a training verb followed by a NUMBER (weight, reps,
# minutes, distance) — specific enough to keep the extractor off ordinary fitness
# chat/advice questions ("how do I improve my squat form", "is 225 a good bench",
# where no number FOLLOWS the verb). A rare false gate just costs one local call
# and falls through to none → normal routing.
#
# Two earlier misses this shape fixes, both of which are why fitness_logs sat at
# one row: a weight UNIT used to be mandatory, so the way Sir actually reports a
# lift ("benched 225 for 5") never gated at all and was never written; and timed
# work ("ran 30 minutes") had no verb in the list. Bare "pr" is also gone from the
# PR branch — it matched "pr manager" in a leadgen query and fired the fitness
# tool on a prospecting turn.
_FITNESS_GATE = re.compile(
    r"\b(prs|personal record|personal best|new max|1rm|one[- ]rep max)\b"
    r"|\b(new|hit a|hit another|another|my)\s+pr\b"
    r"|\b(bench(ed)?|squat(ted)?|deadlift(ed)?|press(ed)?|clean(ed)?|snatch(ed)?|"
    r"curl(ed)?|row(ed)?|ran|run|jog(ged)?|swam|cycled|lifted|trained)\b[^.]{0,40}"
    r"\b(\d+|lbs?|kgs?|pounds?|kilo(gram)?s?)\b"
    r"|\blog(ged)?\b[^.]{0,20}\b(workout|lift|set|rep)s?\b",
    re.IGNORECASE,
)

_FITNESS_FN_NAME = {
    "log": "log_fitness",
    "list": "list_fitness",
}

# Leads gate: saving/listing the prospecting rows a research turn produced. A save
# verb paired with a lead/studio/result noun ("save those leads", "log these
# studios"), or a plain listing ask. The rows themselves come from the session's
# leadgen cache, not from this line — see _run_leads.
_LEADS_GATE = re.compile(
    r"\b(save|log|store|add|keep|record)\b[^.]{0,40}"
    r"\b(leads?|studios?|prospects?|compan(y|ies)|results?|those|these)\b"
    r"|\b(list|show|what)\b[^.]{0,25}\b(leads?|prospects?)\b",
    re.IGNORECASE,
)

_LEADS_FN_NAME = {
    "save": "save_leads",
    "list": "list_leads",
}

# Study gate: reporting/listing a study session. A study verb paired with a
# duration/session/day word ("studied DP for two hours today"), an explicit log
# verb on a study noun, or a listing ask. Specific enough to keep the extractor off
# ordinary study QUESTIONS ("explain dynamic programming"), which must still route
# to the StudyTeacher.
_STUDY_GATE = re.compile(
    r"\b(stud(y|ied|ying)|revis(e|ed|ing|ion)|practic(e|ed|ing))\b[^.]{0,60}"
    r"\b(hours?|hrs?|minutes?|mins?|session|today|yesterday|morning|evening|tonight)\b"
    r"|\b(log|track|record|save)\b[^.]{0,25}\b(stud(y|ies|ying)|revision|session)\b"
    r"|\b(what|how much|show|list)\b[^.]{0,30}\b(stud(y|ied|ies|ying)|revision)\b",
    re.IGNORECASE,
)

_STUDY_FN_NAME = {
    "log": "log_study",
    "list": "list_study",
}

# Habits gate: marking a habit done, or asking whether one was done. Covers "mark
# meditation done", "did I meditate today", "log my habits", "habit streak". A rare
# false gate just costs one local call and falls through to none → normal routing.
_HABITS_GATE = re.compile(
    r"\bhabits?\b|\bstreak\b"
    r"|\b(mark|tick|check)\b[^.]{0,40}\b(done|off|complete[d]?)\b"
    r"|\bdid i\b[^.]{0,40}\b(today|yesterday|yet)\b",
    re.IGNORECASE,
)

_HABITS_FN_NAME = {
    "mark": "mark_habit",
    "check": "check_habit",
    "list": "list_habits",
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

# Email gate: composing/sending an email. The confirm-gate (yes/no/edit on a pending
# draft) is handled separately in detect_action and does NOT need this gate.
_EMAIL_GATE = re.compile(r"\bemail\b|\be-mail\b|\bsend\b[^.]*\bmail\b", re.IGNORECASE)
_EMAIL_FN_NAME = {
    "draft": "draft_email",
    "send": "send_email",
    "revise": "revise_email",
    "cancel": "cancel_email",
    "reconfirm": "reconfirm_email",
    "need_address": "draft_email",
    "cancel_awaiting": "cancel_email",
}

# A bare email address — to recognise the recipient on the turn AFTER we asked for it.
_ADDR_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

# session_id -> {"recipient","topic","subject"} while a draft is waiting on the
# recipient's address. This is the "draft started, needs the address" phase, distinct
# from mailer's "draft staged, awaiting yes/no" phase. Providing the address here
# COMPLETES the draft (stage + show) — it never sends.
_AWAITING_EMAIL = {}

# Confirm-gate intent on a PENDING email. Precedence: cancel, then edit, then send;
# anything else re-asks (never an accidental send). Negate is checked first so
# "no, change it" cancels rather than edits.
_AFFIRM = re.compile(
    r"\b(yes|yep|yeah|yup|sure|send( it| that| it now)?|do it|go( ahead)?|"
    r"confirm(ed)?|ship it|sounds good|looks good|approve[d]?|fire it|that works)\b",
    re.IGNORECASE,
)
_NEGATE = re.compile(
    r"\b(no|nope|don'?t|do not|cancel|stop|scrap( it)?|never ?mind|forget it|"
    r"hold off|abort|drop it|discard)\b",
    re.IGNORECASE,
)
_EDIT = re.compile(
    r"\b(change|edit|revise|rewrite|reword|make it|instead|add|remove|shorter|"
    r"longer|subject|tone|fix|adjust|tweak|rephrase)\b",
    re.IGNORECASE,
)


# The local 3B routinely emits the literal STRING "null" (and "none"/"N/A") where the
# schema asks for JSON null — which passes an `or ""` check and lands in the DB as the
# word "null". These two normalise a raw extracted field into a real value or None.
_NULL_WORDS = {"", "null", "none", "n/a", "na", "nil", "undefined", "-"}


def _clean(value) -> str | None:
    """Extracted text field → stripped string, or None if it's empty/a null word."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULL_WORDS else text


def _num(value) -> int | None:
    """Extracted count → int, or None. Tolerates the model returning "5" or "null"."""
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _classify_confirm(message: str) -> str:
    """Map a reply to a pending email into send | cancel | revise | reconfirm."""
    msg = message or ""
    if _NEGATE.search(msg):
        return "cancel"
    if _EDIT.search(msg):
        return "revise"
    if _AFFIRM.search(msg):
        return "send"
    return "reconfirm"

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
        resp = ollama_chat(
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
        resp = ollama_chat(
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


_FITNESS_EXTRACT_PROMPT = """You convert a short spoken report of a LIFT / WORKOUT into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "log|list|none", "exercises": [{"lift": "<the exercise name, e.g. 'bench press', 'back squat', 'run'>", "weight": "<the weight exactly as said, e.g. '225 lbs', '100kg', or empty>", "reps": <integer rep count if given, else null>, "sets": <integer set count if given, else null>, "duration": "<how long, exactly as said, e.g. '30 minutes', or empty>"}], "date": "<when this happened, exactly as said e.g. 'today','yesterday', or empty — empty means just now>"}

Rules:
- log  = Sir reporting a lift/workout result to be recorded ("I hit a new PR on bench, 225 for 5", "squatted 315 for 3 yesterday").
- list = show past PRs/workouts ("what are my PRs", "show my lift log"). Use "exercises": [].
- none = anything that is NOT actually reporting/logging a workout — including a training QUESTION ("how do I bench more", "is 225 a good bench"). Use "exercises": [].
- exercises: ONE ENTRY PER EXERCISE. "benched 225 for 5 and squatted 315 for 3" is TWO entries. Never merge two exercises into one entry, never split one exercise into two.
- lift: the exercise name only, keep Sir's wording, strip filler like "a new PR on".
- weight: copy the number + unit exactly as said. Do NOT convert units.
- reps/sets: integers only if explicitly stated; else null.
- duration: only for timed work ("ran 30 minutes", "45 min on the bike"); empty for a loaded lift.
- date: copy the time phrasing verbatim if given (e.g. 'yesterday', 'last Monday'); empty if not said. It applies to the whole message.
- If unsure whether it's a workout log at all, use "none".
JSON only."""


def _extract_fitness(message: str) -> dict | None:
    """Local strict-JSON extraction for a PR/workout-log command. Returns the
    parsed dict, or None for anything that isn't a usable log/list command.
    Same hard fail-safe contract as _extract."""
    try:
        resp = ollama_chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _FITNESS_EXTRACT_PROMPT},
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
        print(f"[orch] fitness extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _FITNESS_FN_NAME:  # 'none' or anything unexpected → not an action
        return None

    # One entry per exercise. A model that ignores the list schema and returns the
    # old flat single-lift shape is normalised into a one-entry list here, so the
    # execute path below only ever deals with one shape.
    raw = parsed.get("exercises")
    if not isinstance(raw, list):
        raw = [parsed] if parsed.get("lift") else []
    exercises = []
    for ex in raw:
        if not isinstance(ex, dict):
            continue
        lift = _clean(ex.get("lift"))
        if not lift:
            continue
        exercises.append({
            "lift": lift,
            "weight": _clean(ex.get("weight")),
            "reps": _num(ex.get("reps")),
            "sets": _num(ex.get("sets")),
            "duration": _clean(ex.get("duration")),
        })
    if action == "log" and not exercises:  # nothing to write → not an action
        return None
    return {
        "tool": "fitness",
        "action": action,
        "exercises": exercises,
        "date": _clean(parsed.get("date")),
    }


_LEADS_EXTRACT_PROMPT = """You convert a short command about SAVING PROSPECTING LEADS into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "save|list|none", "name": "<a single studio/company name IF Sir named one, else empty>", "location": "<city/area if said, else empty>", "contact": "<phone or website if said, else empty>", "research": "<any note about the lead if said, else empty>"}

Rules:
- save = Sir wants the leads kept ("save those leads", "log these studios", "add Studio Nine to my leads").
- list = show saved leads ("what leads do I have", "list my leads").
- none = anything that is NOT saving or listing leads — including a SEARCH request ("find 10 studios in Mumbai"), which is research, not a save.
- name: fill this ONLY when Sir named one specific studio/company. For "save those/these/the results" leave it EMPTY — those refer to the search results already on screen.
- Do NOT invent a name, location or contact. Empty is correct when it wasn't said.
- If unsure whether it's a save/list command at all, use "none".
JSON only."""


def _extract_leads(message: str) -> dict | None:
    """Local strict-JSON extraction for a leads save/list command. Returns the
    parsed dict, or None for anything that isn't one. Same hard fail-safe contract
    as _extract."""
    try:
        resp = ollama_chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _LEADS_EXTRACT_PROMPT},
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
        print(f"[orch] leads extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _LEADS_FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    return {
        "tool": "leads",
        "action": action,
        "name": _clean(parsed.get("name")) or "",
        "location": _clean(parsed.get("location")),
        "contact": _clean(parsed.get("contact")),
        "research": _clean(parsed.get("research")),
    }


_STUDY_EXTRACT_PROMPT = """You convert a short spoken report of a STUDY SESSION into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "log|list|none", "subject": "<the subject studied, e.g. 'DSA', 'organic chemistry', or empty>", "duration": "<how long, exactly as said, e.g. '2 hours', '90 minutes', or empty>", "topics": "<the specific topics covered if said, else empty>", "notes": "<any other detail Sir added, else empty>", "date": "<when, exactly as said e.g. 'today','yesterday', or empty>"}

Rules:
- log  = Sir reporting study he ALREADY did ("studied DP for two hours today", "did an hour of chemistry").
- list = show logged sessions ("what have I studied this week", "show my study log").
- none = anything that is NOT logging or listing a session — especially a LEARNING REQUEST ("teach me dynamic programming", "explain the limbic system"), which is not a log.
- subject: the subject only. topics: the specific things covered within it.
- duration: copy the phrasing exactly. Do NOT convert or invent a number.
- date: copy the time phrasing verbatim if given; empty if not said.
- If unsure whether it's a study log at all, use "none".
JSON only."""


def _extract_study(message: str) -> dict | None:
    """Local strict-JSON extraction for a study log/list command. Returns the parsed
    dict, or None for anything that isn't one. Same hard fail-safe contract as
    _extract."""
    try:
        resp = ollama_chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _STUDY_EXTRACT_PROMPT},
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
        print(f"[orch] study extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _STUDY_FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    return {
        "tool": "study",
        "action": action,
        "subject": _clean(parsed.get("subject")) or "",
        "duration": _clean(parsed.get("duration")),
        "topics": _clean(parsed.get("topics")),
        "notes": _clean(parsed.get("notes")),
        "date": _clean(parsed.get("date")),
    }


_HABITS_EXTRACT_PROMPT = """You convert a short command about a DAILY HABIT into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "mark|check|list|none", "name": "<the habit's name, e.g. 'meditation', 'gym', or empty>", "done": true|false, "date": "<when, exactly as said e.g. 'today','yesterday', or empty — empty means today>"}

Rules:
- mark  = record that a habit WAS done (or explicitly was not) ("mark meditation done", "I did my run today", "didn't meditate today" -> done false).
- check = ASK whether a habit was done ("did I meditate today", "have I done my run yet").
- list  = show today's habits ("what habits do I have today", "show my habits").
- none  = anything that is NOT about a habit — including marking a REMINDER or a TASK done.
- name: the habit only, stripped of "mark", "my", "done" and the date.
- done: true unless Sir clearly says he did NOT do it. For check/list, use true.
- date: copy the phrasing verbatim if given; empty means today.
- If unsure whether it's a habit command at all, use "none".
JSON only."""


def _extract_habits(message: str) -> dict | None:
    """Local strict-JSON extraction for a habit mark/check/list command. Returns the
    parsed dict, or None for anything that isn't one. Same hard fail-safe contract
    as _extract."""
    try:
        resp = ollama_chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _HABITS_EXTRACT_PROMPT},
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
        print(f"[orch] habits extract failed ({e}) — treating as none")
        return None

    action = (parsed.get("action") or "none").strip().lower()
    if action not in _HABITS_FN_NAME:  # 'none' or anything unexpected → not an action
        return None
    name = _clean(parsed.get("name"))
    if action in ("mark", "check") and not name:  # nothing to act on → not an action
        return None
    done = parsed.get("done", True)
    return {
        "tool": "habits",
        "action": action,
        "name": name or "",
        # A model that answers "false"/"no" as a STRING must not read as truthy.
        "done": str(done).strip().lower() not in ("false", "0", "no", "none"),
        "date": _clean(parsed.get("date")),
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
        resp = ollama_chat(
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


_EMAIL_EXTRACT_PROMPT = """You convert a command to SEND AN EMAIL into strict JSON. Output ONLY a JSON object, nothing else.

Schema:
{"action": "draft|none", "to": "<recipient email address if one is explicitly given, else empty>", "recipient": "<recipient name if given, else empty>", "subject": "<subject if Sir dictated one, else empty>", "topic": "<what the email should say / its purpose, in a few words>"}

Rules:
- draft = Sir wants to compose/send an email ("email Bob about the meeting", "send an email to x@y.com saying ...").
- none = anything that is NOT an email command.
- to: copy an email ADDRESS only if present; a name goes in "recipient", not "to".
- topic: the gist of what to say.
- If unsure whether it's an email command at all, use "none".
JSON only."""


def _extract_email(message: str) -> dict | None:
    """Local strict-JSON extraction for an email-draft command. None if it isn't one."""
    try:
        resp = ollama_chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": _EMAIL_EXTRACT_PROMPT},
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
        print(f"[orch] email extract failed ({e}) — treating as none")
        return None

    if (parsed.get("action") or "none").strip().lower() != "draft":
        return None
    return {
        "tool": "email",
        "action": "draft",
        "to": (parsed.get("to") or "").strip(),
        "recipient": (parsed.get("recipient") or "").strip(),
        "subject": (parsed.get("subject") or "").strip(),
        "topic": (parsed.get("topic") or "").strip(),
        "raw_message": message,
    }


def detect_action(message: str, session_id: str = "default") -> dict | None:
    """Cheap per-tool gate, then local extraction (see _detect), plus the idempotency
    key the execute stage checks. Returns the intent dict for a real action command,
    or None for ordinary chat."""
    action = _detect(message, session_id)
    if action is not None:
        action.setdefault("raw_message", message or "")
        action["idem_key"] = _idempotency_key(action, message or "")
    return action


def _detect(message: str, session_id: str = "default") -> dict | None:
    """Cheap per-tool gate, then local extraction. Returns the intent dict (tagged
    with its "tool" and the session) for a real action command, or None for ordinary
    chat. A PENDING email confirmation takes priority over every gate so a bare
    "yes"/"send it" confirms the draft instead of routing as small-talk."""
    msg = message or ""

    # 0. CONFIRM GATE — a STAGED draft (complete, awaiting yes/no) owns the turn.
    etool = get_tool("email")
    if etool is not None and etool.has_pending(session_id):
        return {"tool": "email", "action": _classify_confirm(msg),
                "session_id": session_id, "raw_message": msg}

    # 0.5 AWAITING ADDRESS — last turn we asked for the recipient's address. THIS
    # turn supplies it (complete + show the draft) or cancels. Never sends here.
    if session_id in _AWAITING_EMAIL:
        if _NEGATE.search(msg):
            return {"tool": "email", "action": "cancel_awaiting",
                    "session_id": session_id, "raw_message": msg}
        addr = _ADDR_RE.search(msg)
        if addr:
            saved = _AWAITING_EMAIL[session_id]
            return {"tool": "email", "action": "draft", "to": addr.group(0),
                    "recipient": saved.get("recipient", ""), "subject": saved.get("subject", ""),
                    "topic": saved.get("topic", ""), "session_id": session_id, "raw_message": msg}
        return {"tool": "email", "action": "need_address",
                "session_id": session_id, "raw_message": msg}

    # 1–7. Tool gates in order; a line that gates but fails extraction can fall
    # through to the next tool's gate.
    if _GATE.search(msg):
        # Bulk clear is DETERMINISTIC — checked before the LLM extractor, which has
        # no "clear" concept in its schema at all. No ollama round trip, never flaky.
        if _CLEAR_RE.search(msg):
            scope = "all" if _CLEAR_ALL_SCOPE_RE.search(msg) else "pending"
            return {"tool": "reminders", "action": "clear", "scope": scope,
                    "session_id": session_id, "raw_message": msg}
        action = _extract(msg)
        if action is not None:
            action["tool"] = "reminders"
            action["session_id"] = session_id
            return action
    if _JOBS_GATE.search(msg):
        action = _extract_jobs(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    # Leads before fitness: "save those PR firm results" would otherwise hit the
    # fitness gate's bare "pr" alternative first.
    if _LEADS_GATE.search(msg):
        action = _extract_leads(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    if _FITNESS_GATE.search(msg):
        action = _extract_fitness(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    if _STUDY_GATE.search(msg):
        action = _extract_study(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    if _HABITS_GATE.search(msg):
        action = _extract_habits(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    if _DOCS_GATE.search(msg):
        action = _extract_docs(msg)
        if action is not None:
            action["session_id"] = session_id
            return action
    if _EMAIL_GATE.search(msg):
        action = _extract_email(msg)
        if action is not None:
            action["session_id"] = session_id
            return action

    # 8. FALLBACK — nothing matched, but this reads like a bulk-clear/delete-all
    # command we have no tool for. Return an honest decline instead of None, so it
    # routes through the ACTION NOT DONE path (real material, a real task_block)
    # rather than falling to general chat with nothing to say — which is what
    # produced the "You asked me to empty all of them" narration.
    if _BULK_CLEAR_INTENT_RE.search(msg):
        return {"tool": "unsupported", "action": "decline",
                "session_id": session_id, "raw_message": msg}
    return None


# ── Idempotency ──
# A double-submit (impatient second Enter, a client retry, the same command spoken
# twice) used to execute twice: two reminders, two logged applications, two SENT
# emails. Nothing downstream could tell the difference, because each turn arrives as
# a fresh request with a fresh ctx.
#
# So every detected action carries a key derived from (session_id, normalized
# message, tool:verb) — stable for the same logical request, different for a
# different one. Before executing, run_action checks the key: if that exact request
# already EXECUTED within the TTL, the ORIGINAL result is returned and the tool is
# never called again.
#
# In-memory with a TTL, matching the other per-session state in this codebase
# (_AWAITING_EMAIL here, _PENDING in mailer, _SECRET_MODE, the leadgen cache). A
# restart clears it, which is correct: 60s of protection has no meaning across one.
_IDEM_TTL_S = 60.0
_idem_store: dict[str, tuple[float, dict]] = {}
_idem_lock = threading.Lock()

# Only WRITE verbs are guarded. Reads (list/check/get/reconfirm) are safe to repeat
# and must stay live — replaying a cached "your reminders" for 60s would show Sir a
# list that no longer matches the table he just changed.
_WRITE_VERBS = {
    "reminders": {"set", "complete", "delete", "clear"},
    "jobs": {"log"},
    "documents": {"save", "delete"},
    "leads": {"save"},
    "fitness": {"log"},
    "study": {"log"},
    "habits": {"mark"},
    "email": {"draft", "send", "revise", "cancel", "cancel_awaiting"},
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+")


def _normalize_message(message: str) -> str:
    """Message → comparison form: lowercase, punctuation dropped, whitespace
    collapsed. So "Remind me to stretch!" and "remind me to stretch" are ONE logical
    request, while a genuinely different command is a different key."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", (message or "").lower())).strip()


def _idempotency_key(action: dict, message: str) -> str:
    """(session_id, normalized message, action type) — the spec's triple.

    Email confirmations get ONE extra ingredient: the staged draft's identity. On the
    confirm gate the message is just "yes", which says nothing about WHICH email —
    two different drafts confirmed with the same word inside a minute would otherwise
    collide on one key and the second send would be swallowed as a replay. The draft
    is read, never modified, so this stays a pure key computation."""
    tool = action.get("tool") or "reminders"
    verb = action.get("action") or ""
    parts = [action.get("session_id") or "default", tool, verb, _normalize_message(message)]

    if tool == "email":
        try:
            etool = get_tool("email")
            pending = etool.get_pending(action.get("session_id", "default")) if etool else None
            if pending:
                parts.append(f"{pending.get('to')}|{pending.get('subject')}")
        except Exception as e:
            # Key computation must never break a turn. Without the draft identity the
            # key is merely coarser (the base triple), never wrong.
            print(f"[idem] draft fingerprint unavailable ({e}) — using the base key")

    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _is_write(action: dict) -> bool:
    return action.get("action") in _WRITE_VERBS.get(action.get("tool") or "reminders", set())


def _idem_get(key: str) -> dict | None:
    """The stored result for an identical request still inside the TTL, else None.
    Sweeps expired entries on the way through so the store can't grow unbounded."""
    now = time.monotonic()
    with _idem_lock:
        for k, (ts, _) in list(_idem_store.items()):
            if now - ts > _IDEM_TTL_S:
                del _idem_store[k]
        entry = _idem_store.get(key)
    return entry[1] if entry else None


def _idem_put(key: str, result: dict) -> None:
    with _idem_lock:
        _idem_store[key] = (time.monotonic(), result)


# ── Reply cache (closes the "two different replies" half of a replay) ──
# Deduping the TOOL WRITE (above) stops a duplicate reminder/email/etc. from ever
# landing twice, but cognition still turns the (correctly reused) tool result into
# spoken text with a fresh LLM call every time — so a genuine duplicate could still
# come back worded two different ways for the exact same fact. This is a second,
# separate cache, keyed by the SAME idem_key, holding the FULL reply cognition
# produced (response/mood/provider_used/tokens) so a replay returns it verbatim —
# see core/cognition.py's use of get_cached_reply/record_reply. Same TTL, same
# sweep-on-read pattern, deliberately a SEPARATE dict from _idem_store: the tool
# result is written the moment execution finishes, the reply only once cognition is
# done — two different write times, so keeping them apart avoids any partial-update
# bookkeeping on a single shared entry.
_idem_reply_store: dict[str, tuple[float, dict]] = {}


def get_cached_reply(idem_key: str) -> dict | None:
    """The FULL cognition reply already produced for this idem_key, if one exists
    and is still inside the TTL; else None (caller then generates fresh, exactly as
    if this cache didn't exist)."""
    if not idem_key:
        return None
    now = time.monotonic()
    with _idem_lock:
        for k, (ts, _) in list(_idem_reply_store.items()):
            if now - ts > _IDEM_TTL_S:
                del _idem_reply_store[k]
        entry = _idem_reply_store.get(idem_key)
    return dict(entry[1]) if entry else None


def record_reply(idem_key: str, reply: dict) -> None:
    """Remember the reply cognition produced for this idem_key, so a later duplicate
    of the SAME logical request (still inside the TTL) gets this exact text back."""
    if not idem_key:
        return
    with _idem_lock:
        _idem_reply_store[idem_key] = (time.monotonic(), dict(reply))


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
    """Execute the detected action against the REAL tool, ONCE per logical request.

    A write that already executed within the idempotency TTL returns its ORIGINAL
    stored result instead of running again — that is what stops a double-submit
    becoming two reminders or two sent emails. Reads and clarify/failure results are
    not cached: nothing was written, so repeating them is free and a transient
    failure stays retryable.

    A guarded result carries "_idem_key" (so cognition can key its OWN reply cache
    off the same identity) and, on a replay, "_idem_replay": True — see
    core/cognition.py's idempotent-replay short-circuit."""
    key = action.get("idem_key")
    guarded = bool(key) and _is_write(action)

    if guarded:
        cached = _idem_get(key)
        if cached is not None:
            print(f"[idem] replay {key[:12]} ({action.get('tool')}:{action.get('action')}) "
                  f"— already executed within {int(_IDEM_TTL_S)}s, returning the ORIGINAL "
                  "result, tool NOT called again")
            return {**cached, "_idem_key": key, "_idem_replay": True}

    res = _execute(action)

    # Only a real execution is recorded. An ok=False result (missing detail, nothing
    # staged, send failed) wrote nothing, so it must stay repeatable.
    if guarded and res.get("ok"):
        _idem_put(key, res)
        res = {**res, "_idem_key": key, "_idem_replay": False}
    return res


def _execute(action: dict) -> dict:
    """Dispatch to the REAL tool via the registry, respecting is_action. Returns the
    actual tool result dict (success, the needs/ask clarify path, or a not-found).
    Never raises."""
    if action.get("tool") == "unsupported":
        return _run_unsupported(action)
    if action.get("tool") == "jobs":
        return _run_jobs(action)
    if action.get("tool") == "fitness":
        return _run_fitness(action)
    if action.get("tool") == "leads":
        return _run_leads(action)
    if action.get("tool") == "study":
        return _run_study(action)
    if action.get("tool") == "habits":
        return _run_habits(action)
    if action.get("tool") == "documents":
        return _run_documents(action)
    if action.get("tool") == "email":
        return _run_email(action)
    return _run_reminders(action)


def _run_unsupported(action: dict) -> dict:
    """Nothing to execute — the message read like a bulk-clear/delete-all command
    but there is no tool for it (see _BULK_CLEAR_INTENT_RE). Returns a real, honest
    ok=False so cognition routes through the ACTION NOT DONE path and declines
    plainly, instead of falling to general chat with no material and narrating the
    request back."""
    return {"ok": False, "error": "unsupported",
            "ask": "I don't have a way to do that yet, Sir."}


def _run_email(action: dict) -> dict:
    """Drive the email confirm-gate. draft → stage (NO send); send → send only after
    explicit confirm; revise → recompose; cancel → discard; reconfirm → re-show."""
    name = "email"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    session = action.get("session_id", "default")
    verb = action["action"]
    try:
        if verb == "draft":
            # Use an address the extractor caught, else one sitting in the raw text.
            to = (action.get("to") or "").strip()
            if not _ADDR_RE.fullmatch(to):
                found = _ADDR_RE.search(action.get("raw_message") or "")
                to = found.group(0) if found else to
            draft = tool.draft(to, action.get("recipient"),
                               action.get("topic"), action.get("raw_message"))
            if not draft.get("ok"):
                # Missing the address → remember the rest so the NEXT turn (the
                # address) completes the draft. Nothing staged, nothing sent.
                if draft.get("needs") == "to":
                    _AWAITING_EMAIL[session] = {"recipient": action.get("recipient", ""),
                                                "topic": action.get("topic", ""),
                                                "subject": action.get("subject", "")}
                return {**draft, "stage": "draft"}
            _AWAITING_EMAIL.pop(session, None)  # address now known — leave the awaiting phase
            tool.stage(session, draft)          # stage (await confirm) — STILL not sent
            print("[email] draft -> awaiting confirm")
            return {"ok": True, "stage": "draft", **draft}
        if verb == "need_address":
            who = (_AWAITING_EMAIL.get(session, {}).get("recipient") or "their").strip()
            return {"ok": False, "needs": "to", "stage": "draft",
                    "ask": f"I still need {who} email address to draft it, Sir."}
        if verb == "cancel_awaiting":
            _AWAITING_EMAIL.pop(session, None)
            return {"ok": True, "cancelled": True, "had_draft": False}
        if verb == "send":
            return tool.confirm_send(session)      # sends ONLY here, after explicit yes
        if verb == "revise":
            return tool.revise(session, action.get("raw_message"))
        if verb == "cancel":
            return tool.cancel(session)
        if verb == "reconfirm":
            return tool.reconfirm(session)
    except Exception as e:
        print(f"[orch] email action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown email verb {verb}"}


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


def _run_fitness(action: dict) -> dict:
    """Execute a fitness-log action against the REAL FitnessTool via the registry.
    A log writes ONE ROW PER EXERCISE reported in the message."""
    name = "fitness"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "log":
            return tool.log_many(action.get("exercises") or [], action.get("date"))
        if verb == "list":
            return {"ok": True, "items": tool.list()}
    except Exception as e:
        print(f"[orch] fitness action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown fitness verb {verb}"}


def _run_leads(action: dict) -> dict:
    """Execute a leads action against the REAL LeadsTool via the registry.

    The rows for a bulk save come from THIS SESSION'S last prospecting run (the
    leadgen result cache) — "save those" refers to what's on screen, not to
    anything in the message. Only when Sir named one specific studio does the
    single-lead path run instead. The cache is the same one the export follow-up
    uses, so a save and a call-sheet export always agree on what "those" means."""
    name = "leads"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "list":
            return {"ok": True, "items": tool.list()}
        if verb == "save":
            # Local import: agents.leadgen pulls the teacher stack, and core is
            # imported by it — keeping this inside the call avoids the cycle.
            from agents.leadgen import get_cached_leadgen
            cached = get_cached_leadgen(action.get("session_id", "default"))
            rows = (cached or {}).get("rows") or []
            if rows:
                res = tool.save_many(rows)
                print(f"[leads] bulk save from cached set "
                      f"(query={(cached or {}).get('query')!r}): {len(rows)} row(s) offered")
                return res
            if action.get("name"):
                return tool.save(action["name"], action.get("location"),
                                 action.get("contact"), action.get("research"))
            return {"ok": False, "needs": "rows",
                    "ask": "I have no search results in hand to save, Sir — "
                           "run the search first, then tell me to save them."}
    except Exception as e:
        print(f"[orch] leads action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown leads verb {verb}"}


def _run_study(action: dict) -> dict:
    """Execute a study-log action against the REAL StudyTool via the registry."""
    name = "study"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "log":
            return tool.log(action.get("subject"), action.get("duration"),
                            action.get("topics"), action.get("notes"), action.get("date"))
        if verb == "list":
            return {"ok": True, "items": tool.list()}
    except Exception as e:
        print(f"[orch] study action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown study verb {verb}"}


def _run_habits(action: dict) -> dict:
    """Execute a habit action against the REAL HabitsTool via the registry. `check`
    is a READ — it answers "did I do X today" without writing a row."""
    name = "habits"
    if not is_action_tool(name):
        return {"ok": False, "error": f"{name} is not an action tool"}
    tool = get_tool(name)
    verb = action["action"]
    try:
        if verb == "mark":
            return tool.mark(action.get("name"), action.get("done", True),
                             action.get("date"))
        if verb == "check":
            return tool.check(action.get("name"), action.get("date"))
        if verb == "list":
            return {"ok": True, "items": tool.list(action.get("date"))}
    except Exception as e:
        print(f"[orch] habits action {verb} failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown habits verb {verb}"}


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

        if verb == "clear":
            return tool.clear(action.get("scope") or "pending")

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
    if action.get("tool") == "unsupported":
        return "decline"
    if action.get("tool") == "jobs":
        return _JOBS_FN_NAME.get(verb, verb)
    if action.get("tool") == "fitness":
        return _FITNESS_FN_NAME.get(verb, verb)
    if action.get("tool") == "leads":
        return _LEADS_FN_NAME.get(verb, verb)
    if action.get("tool") == "study":
        return _STUDY_FN_NAME.get(verb, verb)
    if action.get("tool") == "habits":
        return _HABITS_FN_NAME.get(verb, verb)
    if action.get("tool") == "documents":
        return _DOCS_FN_NAME.get(verb, verb)
    if action.get("tool") == "email":
        return _EMAIL_FN_NAME.get(verb, verb)
    return _FN_NAME.get(verb, verb)


def action_material(action: dict, res: dict) -> str:
    """Render the REAL tool result into a text block for cognition. This is the
    truth she confirms from — she must not embellish past what it states.
    Dispatches by the action's "tool" tag."""
    if action.get("tool") == "unsupported":
        return _unsupported_material(action.get("action"), res)
    if action.get("tool") == "jobs":
        return _jobs_material(action.get("action"), res)
    if action.get("tool") == "fitness":
        return _fitness_material(action.get("action"), res)
    if action.get("tool") == "leads":
        return _leads_material(action.get("action"), res)
    if action.get("tool") == "study":
        return _study_material(action.get("action"), res)
    if action.get("tool") == "habits":
        return _habits_material(action.get("action"), res)
    if action.get("tool") == "documents":
        return _documents_material(action.get("action"), res)
    if action.get("tool") == "email":
        return _email_material(action.get("action"), res)
    return _reminder_material(action.get("action"), res)


def _unsupported_material(verb: str, res: dict) -> str:
    """Render a declined/unsupported action for cognition. She must say plainly she
    can't do it — never restate or paraphrase Sir's request back to him as if
    acknowledging or handling it (that's how a decline reads as a stalled "yes")."""
    return (
        "ACTION NOT DONE — Sir asked for something you have NO tool to do. Do NOT "
        "claim it's done, do NOT restate or paraphrase his request back to him as if "
        "you're acknowledging or handling it. Tell him PLAINLY, in one short line, "
        f"that you can't do that (yet). {res.get('ask') or ''}"
    ).strip()


def _email_material(verb: str, res: dict) -> str:
    """Render an email action result for cognition. The confirm gate lives here in
    how she's told to respond: a draft is shown and confirmation requested; a send
    is reported only from the REAL result."""
    def _draft_view(r):
        return (f"TO: {r.get('to')}\nSUBJECT: {r.get('subject')}\n\n{r.get('body')}")

    if verb in ("draft", "revise", "reconfirm", "need_address"):
        if not res.get("ok"):
            # draft not built yet (missing recipient/topic) — relay the ask, nothing staged.
            return (f"ACTION NOT DONE — no draft is staged yet ({res.get('needs') or res.get('error')}). "
                    f"Do NOT claim anything was sent or drafted. Put this to Sir: {res.get('ask')}")
        lead = {"draft": "Here is the draft — NOT sent yet",
                "revise": "Here is the revised draft — still NOT sent",
                "reconfirm": "This email is still staged, NOT sent"}[verb]
        return (
            f"ACTION STAGED (NOT SENT) — {lead}. Show Sir the draft EXACTLY as below "
            "(do not rewrite the body — this is what will actually send), then ask him "
            "to confirm: say 'yes'/'send it' to send, 'cancel' to drop it, or tell you "
            "what to change. Do NOT claim it was sent.\n\n" + _draft_view(res)
        )

    if verb == "send":
        if res.get("ok") and res.get("sent"):
            return (f"ACTION EXECUTED — the email was ACTUALLY SENT to {res.get('to')} "
                    f"(subject {res.get('subject')!r}). This really happened. Confirm to Sir, minimally.")
        # not sent — config missing or send error; draft may still be pending.
        return (f"ACTION NOT DONE — the email was NOT sent ({res.get('error')}). Do NOT "
                f"claim it sent. Relay to Sir: {res.get('ask')}")

    # cancel
    if res.get("cancelled"):
        if res.get("had_draft"):
            return ("ACTION EXECUTED — the draft was discarded, nothing was sent. "
                    "Confirm to Sir that it's cancelled.")
        return ("ACTION EXECUTED — there was no draft to cancel; nothing was sent. "
                "Tell Sir there was nothing pending.")
    return "ACTION NOT DONE — nothing to cancel."


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
            if res.get("resume_note") == "no resume on file":
                resume_part = ("resume=NONE ON FILE (no resume saved to match against — tell "
                               "Sir to save one and you'll match future logs)")
                close = ("Confirm the application was logged, and add that there's no resume on "
                         "file to match — he should save one.")
            else:
                resume_part = (f"resume_used={res['resume_used']!r} (matched from his real "
                               "stored resume)")
                close = "Confirm to Sir minimally, naming the company, role and which resume."
            return (
                "ACTION EXECUTED — you just LOGGED a job application to the database. "
                "This actually happened, the write is done. Details: "
                f"company={res['company']!r}, role={res['role']!r}, {resume_part}, "
                f"status={res['status']!r}"
                + (f", follow_up={res['follow_up_at']!r}" if res.get("follow_up_at") else "")
                + ". " + close
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


def _fitness_material(verb: str, res: dict) -> str:
    """Render a fitness-log action result for cognition. This is the truth she
    confirms from — she must not embellish past what it states. A log may have
    written SEVERAL rows (one per exercise reported), so every stored row is
    listed and the count is stated exactly."""
    if verb == "log":
        if res.get("ok"):
            items = res.get("items") or [res]  # single-row result reads the same way
            lines = "\n".join(
                f"- {r['activity']}: {r.get('notes') or 'no detail'}"
                + (f", {r['duration_min']} min" if r.get("duration_min") else "")
                + f" (logged {r['logged_at']})"
                for r in items
            )
            partial = ""
            if res.get("failed"):
                partial = (" NOT everything landed: " + " ".join(str(f) for f in res["failed"])
                           + " Tell him that part is still missing — do not gloss over it.")
            return (
                f"ACTION EXECUTED — you just SAVED {len(items)} PR/workout row(s) to the "
                "database. This actually happened, the write is done. Exactly what was "
                f"stored:\n{lines}\nConfirm to Sir, naming the lift(s) and the number(s) — "
                "a short, genuinely-pleased-for-him beat, not a lecture. Never claim a "
                "lift that is not listed above." + partial
            )
        # clarify path — nothing was written
        return (
            "ACTION NOT DONE — no PR/workout was saved because a detail is missing "
            f"({res.get('needs')}). Do NOT claim it's logged. Put this to Sir: "
            f"{res.get('ask')}"
        )

    # list
    items = res.get("items", [])
    if not items:
        return ("ACTION EXECUTED — you checked the fitness log and it is EMPTY. "
                "Tell Sir he hasn't logged any PRs/workouts yet.")
    lines = "\n".join(
        f"#{r['id']} {r['activity']} — {r['notes'] or 'no detail'} ({r['logged_at']})"
        for r in items
    )
    return ("ACTION EXECUTED — these are Sir's logged PRs/workouts. Present them "
            "cleanly:\n" + lines)


def _leads_material(verb: str, res: dict) -> str:
    """Render a leads action result for cognition. The saved/skipped split is stated
    exactly — a re-save of an existing set writes nothing new and she must say so
    rather than claim N fresh rows."""
    if verb == "save":
        if res.get("ok"):
            saved = res.get("items") or []
            skipped = res.get("skipped") or []
            if not saved and skipped:
                return (
                    f"ACTION EXECUTED — nothing new was written: all {len(skipped)} of those "
                    "leads were ALREADY saved. Tell Sir they're already on file; do NOT claim "
                    "a fresh save."
                )
            lines = "\n".join(
                f"- {r['studio_name']}" + (f" ({r['location']})" if r.get("location") else "")
                for r in saved
            )
            dupe = (f" {len(skipped)} were already on file and were skipped — mention that "
                    "count, briefly." if skipped else "")
            return (
                f"ACTION EXECUTED — you just SAVED {len(saved)} lead(s) to the database. This "
                f"actually happened, the write is done. Exactly what was stored:\n{lines}\n"
                "Confirm to Sir minimally with the COUNT — do not re-list every lead unless he "
                "asks, and never name one that isn't above." + dupe
            )
        # clarify path — nothing was written
        return (
            "ACTION NOT DONE — no leads were saved "
            f"({res.get('needs') or res.get('error')}). Do NOT claim anything is saved. "
            f"Put this to Sir: {res.get('ask')}"
        )

    # list
    items = res.get("items", [])
    if not items:
        return ("ACTION EXECUTED — you checked the leads table and it is EMPTY. "
                "Tell Sir he hasn't saved any leads yet.")
    lines = "\n".join(
        f"#{r['id']} {r['studio_name']} — {r['location'] or 'no location'}"
        + (f" | {r['contact']}" if r.get("contact") else "")
        + (f" | {r['research']}" if r.get("research") else "")
        for r in items
    )
    return ("ACTION EXECUTED — these are Sir's saved leads. Present them "
            "cleanly:\n" + lines)


def _study_material(verb: str, res: dict) -> str:
    """Render a study-log action result for cognition."""
    if verb == "log":
        if res.get("ok"):
            length = (f"{res['duration_min']} min" if res.get("duration_min")
                      else "no duration given")
            return (
                "ACTION EXECUTED — you just SAVED a study session to the database. This "
                "actually happened, the write is done. Details: "
                f"subject={res['topic']!r}, duration={length}, "
                f"notes={res.get('notes') or 'none'}, logged_at={res['logged_at']!r}. "
                "Confirm to Sir minimally, naming the subject and the time."
            )
        # clarify path — nothing was written
        return (
            "ACTION NOT DONE — no study session was saved because a detail is missing "
            f"({res.get('needs')}). Do NOT claim it's logged. Put this to Sir: "
            f"{res.get('ask')}"
        )

    # list
    items = res.get("items", [])
    if not items:
        return ("ACTION EXECUTED — you checked the study log and it is EMPTY. "
                "Tell Sir he hasn't logged any sessions yet.")
    lines = "\n".join(
        f"#{r['id']} {r['topic']}"
        + (f" — {r['duration_min']} min" if r.get("duration_min") else "")
        + (f" — {r['notes']}" if r.get("notes") else "")
        + f" ({r['logged_at']})"
        for r in items
    )
    return ("ACTION EXECUTED — these are Sir's logged study sessions. Present them "
            "cleanly:\n" + lines)


def _habits_material(verb: str, res: dict) -> str:
    """Render a habit action result for cognition. `check` is a READ — she reports
    what the table says and nothing more; a habit never logged is NOT the same as
    one logged as not-done, and she must not blur the two."""
    if verb == "mark":
        if not res.get("ok"):
            return (f"ACTION NOT DONE — {res.get('ask')} Do not claim anything was marked.")
        state = "DONE" if res.get("done") else "NOT done"
        if res.get("already"):
            return (f"ACTION EXECUTED — {res['name']!r} was ALREADY marked {state} for "
                    f"{res['logged_at']}; nothing changed. Tell Sir it was already logged — "
                    "do NOT imply you just did it.")
        return (f"ACTION EXECUTED — you just marked {res['name']!r} {state} for "
                f"{res['logged_at']}. This really happened, the write is done. Confirm to "
                "Sir, minimally.")

    if verb == "check":
        if not res.get("ok"):
            return (f"ACTION NOT DONE — {res.get('ask')} Do not guess an answer.")
        if not res.get("found"):
            return (f"ACTION EXECUTED — you checked the habit log: there is NO entry for "
                    f"{res['name']!r} on {res['logged_at']}. Tell Sir plainly it isn't "
                    "logged — do NOT say he didn't do it, only that nothing is recorded.")
        verdict = "DONE" if res.get("done") else "logged as NOT done"
        return (f"ACTION EXECUTED — you checked the habit log: {res['name']!r} is {verdict} "
                f"for {res['logged_at']}. Answer Sir from exactly that, in one line.")

    # list
    items = res.get("items", [])
    if not items:
        return ("ACTION EXECUTED — you checked today's habits and there are NONE logged. "
                "Tell Sir nothing is marked for today yet.")
    lines = "\n".join(
        f"#{r['id']} {r['name']} — {'done' if r['done'] else 'not done'} ({r['logged_at']})"
        for r in items
    )
    return ("ACTION EXECUTED — these are Sir's habits for that day. Present them "
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

    if verb == "clear":
        if res.get("ok"):
            n = res.get("deleted", 0)
            scope = res.get("scope", "pending")
            if n == 0:
                # Mirrors the (reliable) empty-list phrasing above: "you checked, it's
                # EMPTY" — no verb near "clear/delete/remove" for a small model to
                # latch onto and falsely confirm. NOTHING was deleted; say that.
                which = "reminders at all (including completed ones)" if scope == "all" else "PENDING reminders"
                return (f"ACTION EXECUTED — you checked the reminders table: there were "
                        f"ZERO {which} to begin with. NOTHING was deleted — there was "
                        "nothing there. Do NOT say 'cleared', 'deleted', or 'removed', and "
                        "do NOT give a count. Tell Sir plainly there was nothing to clear.")
            scope_note = " (including completed ones)" if scope == "all" else ""
            return (
                f"ACTION EXECUTED — you just DELETED {n} reminder(s){scope_note}. This "
                "actually happened, the write is done. Confirm to Sir with the EXACT "
                f"count — {n}. Never say 'all' unless {n} genuinely is the whole list; "
                "never invent a different number."
            )
        return (f"ACTION NOT DONE — the reminders were NOT cleared "
                f"({res.get('error')}). Do NOT claim anything was deleted.")

    # complete / delete
    done_word = "marked done" if verb == "complete" else "deleted"
    if res.get("ok"):
        return (f"ACTION EXECUTED — reminder #{res.get('id')} was {done_word}. This "
                "really happened. Confirm to Sir, minimally.")
    if res.get("needs"):
        return f"ACTION NOT DONE — {res.get('ask')} Do not claim it's {done_word}."
    return ("ACTION NOT DONE — no matching reminder was found, so nothing changed. "
            "Tell Sir you couldn't find that one.")
