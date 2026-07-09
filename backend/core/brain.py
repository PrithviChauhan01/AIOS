"""Teacher brain — one cheap per-task selector call (Slice X + Slice 8).

Runs BEFORE the answer book. A single groq_8b (groq_fast) call reasons about the
task and returns a strict-JSON plan:

    {"tools": [...], "book_tier": "fast|strong|frontier", "ensemble": bool,
     "domains": [...], "reason": "..."}

  * tools     — which SHARED-POOL tools this task needs (may be empty). Read-only
                fetch tools only; action tools (reminders/email/…) stay behind the
                action-dispatch confirm gates and are never brain-selectable.
  * book_tier — reasoning tier for the ANSWER book. Replaces the teacher's fixed
                tier with a per-task call; the teacher's declared tier is the
                default the brain is given and the fallback when it fails.
  * ensemble  — whether THIS task genuinely benefits from two parallel books.
                This is the QUOTA GUARD: straightforward tasks get one book.
  * domains   — (Slice 8) which teacher domain(s) this task genuinely needs. Almost
                always one (the caller's given domain). 2-3 only when the task
                genuinely spans separate specialties in the SAME turn ("plan my
                week and suggest a workout" → life + fitness). Capped at 3. The
                caller (orchestrator) treats len(domains) >= 2 as a signal to fan
                out those teachers concurrently — this module only DETECTS the
                split, it never runs anything itself.

Fail-soft by contract: a failed call or malformed JSON falls back to deterministic
keyword rules + the caller's default tier + ensemble=False + domains=[given domain].
plan() NEVER raises.

Privacy is absolute and clamps BEFORE any cloud call: secret/private turns never
send their text to the cloud selector, never get cloud-egress tools, never
ensemble, and never fan out to multiple domains (domains=[] — no multi-domain
signal, so the caller keeps its existing single-domain routing unchanged). Book
routing itself keeps the existing guard (secret → ollama only via select_book*;
the retrieval guard in cognition) — this module can only ever RESTRICT a turn,
never widen one.
"""

import json
import re

from core.books import call_book

# The selector book — cheapest lane. This is routing, not answering.
BRAIN_BOOK = "groq_fast"

# Read-only pool tools the brain may draw. Action tools are deliberately absent.
SELECTABLE_TOOLS = ("places", "search", "wikipedia")

# Model-emitted synonyms → registry names ("tavily" is the common one).
_TOOL_ALIASES = {
    "tavily": "search", "web": "search", "websearch": "search",
    "web_search": "search", "google_places": "places", "wiki": "wikipedia",
}

_TIERS = ("fast", "strong", "frontier")

# Fallback list of teacher domains, used only when a caller doesn't pass its own
# known_domains (e.g. standalone tests). The orchestrator is the source of truth —
# it always passes tuple(TEACHERS.keys()) explicitly — this mirrors it so brain.py
# has no import-time dependency on core.orchestrator (would be circular: orchestrator
# imports brain).
DEFAULT_TEACHER_DOMAINS = (
    "leadgen", "study", "work", "fitness", "spirit", "life", "brainstorm", "jobs",
)

# Slice 8 quota/latency bound — even if the model or a caller hands back more, only
# the first 3 are ever kept.
MAX_FANOUT_DOMAINS = 3

# One-line descriptions the selector can reason from — mirrors triage.py's own domain
# semantics (duplicated intentionally: the 3B triage prompt is explicitly off-limits,
# and this is a different consumer with a different, cheaper model).
_DOMAIN_DESCRIPTIONS = {
    "study": "learning/teaching/explaining/researching a topic, concept, subject, science, history, language, or how-something-works",
    "work": "general professional tasks — objectives, plans, briefs",
    "leadgen": "researching a company/studio/business/prospect to pitch or sell to",
    "jobs": "job hunting for Sir himself — finding roles to apply to, shortlists, applications he made",
    "fitness": "workouts/exercise/training",
    "spirit": "Sir's OWN journaling, meditation, mood logging, personal reflection",
    "brainstorm": "discussing/developing ideas, finding directions, next steps, thinking through a problem or strategy",
    "life": "schedule/habits/reminders/organizing Sir's day or week",
}

_BRAIN_PROMPT = """You are the routing brain of a personal AI system. Decide, for the task below, \
which shared tools to draw, how much model power the answer needs, and which specialist \
domain(s) it requires. Respond with ONLY a JSON object, nothing else.

Format:
{{"tools": [], "book_tier": "fast|strong|frontier", "ensemble": true|false, "domains": ["{domain}"], \
"reason": "one line"}}

Tools exist ONLY to fetch EXTERNAL data the system does not already have. The DEFAULT is \
tools=[] — most turns are conversation and need NOTHING. Draw a tool only when the task truly \
cannot be answered without live external data.

Available tools — pick ONLY what the task genuinely needs; [] if none:
- "places": real business/venue directory (Google Places). Fire ONLY to find/recommend physical \
businesses or venues in a location — "find studios in Mumbai", "good cafes near me", "hidden gems \
in Delhi".
- "search": live web search. Fire ONLY to research a NAMED EXTERNAL entity the system must look up \
— a real company/person/product the user wants YOU to find facts about ("research Luma Labs", \
"who is <name>", "latest on <product>"), or news/fresh web data.
- "wikipedia": encyclopedia. Fire ONLY to explain a general concept/topic/history/science the user \
wants taught ("explain CAP theorem", "what is entropy").

Do NOT fire any tool — tools=[] — when the user is:
- TELLING or DISCUSSING their OWN project, idea, work, plan, or opinion ("I wanna talk about Simlr \
my project", "let me tell you about my startup", "here's my idea", "thoughts on my plan?"). A name \
the user calls THEIR OWN thing is NOT a search target — you already have it from them.
- just chatting, greeting, reacting, or asking for your take/advice on something they described.
- asking something answerable from reasoning or the conversation alone.
When unsure whether a name is an external entity to look up or the user's own thing they're \
describing, prefer tools=[] — do not search a personal project.

book_tier: "fast" = trivial ask, lookup, or formatting real data. "strong" = real reasoning or \
synthesis. "frontier" = hardest multi-step reasoning.

ensemble: true ONLY when answer quality genuinely benefits from two independent model passes \
(complex reasoning, a high-stakes deliverable). false for anything straightforward — never \
waste two models on a simple lookup or small talk.

domains: which teacher specialty/specialties this task needs, chosen ONLY from this list: \
{domain_list}. Return exactly ONE domain — normally the given domain, "{domain}", if it is one of \
the choices above; otherwise your single best pick from the list, or ["{domain}"] as-is if this is \
casual conversation needing no specialist. Return TWO OR THREE domains ONLY when the task \
GENUINELY asks for distinct expertise from SEPARATE specialties IN THE SAME TURN — e.g. "plan my \
week and suggest a workout" needs BOTH life and fitness; "research Luma Labs and draft outreach" \
needs BOTH leadgen and work. Do NOT split a single-topic task into multiple domains just because it \
touches adjacent ideas — most tasks are ONE domain. Max 3.

Domain meanings:
{domain_meanings}

Task (domain={domain}, complexity={complexity}):
{message}

JSON only."""


# ── Deterministic keyword fallback (no LLM) ──
# Fires when the selector call fails or returns junk. Conservative: ensemble is
# ALWAYS False here (quota guard), tier is the caller's default.
_PLACES_RE = re.compile(
    r"\b(restaurants?|cafes?|coffee shops?|studios?|gyms?|shops?|stores?|salons?|"
    r"hotels?|bars?|clinics?|garages?|bakeries|dentists?|plumbers?|photographers?|"
    r"boutiques?|barbers?|florists?|vendors?|agencies|businesses|"
    r"places (?:to|in|near|for)|spots?|hidden gems?|gems)\b", re.IGNORECASE)
_SEARCH_RE = re.compile(
    r"\b(research|look up|lookup|latest|news|reviews? of|web presence|"
    r"who is|background on|dig into|deep dive)\b", re.IGNORECASE)
_WIKI_RE = re.compile(
    r"\b(explain|what is|what's|define|definition of|history of|how does|"
    r"theorem|concept of|teach me)\b", re.IGNORECASE)


def _clean_domains(raw, known_domains: tuple, given_domain: str) -> list:
    """Validate a domains list against the caller's known teacher domains: unknown
    names dropped, deduped preserving order, capped at MAX_FANOUT_DOMAINS. Empty or
    entirely-invalid input defaults to [given_domain] (single-domain, i.e. the
    caller's existing routing is untouched) — or [] if given_domain isn't a real
    teacher domain (e.g. 'none' — casual conversation, nothing to fan out to)."""
    if not isinstance(raw, list):
        raw = []
    cleaned, seen = [], set()
    for d in raw:
        name = str(d).strip().lower()
        if name in known_domains and name not in seen:
            cleaned.append(name)
            seen.add(name)
    if not cleaned:
        cleaned = [given_domain] if given_domain in known_domains else []
    return cleaned[:MAX_FANOUT_DOMAINS]


def _fallback(message: str, default_tier: str | None, complexity: str,
              loop_worthy: bool, why: str, known_domains: tuple,
              given_domain: str) -> dict:
    """Keyword-rule plan. Tier from the caller's default (teacher's declared tier)
    or complexity; ensemble always False — the fallback never spends two books.
    domains is NEVER guessed from keywords here — too fragile to safely widen scope
    on a failure — it's always just the caller's existing single given_domain."""
    low = message or ""
    tools = []
    if _PLACES_RE.search(low):
        tools.append("places")
    if _SEARCH_RE.search(low):
        tools.append("search")
    if _WIKI_RE.search(low) and "places" not in tools:
        tools.append("wikipedia")

    if default_tier in _TIERS:
        tier = default_tier
    else:
        tier = "strong" if (complexity == "complex" or loop_worthy) else "fast"

    return {"tools": tools, "book_tier": tier, "ensemble": False,
            "domains": _clean_domains([], known_domains, given_domain),
            "reason": f"keyword fallback ({why})", "source": "fallback"}


def _validate(parsed: dict, default_tier: str | None, complexity: str,
              loop_worthy: bool, known_domains: tuple, given_domain: str) -> dict:
    """Normalize a parsed brain reply into a safe plan. Unknown tools/domains are
    dropped (tool aliases mapped first), a junk tier falls back to the default.
    Raises only on a structurally hopeless reply — the caller turns that into the
    keyword fallback."""
    raw_tools = parsed.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("tools is not a list")
    tools, seen = [], set()
    for t in raw_tools:
        name = _TOOL_ALIASES.get(str(t).strip().lower(), str(t).strip().lower())
        if name in SELECTABLE_TOOLS and name not in seen:
            tools.append(name)
            seen.add(name)

    tier = str(parsed.get("book_tier", "")).strip().lower()
    if tier not in _TIERS:
        tier = default_tier if default_tier in _TIERS else (
            "strong" if (complexity == "complex" or loop_worthy) else "fast")

    return {
        "tools": tools,
        "book_tier": tier,
        "ensemble": bool(parsed.get("ensemble", False)),
        "domains": _clean_domains(parsed.get("domains", []), known_domains, given_domain),
        "reason": str(parsed.get("reason", ""))[:200] or "no reason given",
        "source": "brain",
    }


async def plan(message: str, *, domain: str = "none", sensitivity: str = "public",
               complexity: str = None, loop_worthy: bool = False,
               default_tier: str | None = None,
               known_domains: tuple = DEFAULT_TEACHER_DOMAINS) -> dict:
    """The brain step. One cheap selector call → {tools, book_tier, ensemble,
    domains, reason, source}. NEVER raises; never weakens privacy.

    source: 'brain' (selector answered), 'fallback' (keyword rules), or
    'privacy' (secret/private clamp — no cloud call was made at all).

    domains (Slice 8): len>=2 is the caller's signal to fan out those teachers
    concurrently. secret/private always get domains=[] — no multi-domain signal —
    so a sensitive turn's existing single-domain routing is never touched by this."""
    # ── PRIVACY CLAMP — before ANY cloud call. ──
    # secret/private text never reaches the cloud selector, cloud-egress tools are
    # disallowed (every selectable fetch tool sends the query to an external API),
    # ensemble is off, and there is no multi-domain fan-out (no fan-out of a
    # sensitive turn to multiple teachers/books). Book selection keeps the existing
    # absolute guard downstream (secret → ollama only).
    if sensitivity in ("secret", "private"):
        return {"tools": [], "book_tier": default_tier if default_tier in _TIERS else "fast",
                "ensemble": False, "domains": [],
                "reason": f"{sensitivity} turn — privacy clamp: no cloud selector, "
                          "no cloud tools, no ensemble, no multi-domain fan-out",
                "source": "privacy"}

    domain_list = ", ".join(known_domains)
    domain_meanings = "\n".join(
        f'- "{d}": {_DOMAIN_DESCRIPTIONS.get(d, "")}' for d in known_domains)
    prompt = _BRAIN_PROMPT.format(domain=domain, complexity=complexity or "unknown",
                                  domain_list=domain_list, domain_meanings=domain_meanings,
                                  message=(message or "")[:2000])
    try:
        result = await call_book(prompt, BRAIN_BOOK, max_tokens=200)
        match = re.search(r"\{.*\}", result["raw_text"] or "", re.DOTALL)
        if not match:
            raise ValueError("no JSON object in selector reply")
        return _validate(json.loads(match.group(0)), default_tier, complexity, loop_worthy,
                         known_domains, domain)
    except Exception as e:
        print(f"[brain] selector failed ({e}) -> keyword fallback")
        return _fallback(message, default_tier, complexity, loop_worthy, str(e)[:80],
                         known_domains, domain)
