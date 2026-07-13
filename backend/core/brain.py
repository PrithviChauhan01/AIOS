"""Teacher brain — one per-task selector + prompt-COMPOSER call (Slice X + Slice 8 + v2).

Runs BEFORE the answer book, on a DEDICATED model: Groq llama-3.3-70b-versatile, keyed
by GROQ_API_KEY. It therefore SHARES Groq quota with the Groq answer book — an accepted
trade: a brain 429 raises, and plan() degrades to the keyword fallback + static template
exactly as it does for any other failure. ONE call per task (the same single call Slice X
made — composition folds into it, no extra call):

    {"tools": [...], "book_tier": "fast|strong|frontier", "ensemble": bool,
     "domains": [...], "composed_prompt": "...", "reason": "..."}

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
  * composed_prompt — (v2) a full, task-specific, PERSONALIZED instruction for the
                answer book, written by the brain FOR THIS TASK using relevant
                mem_core facts about Sir (public-tier only — the hard Chroma filter
                in profile.get_relevant_facts(cloud_bound=True)). The executor
                (agents.base) runs THIS instead of the teacher's static
                build_book_prompt() template, with tool results appended the same
                way. Empty/thin/malformed → None → the static template runs —
                worst case is exactly today's behavior.

Fail-soft by contract: a failed/timed-out brain call or malformed JSON falls back
to deterministic keyword rules + the caller's default tier + ensemble=False +
domains=[given domain] + composed_prompt=None (static template). plan() NEVER
raises, and a slow brain is cut off at _BRAIN_TIMEOUT_S so it can never hang a turn.

Privacy is absolute and clamps BEFORE any cloud call — it is MODEL-AGNOSTIC and sits
above the model choice, so swapping the brain cannot weaken it: secret/private turns
never send their text OR mem_core to the cloud brain (Groq included), never get
cloud-egress tools, never ensemble, never fan out, and never get a composed prompt
(static template + ollama only). Book routing itself keeps the existing guard
(secret → ollama only via select_book*; the retrieval guard in cognition) — this
module can only ever RESTRICT a turn, never widen one.
"""

import asyncio
import json
import re

from config import Config

# ── The dedicated brain model — Groq llama-3.3-70b-versatile. ──
# On api.groq.com, keyed by GROQ_API_KEY. It reuses core.books' shared Groq client (the
# same one the Groq answer book rides), so the brain and the answer book share ONE Groq
# connection pool AND one quota — an accepted trade: on a 429 the call raises and plan()
# degrades to the keyword fallback + static template, identical to any other brain
# failure. Groq is fast enough (sub-second to a few seconds) that the brain step stays
# imperceptible per turn — the earlier NVIDIA-NIM Qwen brain ran 130-200s/call and was
# abandoned. The call is capped at _BRAIN_TIMEOUT_S so a slow brain can't hang a turn.
BRAIN_MODEL = "llama-3.3-70b-versatile"
_BRAIN_TIMEOUT_S = 8    # a slow brain must never hang a turn — timeout → static fallback
_BRAIN_MAX_TOKENS = 1200  # room for the composed prompt (~250 words) + the routing JSON


def _call_brain_sync(prompt: str) -> str:
    """Blocking Groq chat completion — run off-thread by _call_brain_model. Returns the
    model's text content (the routing JSON)."""
    from core.books import groq_client
    r = groq_client().chat.completions.create(
        model=BRAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_BRAIN_MAX_TOKENS,
    )
    return r.choices[0].message.content or ""


async def _call_brain_model(prompt: str) -> str:
    """One Groq (llama-3.3-70b) call, hard-capped at _BRAIN_TIMEOUT_S. Raises on missing
    key, timeout, or any SDK error — plan() turns every raise into the static fallback."""
    if not (Config.GROQ_API_KEY or "").strip():
        raise RuntimeError("GROQ_API_KEY not set")
    return await asyncio.wait_for(
        asyncio.to_thread(_call_brain_sync, prompt),
        timeout=_BRAIN_TIMEOUT_S,
    )

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

_BRAIN_PROMPT = """You are the routing brain of a personal AI system serving one user ("Sir"). \
For the task below: decide which shared tools to draw, how much model power the answer needs, \
which specialist domain(s) it requires — and WRITE the working prompt the answer model will run. \
Respond with ONLY a JSON object, nothing else.

Format:
{{"tools": [], "book_tier": "fast|strong|frontier", "ensemble": true|false, "domains": ["{domain}"], \
"composed_prompt": "full instruction for the answer model", "reason": "one line"}}

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

composed_prompt: write the FULL instruction the answer model will execute for this task — you are \
composing its working prompt, not answering the task yourself. Rules for it:
- Frame the answer model as a domain analyst for the picked domain, working THIS specific task — \
restate the task concretely, not generically.
- PERSONALIZE it using WHAT IS KNOWN ABOUT SIR below: fold in whatever is genuinely relevant \
(his goals, background, preferences); ignore what isn't. Never invent facts about him.
- Demand RAW structured material only: clearly labeled ALL-CAPS section headers suited to the task \
(e.g. OVERVIEW / KEY CONCEPTS / EXAMPLES, or OBJECTIVE / APPROACH / RISKS, or SERVICES / CONTACT / \
FIT SIGNALS), no greeting, no personality, no sign-off.
- Demand facts only: anything unknown is written as "unknown", never invented. If live tool data is \
provided with the prompt, it must be used as the factual basis and never contradicted.
- Keep it under ~250 words. It must stand alone — the answer model sees ONLY your composed prompt \
plus tool data.

WHAT IS KNOWN ABOUT SIR (public-tier memory; relevant facts only, may be empty):
{sir_context}

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


def _extract_json(text: str) -> dict:
    """Parse the brain's reply into a dict, tolerating a model's habits: markdown
    code fences and prose around the object are stripped; the greedy first-{ to
    last-} span is what gets parsed (composed_prompt legitimately contains braces
    almost never, but nested braces inside the object survive a greedy span).
    Raises on anything unparseable — plan() turns that into the static fallback."""
    t = re.sub(r"```(?:json)?", "", text or "")
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in brain reply")
    return json.loads(match.group(0))


def _sir_context(message: str) -> str:
    """Relevant mem_core facts about Sir for personalizing the composed prompt.
    HARD privacy line: cloud_bound=True — profile.get_relevant_facts filters to
    tier='public' AT THE CHROMA QUERY, so private/secret facts can never reach the
    cloud brain even here on a public turn. (plan() never calls this for secret/private
    turns at all — the clamp returns first.) Fail-soft: any store error just means
    composing without personal context."""
    try:
        from core.profile import get_relevant_facts
        facts = get_relevant_facts(message, cloud_bound=True)
        if facts:
            return "\n".join(f"- {f}" for f in facts)
    except Exception as e:
        print(f"[brain] mem_core retrieval failed ({e}) — composing without personal context")
    return "(nothing relevant on file)"


# Bounds on a usable composed prompt: under the floor it's a stub (treat as absent →
# static template); over the ceiling it's runaway generation (truncate — the answer
# book's instruction, not an essay).
_COMPOSED_MIN_CHARS = 80
_COMPOSED_MAX_CHARS = 6000


def _clean_composed(raw) -> str | None:
    composed = str(raw).strip() if raw else ""
    if len(composed) < _COMPOSED_MIN_CHARS:
        return None
    return composed[:_COMPOSED_MAX_CHARS]


def _fallback(message: str, default_tier: str | None, complexity: str,
              loop_worthy: bool, why: str, known_domains: tuple,
              given_domain: str) -> dict:
    """Keyword-rule plan. Tier from the caller's default (teacher's declared tier)
    or complexity; ensemble always False — the fallback never spends two books.
    composed_prompt is None — the teacher's static template runs, i.e. exactly
    today's behavior. domains is NEVER guessed from keywords here — too fragile to
    safely widen scope on a failure — it's always just the caller's existing single
    given_domain."""
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
            "composed_prompt": None,
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
        # None (too thin / absent) → executor falls back to the static template.
        "composed_prompt": _clean_composed(parsed.get("composed_prompt")),
        "reason": str(parsed.get("reason", ""))[:200] or "no reason given",
        "source": "groq",
    }


async def plan(message: str, *, domain: str = "none", sensitivity: str = "public",
               complexity: str = None, loop_worthy: bool = False,
               default_tier: str | None = None,
               known_domains: tuple = DEFAULT_TEACHER_DOMAINS) -> dict:
    """The brain step. ONE Groq llama-3.3-70b call → {tools, book_tier, ensemble,
    domains, composed_prompt, reason, source}. NEVER raises; never weakens privacy.
    Shares Groq quota with the Groq answer book (accepted — the fallback chain handles
    exhaustion).

    source: 'groq' (brain answered — composed_prompt may still be None if it came
    back thin), 'fallback' (keyword rules + static template), or 'privacy'
    (secret/private clamp — no cloud call, no mem_core egress, static template).

    domains (Slice 8): len>=2 is the caller's signal to fan out those teachers
    concurrently. secret/private always get domains=[] — no multi-domain signal —
    so a sensitive turn's existing single-domain routing is never touched by this."""
    # ── PRIVACY CLAMP — before ANY cloud call, INCLUDING mem_core retrieval. ──
    # This is MODEL-AGNOSTIC: it returns before _call_brain_model is ever reached, so the
    # brain model (Groq, or anything it's swapped for) is never even constructed on a
    # sensitive turn. secret/private text never reaches the cloud brain, mem_core is never
    # queried for the compose (no personal facts egress), cloud-egress tools are
    # disallowed, ensemble is off, no multi-domain fan-out, and composed_prompt=None means
    # the local path runs the teacher's static template. Book selection keeps the existing
    # absolute guard downstream (secret → ollama).
    if sensitivity in ("secret", "private"):
        return {"tools": [], "book_tier": default_tier if default_tier in _TIERS else "fast",
                "ensemble": False, "domains": [], "composed_prompt": None,
                "reason": f"{sensitivity} turn — privacy clamp: no cloud brain, no "
                          "mem_core egress, no cloud tools, no ensemble, no fan-out",
                "source": "privacy"}

    domain_list = ", ".join(known_domains)
    domain_meanings = "\n".join(
        f'- "{d}": {_DOMAIN_DESCRIPTIONS.get(d, "")}' for d in known_domains)
    # mem_core → the compose. Public turns only (the clamp above already returned for
    # secret/private) and cloud_bound=True besides — public-tier facts only, filtered
    # at the Chroma query itself.
    sir_context = _sir_context(message)
    prompt = _BRAIN_PROMPT.format(domain=domain, complexity=complexity or "unknown",
                                  domain_list=domain_list, domain_meanings=domain_meanings,
                                  sir_context=sir_context,
                                  message=(message or "")[:2000])
    try:
        raw = await _call_brain_model(prompt)
        plan_out = _validate(_extract_json(raw), default_tier, complexity, loop_worthy,
                             known_domains, domain)
        print(f"[brain] brain=groq composed "
              f"(prompt={'yes' if plan_out['composed_prompt'] else 'none'} "
              f"tools={plan_out['tools']} tier={plan_out['book_tier']})")
        return plan_out
    except Exception as e:
        print(f"[brain] brain=groq failed ({e}) -> fallback: keyword rules + static template")
        return _fallback(message, default_tier, complexity, loop_worthy, str(e)[:80],
                         known_domains, domain)
