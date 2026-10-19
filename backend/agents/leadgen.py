import asyncio
import json
import os
import re
from urllib.parse import urlparse

from agents.base import Teacher
from core.books import select_book_for_tier

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real research dossier touches at least some of these ──
_RESEARCH_SIGNALS = (
    "service", "contact", "email", "phone", "website", "instagram",
    "social", "founder", "owner", "studio", "company", "address",
    "fit", "pricing", "portfolio", "client",
)

# ── Pass-through sentinels ──
# These prefixes mark material that is already final and TRUE (a not-found flag,
# verified Places rows, or an export confirmation). self_check passes them verbatim
# so the looper never re-asks a book to "improve" them — that re-ask is exactly how
# fabrication creeps in.
_NOT_FOUND_PREFIX = "NO_RESULTS:"           # search/Places ran, found nothing
_PLACES_PREFIX = "REAL BUSINESS RESULTS"    # verified Google Places rows
_WEB_PREFIX = "REAL WEB LEADS"              # Tavily-sourced, field-verified leads
_EXPORT_PREFIX = "CALL SHEET EXPORTED:"     # xlsx written to outputs/
_PASSTHROUGH_PREFIXES = (_NOT_FOUND_PREFIX, _PLACES_PREFIX, _WEB_PREFIX, _EXPORT_PREFIX)

# ── Web prospecting (Tavily → book structures → code verifies) ──
_NOT_FOUND = "not found"
_DEFAULT_LEAD_COUNT = 5
_MAX_LEAD_COUNT = 10
_SEARCH_RESULTS_PER_QUERY = 8
_PROMPT_SOURCE_CHARS = 800
# Up to 20 JSON rows (~80 tokens each) — the default 1024 cap truncated the array
# mid-object and the reply failed to parse.
_STRUCTURE_MAX_TOKENS = 3000
_COUNT_RE = re.compile(r"\b(\d{1,2})\b")
_LOCATION_RE = re.compile(
    r"\b(?:in|near|around)\s+([a-z][a-z .'-]*?)\s*(?:[,.?!]|$|\band\b|\bwith\b|\bfor\b)",
    re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Only internationally formatted numbers are pulled from an extracted page — a bare
# digit run on a web page is as likely an order id as a phone.
_INTL_PHONE_RE = re.compile(r"\+\d{1,3}[\s-]?\(?\d[\d\s()-]{7,14}\d")
# A phone-shaped run in source text. No parentheses: "+918693838888) 03" must split
# into the number and the junk, not read as one 14-digit phone.
_PHONE_RUN_RE = re.compile(r"\+?\d[\d\s-]{6,16}\d")
# Mail hosts anyone can sign up on. An address elsewhere must sit on the business's
# own or source domain — lead-list sites print obfuscated look-alikes (anuj@e795f9fte.in).
_WEBMAIL = {"gmail.com", "yahoo.com", "yahoo.co.in", "outlook.com", "hotmail.com",
            "rediffmail.com", "icloud.com", "proton.me", "protonmail.com"}

# Tokens in the target/query that carry no business IDENTITY — trade descriptors
# plus instruction filler. A web result that only echoes these is NOT evidence the
# specific named business was actually found (used by the Tavily single-research path).
_GENERIC_NAME_TOKENS = {
    # trade descriptors (kept broad; single-business research spans any category)
    "video", "editing", "editor", "editors", "studio", "studios", "film",
    "films", "media", "production", "productions", "post", "company", "shop",
    "store", "agency", "creative", "creatives", "services", "service",
    # request / instruction filler
    "research", "find", "lead", "leads", "about", "info", "information",
    "get", "pull", "look", "lookup", "the", "and", "for", "with",
    "co", "llc", "inc",
}

# ── Prospecting vs single-business routing ──
# Prospecting ("find me businesses in X") → Google Places (real directory data).
# Deep web-presence research on ONE named business → Tavily stays.
_SINGLE_RESEARCH_SIGNALS = (
    "research ", "look up", "lookup", "tell me about", "details on", "details about",
    "info on", "information on", "information about", "web presence", "profile of",
    "profile on", "who is", "what is", "background on", "dig into", "deep dive",
)
_LIST_VERBS = (
    "find", "list", "search for", "look for", "get me", "give me", "pull",
    "show me", "source", "scout", "prospect", "generate", "build a list", "compile",
    "leads for", "leads in",
)
# Only PLURAL category nouns signal a list ("find studios"), not the singular form
# that shows up inside a proper name ("Studio Nine"). Category asks are reliably
# plural; a lone singular noun is left to the single-research verbs to decide.
_PROSPECT_NOUNS = (
    "businesses", "studios", "shops", "stores", "services", "vendors", "prospects",
    "leads", "companies", "agencies", "restaurants", "gyms", "salons", "clinics",
    "cafes", "contractors", "suppliers", "dealers", "firms", "photographers",
    "realtors", "dentists", "plumbers", "bakeries", "hotels", "boutiques", "barbers",
    "florists", "garages",
)

# ── Tool manifest — the HARD boundary of what leadgen may reach ──
# Leadgen reasons over real data from EXACTLY these two tools and no others. A cheap
# rules-based selection step (below) picks which of them to call per task; every fetch
# site asserts the tool it's about to hit is in here, so leadgen can never call outside
# it. "tavily" is the shared pool's Tavily tool (tools/tavily.py, registry "tavily");
# "places" is Google Places. Leadgen uses each tool's STRUCTURED layer (search_places /
# tavily.search return dicts) rather than the registry's string wrapper — it needs the raw
# fields (phone/website/address) for the call sheet — but the two names still name the
# same tools the registry exposes.
_PLACES = "places"
_TAVILY = "tavily"          # Tavily search + extract, as named in tools/registry.py
MANIFEST = (_PLACES, _TAVILY)

# Words that mean "hand me the leads as a file" — any of these in a message is an
# export request. Leadgen always exports the rich .xlsx call sheet (openpyxl).
_EXPORT_INTENT_SIGNALS = (
    "call sheet", "callsheet", "call-sheet", "spreadsheet", "sheet",
    "xlsx", "excel", "csv", "export", "download",
)


# ── Pipeline commands (Slice B2) ──
# Leadgen owns the LANGUAGE of the sales pipeline; core.action_dispatch owns the
# EXECUTION (the only stage that runs a tool) and calls parse_pipeline_command from
# its gate. Deterministic regexes, no LLM: a pipeline write must never hinge on a
# model's reading, and a line that matches none of these is ordinary chat.
_PIPELINE_STAGE_WORDS = {
    "found": "found",
    "contacted": "contacted", "reached out": "contacted",
    "replied": "replied", "responded": "replied",
    "demo": "demo", "demo booked": "demo", "demo scheduled": "demo",
    "won": "won", "signed": "won", "closed won": "won",
    "lost": "lost", "dead": "lost", "closed lost": "lost",
}
_STAGE_ALT = "|".join(sorted((re.escape(w) for w in _PIPELINE_STAGE_WORDS),
                             key=len, reverse=True))
_FU = r"follow[\s-]?ups?"
_P_DUE = re.compile(
    rf"\b(?:who|which\s+leads?)\b[^.?]{{0,40}}\b{_FU}\b"
    rf"|\b{_FU}\b[^.?]{{0,20}}\b(?:today|due|overdue)\b"
    rf"|\bwho\s+needs\s+(?:a\s+)?{_FU}\b", re.IGNORECASE)
_P_PIPELINE = re.compile(
    r"\b(?:show|view|see|display|give|what'?s|whats|how'?s)\b[^.?]{0,25}\bpipeline\b"
    r"|^\s*(?:my\s+)?pipeline\s*\??$", re.IGNORECASE)
_P_LIST_STAGE = re.compile(
    rf"\b(?:show|list|which|what)\b[^.?]{{0,25}}\bleads?\b[^.?]{{0,15}}"
    rf"\b(?:in|at|marked(?:\s+as)?)\s+(?:the\s+)?(?P<stage>{_STAGE_ALT})\b"
    rf"|\b(?:show|list)\b\s+(?:me\s+)?(?:my\s+|all\s+)?(?:the\s+)?(?P<stage2>{_STAGE_ALT})\s+leads\b",
    re.IGNORECASE)
_P_NOTE = re.compile(
    r"^\s*(?:(?:please\s+)?(?:add|log|put)\s+(?:a\s+)?note\s+(?:to|for|on)|note\s+(?:for|on|to))"
    r"\s+(?P<name>.+?)(?:\s*:\s*|\s+[-–—]\s+)(?P<text>.+?)\s*$", re.IGNORECASE)
_P_STAGE = re.compile(
    rf"^\s*(?:please\s+)?(?:mark|move|set|put|update)\s+(?P<name>.+?)\s+"
    rf"(?:(?:as|to|in(?:to)?|at)\s+)?(?:the\s+)?(?P<stage>{_STAGE_ALT})(?:\s+stage)?\b"
    rf"(?P<rest>.*)$", re.IGNORECASE)
_P_FOLLOW_UP = re.compile(
    rf"^\s*(?:please\s+)?(?:set\s+(?:a\s+|the\s+)?)?{_FU}\s+(?:with|for|on)\s+(?P<name>.+?)\s+"
    rf"(?P<when>(?:on|for|to|by|in)\s+.+|tomorrow|today|next\s+\w+)\s*$", re.IGNORECASE)
_P_REST_WHEN = re.compile(rf"\b{_FU}\s+(?P<when>(?:on|by|in|for)\s+.+|tomorrow|next\s+\w+)",
                          re.IGNORECASE)


def _clean_lead_name(name: str) -> str:
    name = (name or "").strip().strip("\"'“”‘’").strip()
    name = re.sub(r"^(?:the|lead)\s+", "", name, flags=re.IGNORECASE)
    return re.sub(r"(?:'s|’s)$", "", name).strip()


def _clean_when(when: str) -> str:
    """'on Friday' → 'Friday'; 'in 3 days' stays (parse_due reads the 'in')."""
    return re.sub(r"^(?:on|for|to|by)\s+", "", (when or "").strip(" .!"),
                  flags=re.IGNORECASE)


# Outreach drafts (Slice B3) — staged through the mailer's confirm gate.
_P_DRAFT = re.compile(
    rf"^\s*(?:please\s+)?(?:draft|write|prepare|compose)\s+(?:an?\s+|the\s+)?"
    rf"(?P<kind>outreach|cold|intro|{_FU})(?:\s+(?:email|e-mail|mail|message|note))?"
    rf"\s+(?:to|for)\s+(?P<name>.+?)\s*[.!]?\s*$", re.IGNORECASE)


def parse_outreach_command(message: str) -> dict | None:
    """'draft outreach to <studio>' / 'draft follow-up to <studio>' →
    {"kind": "outreach"|"follow_up", "name"}, else None."""
    m = _P_DRAFT.search(message or "")
    if not m or not _clean_lead_name(m.group("name")):
        return None
    kind = "follow_up" if m.group("kind").lower().startswith("follow") else "outreach"
    return {"kind": kind, "name": _clean_lead_name(m.group("name"))}


def parse_pipeline_command(message: str) -> dict | None:
    """A pipeline command → {"action": stage|note|follow_up|due|pipeline|list_stage,
    ...fields}, or None. Reads are checked before writes so "who should I follow up
    with today?" can never parse as setting a follow-up for "who"."""
    msg = (message or "").strip()
    if not msg or parse_outreach_command(msg):
        return None
    if _P_DUE.search(msg):
        return {"action": "due"}
    if _P_PIPELINE.search(msg):
        return {"action": "pipeline"}
    m = _P_LIST_STAGE.search(msg)
    if m:
        word = (m.group("stage") or m.group("stage2")).lower()
        return {"action": "list_stage", "stage": _PIPELINE_STAGE_WORDS[word]}
    m = _P_NOTE.search(msg)
    if m and _clean_lead_name(m.group("name")):
        return {"action": "note", "name": _clean_lead_name(m.group("name")),
                "text": m.group("text").strip()}
    m = _P_STAGE.search(msg)
    if m and _clean_lead_name(m.group("name")):
        when = _P_REST_WHEN.search(m.group("rest") or "")
        return {"action": "stage", "name": _clean_lead_name(m.group("name")),
                "stage": _PIPELINE_STAGE_WORDS[m.group("stage").lower()],
                "when": _clean_when(when.group("when")) if when else None}
    m = _P_FOLLOW_UP.search(msg.rstrip(".!"))
    if m and _clean_lead_name(m.group("name")):
        return {"action": "follow_up", "name": _clean_lead_name(m.group("name")),
                "when": _clean_when(m.group("when"))}
    return None


# ── Outreach composition (Slice B3) ──
# The book writes; code decides what it may say. Facts come only from the lead's
# saved row (name, location, website, the source snippet) and Config.STUDIO_OS_PITCH.
# A draft that carries a URL, email or phone number not in those facts is thrown
# away for the deterministic template — the cheapest reliable no-invention check.
_OUTREACH_PROMPT = """You write a short, personalised cold email from Prithvi to a studio.
Return ONLY a JSON object: {"subject": "...", "body": "..."}
Rules:
- Use ONLY the facts provided. Do not invent clients, projects, awards, numbers, names, prices, links or anything else about the studio.
- Describe Studio OS ONLY with the pitch line provided — no extra features or claims.
- Body: 3-5 short sentences: one specific line about the studio from the facts, what Studio OS is, one low-pressure ask (a short call or reply).
- Subject: short and specific, no clickbait.
- Address the studio by name ("Hi <studio> team,"). Sign off exactly as "Prithvi".
JSON only."""

_FOLLOW_UP_PROMPT = """You write a brief follow-up email from Prithvi to a studio he already emailed.
Return ONLY a JSON object: {"subject": "...", "body": "..."}
Rules:
- 2-3 short sentences. Reference the previous email by its subject. Polite, no guilt-tripping.
- Use ONLY the facts provided — invent nothing about the studio or the product.
- Subject: "Re: <previous subject>".
- Sign off exactly as "Prithvi".
JSON only."""

_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|in|co|io|net|org|ai)\b",
                     re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\+?\d[\d\s-]{6,}\d")


def _lead_facts(lead: dict) -> dict:
    try:
        research = json.loads(lead.get("research") or "{}") or {}
    except ValueError:
        research = {}
    def known(v):
        return v if v and str(v).strip().lower() != _NOT_FOUND else ""
    return {
        "studio": lead["studio_name"],
        "location": known(lead.get("location")) or known(research.get("address")),
        "website": known(research.get("website")),
        "source_excerpt": (research.get("snippet") or "")[:600],
    }


def _invents(body: str, allowed: str) -> bool:
    """True if the body carries a URL, email or phone number the facts don't."""
    low = allowed.lower()
    tokens = _URL_RE.findall(body) + _EMAIL_RE.findall(body)
    if any(t.lower().rstrip(".,)") not in low for t in tokens):
        return True
    digits_allowed = re.sub(r"\D", "", allowed)
    return any(re.sub(r"\D", "", d) not in digits_allowed for d in _DIGIT_RUN_RE.findall(body))


def _signed(body: str) -> str:
    body = body.rstrip()
    return body if body.splitlines()[-1].strip().lower().endswith("prithvi") \
        else f"{body}\n\nBest,\nPrithvi"


def _outreach_fallback(facts: dict, pitch: str) -> tuple:
    where = f" in {facts['location']}" if facts["location"] else ""
    body = (f"Hi {facts['studio']} team,\n\n"
            f"I came across {facts['studio']}{where} while looking at studios doing "
            f"video work. I'm Prithvi — I'm building {pitch}.\n\n"
            "Would you be open to a short call, or a quick reply if it sounds relevant?\n\n"
            "Best,\nPrithvi")
    return f"{facts['studio']} x Studio OS", body


def _follow_up_fallback(facts: dict, prev: dict) -> tuple:
    body = (f"Hi {facts['studio']} team,\n\n"
            f"Just following up on my email \"{prev['subject']}\". Happy to share more "
            "or set up a short call whenever suits you.\n\nBest,\nPrithvi")
    return f"Re: {prev['subject']}", body


def compose_lead_email(lead: dict, kind: str = "outreach") -> dict:
    """{"ok": True, "subject", "body"} for a lead outreach / follow-up, or a clarify
    when a follow-up has no earlier email on record. Never raises."""
    from config import Config
    from tools import leads as leads_tool
    from tools.mailer import compose_with

    facts = _lead_facts(lead)
    pitch = (Config.STUDIO_OS_PITCH or "").strip() or "Studio OS"
    if kind == "follow_up":
        prev = leads_tool.last_sent(lead)
        if prev is None:
            return {"ok": False, "needs": "previous_email",
                    "ask": f"I have no earlier email to {facts['studio']} on record, Sir — "
                           "draft the outreach first."}
        facts["previous_email"] = {"subject": prev["subject"], "sent": prev["at"]}
        system, fallback = _FOLLOW_UP_PROMPT, _follow_up_fallback(facts, prev)
    else:
        facts["studio_os_pitch"] = pitch
        system, fallback = _OUTREACH_PROMPT, _outreach_fallback(facts, pitch)

    allowed = json.dumps(facts, ensure_ascii=False) + " " + pitch
    composed = compose_with(system, "FACTS:\n" + json.dumps(facts, ensure_ascii=False, indent=1))
    if composed and not _invents(composed[1], allowed) and not _invents(composed[0], allowed):
        subject, body = composed
    else:
        if composed:
            print(f"[leadgen] {kind} draft carried a detail not in the facts — template used")
        subject, body = fallback
    return {"ok": True, "subject": subject.strip(), "body": _signed(body)}


def is_export_request(message: str) -> bool:
    """True if the message asks for the leads as a file. Shared with the orchestrator
    so an export follow-up ('make a call sheet for those') can be routed back here."""
    return any(k in (message or "").lower() for k in _EXPORT_INTENT_SIGNALS)


# ── Last-result cache (per session) ──
# A prospecting run caches its FULL verified row set here keyed by session_id, so a
# follow-up like "give me a call sheet for those" can export every row without
# re-querying Places (the follow-up isn't a query — it references prior results).
# In-process is enough: AIOS is a single local process; a restart just means Sir
# re-runs the search before exporting.
_RESULT_CACHE: dict = {}


def cache_leadgen_results(session_id: str, query: str, rows: list) -> None:
    _RESULT_CACHE[session_id or "default"] = {"query": query, "rows": rows}


def get_cached_leadgen(session_id: str) -> dict | None:
    """{'query': str, 'rows': list} for this session's last prospecting run, or None."""
    return _RESULT_CACHE.get(session_id or "default")

# Conservative location → ISO region code map. Only forces a regionCode when the
# location is UNAMBIGUOUS — explicit countries, or major non-US metros that aren't
# shared with a US city. US is left to Places' own inference (too many shared names).
_REGION_PATTERNS = (
    (r"\b(united states of america|united states|u\.?s\.?a|america)\b", "US"),
    (r"\b(united kingdom|u\.?k|england|scotland|wales|britain|british)\b", "GB"),
    (r"\bindia\b", "IN"),
    (r"\bcanada\b", "CA"),
    (r"\baustralia\b", "AU"),
    (r"\b(germany|deutschland)\b", "DE"),
    (r"\bfrance\b", "FR"),
    (r"\bspain\b", "ES"),
    (r"\bitaly\b", "IT"),
    (r"\b(netherlands|holland)\b", "NL"),
    (r"\bireland\b", "IE"),
    (r"\bsingapore\b", "SG"),
    (r"\b(united arab emirates|u\.?a\.?e)\b", "AE"),
    (r"\b(new zealand)\b", "NZ"),
    (r"\bjapan\b", "JP"),
    (r"\bbrazil\b", "BR"),
    (r"\bmexico\b", "MX"),
)
_CITY_REGION = {
    "london": "GB", "manchester": "GB", "liverpool": "GB", "glasgow": "GB", "leeds": "GB",
    "mumbai": "IN", "delhi": "IN", "new delhi": "IN", "bangalore": "IN", "bengaluru": "IN",
    "hyderabad": "IN", "chennai": "IN", "kolkata": "IN", "pune": "IN", "ahmedabad": "IN",
    "toronto": "CA", "vancouver": "CA", "montreal": "CA", "ottawa": "CA", "calgary": "CA",
    "sydney": "AU", "melbourne": "AU", "brisbane": "AU", "perth": "AU", "adelaide": "AU",
    "dubai": "AE", "abu dhabi": "AE", "auckland": "NZ", "wellington": "NZ",
}


class LeadgenTeacher(Teacher):
    """Domain expert for lead research. Two modes, real data in both:
      * PROSPECTING — "find/research <category> in <place>" → Tavily searches; a book
        only STRUCTURES the hits into leads, then code verifies every field against
        the source text (unverifiable → "not found"; no real source URL → dropped).
        Leads are saved to the `leads` table. Google Places is the fallback when the
        web search can't run or verifies nothing.
      * SINGLE-BUSINESS RESEARCH — deep web-presence dig on ONE named business →
        Tavily, then a book frames a dossier from the REAL results.
    Never writes the outreach, never adds personality."""

    domain = "leadgen"
    memory_ns = "mem_leadgen"
    deliverable = True  # lead research is inherently a structured deliverable

    # Leadgen draws Places and Tavily from the SHARED pool itself, via their
    # STRUCTURED layers (search_places / tavily.search return dicts — it needs raw
    # phone/website/address fields for the call sheet, not the registry's string
    # wrapper). The generic brain fetch skips these so nothing fires twice; the
    # MANIFEST above stays the hard boundary of what leadgen may reach.
    owns_tools = ("places", "search", "tavily")

    def reasoning_tier(self, ctx):
        return "fast"  # formats real Places/Tavily data — no heavy reasoning needed

    # ── Top-level routing ──
    async def run(self, ctx: dict) -> dict:
        message = ctx.get("message") or ctx.get("query", "")
        session_id = ctx.get("session_id", "default")

        # Export-of-prior-results: an export request that is NOT itself a fresh query
        # ("give me a call sheet for those") → export the CACHED full list, no re-query.
        if (is_export_request(message) and get_cached_leadgen(session_id)
                and not self._looks_like_fresh_query(message)):
            return self._export_cached(ctx)

        # Rules only, no LLM call (quota).
        mode = self._select_mode(message)
        print(f"[leadgen] mode for {message!r}: {mode}")
        if mode == "single":
            return await self._run_single_research(ctx)  # web dig on one business
        return await self._run_prospecting(ctx)

    def _select_mode(self, message: str) -> str:
        """'prospect' or 'single'. A plural category ("find/research studios in
        Mumbai") is a list ask even when it says "research"; a research verb with no
        category noun is a dig on ONE named business ("research Luma Labs").
        Anything else defaults to prospecting — real data or an honest nothing."""
        low = (message or "").lower()
        if any(re.search(rf"\b{re.escape(n)}\b", low) for n in _PROSPECT_NOUNS):
            return "prospect"
        if any(s in low for s in _SINGLE_RESEARCH_SIGNALS):
            return "single"
        return "prospect"

    @staticmethod
    def _assert_in_manifest(name: str) -> None:
        """Hard boundary: refuse to fetch from any tool leadgen isn't allowed to call.
        Every fetch site passes through here, so the manifest is enforced, not advisory."""
        if name not in MANIFEST:
            raise ValueError(
                f"leadgen may not call tool {name!r} — manifest is {MANIFEST}")

    def _looks_like_fresh_query(self, message: str) -> bool:
        """True if the message names a business CATEGORY ('studios', 'gyms') — i.e. a
        new prospecting search — as opposed to a bare export command referencing the
        prior set ('those', 'them', 'the list')."""
        low = (message or "").lower()
        return any(re.search(rf"\b{re.escape(n)}\b", low) for n in _PROSPECT_NOUNS)

    # ── PROSPECTING — Tavily first, Places as the fallback ──
    async def _run_prospecting(self, ctx: dict) -> dict:
        # Privacy routing as before: the orchestrator never hands a secret turn to a
        # teacher. This is the structural backstop — no web egress for secret.
        if ctx.get("sensitivity") != "secret":
            web = await self._run_web_prospecting(ctx)
            if web is not None:
                return web
        return await self._run_places_prospecting(ctx)

    async def _run_web_prospecting(self, ctx: dict) -> dict | None:
        """Tavily searches → a book structures the hits → code verifies every field.
        Returns None when the search could not run or nothing verified, so the caller
        falls back to Places rather than reporting not-found off a dead search."""
        message = ctx.get("message") or ctx.get("query", "")
        want = self._lead_count(message)
        location = self._location_of(message)

        sources = await self._web_sources(message)
        if not sources:
            return None
        leads = await self._structure_leads(ctx, message, sources)
        leads = self._verified_leads(leads, sources, location)
        if not leads:
            print(f"[leadgen] web prospecting: {len(sources)} sources, 0 verified leads "
                  "-> places fallback")
            return None
        # Most-verified first: a lead whose own site carries its contacts beats a bare
        # name off a directory listing. Stable, so the book's order breaks ties.
        leads.sort(key=self._evidence, reverse=True)
        leads = await self._enrich_contacts(leads[:want])
        print(f"[leadgen] web prospecting query={message!r} -> {len(leads)} verified "
              f"lead(s) from {len(sources)} sources (wanted {want})")

        saved = self._save_web_leads(leads, message)
        rows = [self._call_sheet_row(lead) for lead in leads]
        cache_leadgen_results(ctx.get("session_id", "default"), message, rows)
        if is_export_request(message):
            return self._export_call_sheet(message, rows)
        return self._web_results(leads, want, saved)

    def _lead_count(self, message: str) -> int:
        m = _COUNT_RE.search(message or "")
        n = int(m.group(1)) if m else _DEFAULT_LEAD_COUNT
        return max(1, min(n, _MAX_LEAD_COUNT))

    def _location_of(self, message: str) -> str:
        m = _LOCATION_RE.search(message or "")
        return m.group(1).strip().title() if m else ""

    async def _web_sources(self, message: str) -> list:
        """Two advanced Tavily searches (the ask itself, and the same ask aimed at
        contact details), merged and de-duplicated by URL. [] when search can't run."""
        self._assert_in_manifest(_TAVILY)
        from tools import tavily
        queries = (message, f"{message} contact phone email")
        try:
            batches = await asyncio.gather(
                *(tavily.search(q, _SEARCH_RESULTS_PER_QUERY, "advanced") for q in queries))
        except Exception as e:
            print(f"[leadgen] tavily unavailable: {e}")
            return []
        seen, sources = set(), []
        for hit in (h for batch in batches for h in batch):
            url = (hit.get("url") or "").strip()
            if url and url not in seen and (hit.get("title") or hit.get("content")):
                seen.add(url)
                sources.append(hit)
        return sources

    def _structure_prompt(self, message: str, sources: list) -> str:
        # Each source is clipped for the PROMPT only (16 advanced hits run ~7k tokens,
        # most of Groq's 8k/min) — verification still checks the full text.
        blocks = "\n\n".join(
            f"[S{i}] {h['url']}\nTITLE: {h.get('title', '')}\n"
            f"TEXT: {(h.get('content') or '')[:_PROMPT_SOURCE_CHARS]}"
            for i, h in enumerate(sources, start=1))
        return f"""You extract business leads from web search results. You do NOT research, guess, or use your own knowledge.

REQUEST: {message}

SEARCH RESULTS:
{blocks}

Return ONLY a JSON array, no prose. One object per distinct business that the results above actually name and that matches the request:
{{"studio_name": "...", "address": "...", "phone": "...", "email": "...", "website": "...", "source_url": "..."}}

Rules:
- source_url MUST be copied exactly from the [S#] line of the result that names the business.
- Copy every field character-for-character from that result's text. If a field is not written there, use "{_NOT_FOUND}".
- Never complete a masked or partial value (e.g. "98xxxx12", "Register to view") — use "{_NOT_FOUND}".
- Directories and lead-list sites (Justdial, Clutch, etc.) are not leads themselves — but DO extract the businesses they name.
- Prefer a business's own website as its source when it appears in the results.
- Extract EVERY matching business in the results, ignoring any number in the request — up to {_MAX_LEAD_COUNT * 2} objects. Empty array if none."""

    async def _structure_leads(self, ctx: dict, message: str, sources: list) -> list:
        """The book's only job: turn real hits into JSON rows. Tried down the tier's
        book list until one returns parseable JSON."""
        prompt = self._structure_prompt(message, sources)
        for book in select_book_for_tier("fast", ctx.get("sensitivity", "public")):
            try:
                result = await self._traced_book(prompt, book, ctx,
                                                 max_tokens=_STRUCTURE_MAX_TOKENS)
            except Exception:
                continue
            parsed = self._parse_json_array(result.get("raw_text"))
            if parsed is not None:
                print(f"[leadgen] structured by {result['book_used']}: "
                      f"{len(parsed)} candidate(s)")
                return parsed
            print(f"[leadgen] {book} returned no parseable JSON — next book")
        return []

    @staticmethod
    def _parse_json_array(text) -> list | None:
        text = text or ""
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else None

    # ── Verification — the book proposes, the sources decide ──
    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _domain(url: str) -> str:
        url = (url or "").strip().lower()
        if not url or url == _NOT_FOUND:
            return ""
        host = urlparse(url if "//" in url else f"//{url}").hostname or ""
        return host[4:] if host.startswith("www.") else host

    @staticmethod
    def _found(value) -> str:
        v = str(value or "").strip()
        return "" if v.lower() in ("", _NOT_FOUND, "unknown", "n/a", "none", "-", "—") else v

    def _verify_phone(self, phone: str, text: str) -> str:
        """The SOURCE's own rendering of the number, or not found. A whole number in
        the text must match the book's digits (one may carry an extra country/trunk
        prefix, or a few trailing junk digits) — so a garbled copy is replaced by what
        the page actually says."""
        if not phone or "x" in phone.lower():
            return _NOT_FOUND
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 8:
            return _NOT_FOUND
        for run in _PHONE_RUN_RE.findall(text):
            cand = re.sub(r"\D", "", run)
            short, long_ = sorted((cand, digits), key=len)
            if (len(short) >= 8 and len(long_) - len(short) <= 3
                    and (long_.endswith(short) or long_.startswith(short))):
                return " ".join(run.split())
        return _NOT_FOUND

    def _verify_email(self, email: str, text: str, own_domains: set) -> str:
        email = email.lower()
        if not _EMAIL_RE.fullmatch(email) or email not in text.lower():
            return _NOT_FOUND
        dom = email.rsplit("@", 1)[1]
        if dom in _WEBMAIL or any(d and (dom == d or dom.endswith("." + d)) for d in own_domains):
            return email
        return _NOT_FOUND

    def _verified_leads(self, candidates: list, sources: list, location: str) -> list:
        """Keep a lead only if its source_url is a real search hit whose text names
        it; every other field survives only if it appears in that source's text."""
        by_url = {h["url"]: h for h in sources}
        leads, seen = [], set()
        for c in candidates:
            name = self._found(c.get("studio_name"))
            src = by_url.get(str(c.get("source_url") or "").strip())
            if not name or src is None:
                print(f"[leadgen] dropped {name!r}: source_url not a search hit")
                continue
            text = f"{src.get('title', '')}\n{src.get('content', '')}"
            key = self._norm(name)
            if len(key) < 3 or key not in self._norm(text) or key in seen:
                print(f"[leadgen] dropped {name!r}: not named in its source (or duplicate)")
                continue
            seen.add(key)
            src_domain = self._domain(src["url"])
            website = self._found(c.get("website"))
            web_domain = self._domain(website)
            if not web_domain or (web_domain not in text.lower() and web_domain != src_domain):
                website = _NOT_FOUND
            address = self._found(c.get("address"))
            if not address or self._norm(address) not in self._norm(text):
                address = _NOT_FOUND
            own = {src_domain, self._domain(website)}
            leads.append({
                "studio_name": name,
                "address": address,
                "location": location or (address if address != _NOT_FOUND else ""),
                "phone": self._verify_phone(self._found(c.get("phone")), text),
                "email": self._verify_email(self._found(c.get("email")), text, own),
                "website": website,
                "source_urls": [src["url"]],
                "snippet": (src.get("content") or "")[:500],
            })
        return leads

    def _evidence(self, lead: dict) -> int:
        found = sum(lead[k] != _NOT_FOUND for k in ("phone", "email", "website", "address"))
        own_site = lead["website"] != _NOT_FOUND and \
            self._domain(lead["website"]) == self._domain(lead["source_urls"][0])
        return found + own_site

    async def _enrich_contacts(self, leads: list) -> list:
        """A lead with a known website but no phone/email: read that page with Tavily
        extract and take only what is literally on it. The page becomes a second
        source URL. Fail-soft per lead."""
        self._assert_in_manifest(_TAVILY)
        from tools import tavily

        async def enrich(lead):
            if lead["website"] == _NOT_FOUND or (
                    lead["phone"] != _NOT_FOUND and lead["email"] != _NOT_FOUND):
                return
            url = lead["website"] if "//" in lead["website"] else f"https://{lead['website']}"
            try:
                page = (await tavily.extract(url)).get("content") or ""
            except Exception as e:
                print(f"[leadgen] extract failed for {url}: {e}")
                return
            got = False
            if lead["email"] == _NOT_FOUND:
                for email in _EMAIL_RE.findall(page):
                    if self._verify_email(email, page, {self._domain(url)}) != _NOT_FOUND:
                        lead["email"], got = email.lower(), True
                        break
            if lead["phone"] == _NOT_FOUND:
                m = _INTL_PHONE_RE.search(page)
                if m:
                    lead["phone"], got = " ".join(m.group(0).split()), True
            if got and url not in lead["source_urls"]:
                lead["source_urls"].append(url)

        await asyncio.gather(*(enrich(lead) for lead in leads))
        return leads

    def _save_web_leads(self, leads: list, query: str) -> dict:
        """Persist through the registry's leads tool — None in public mode, so a public
        host can never write. Dedup (name + location) lives in save_leads."""
        from tools.registry import get_tool
        tool = get_tool("leads")
        if tool is None:
            return {"count": 0, "skipped": [], "disabled": True}
        rows = []
        for lead in leads:
            contact = " · ".join(v for v in (lead["phone"], lead["email"], lead["website"])
                                 if v != _NOT_FOUND) or _NOT_FOUND
            research = {k: lead[k] for k in
                        ("address", "phone", "email", "website", "source_urls", "snippet")}
            research.update(found_via="tavily", query=query)
            rows.append({"studio_name": lead["studio_name"], "location": lead["location"],
                         "contact": contact,
                         "research": json.dumps(research, ensure_ascii=False)})
        try:
            return tool.save_many(rows)
        except Exception as e:
            print(f"[leadgen] saving leads failed: {e}")
            return {"count": 0, "skipped": [], "error": str(e)}

    @staticmethod
    def _call_sheet_row(lead: dict) -> dict:
        """The cached row shape the call sheet export and the "save those" action read."""
        def blank(v):
            return "" if v == _NOT_FOUND else v
        return {"name": lead["studio_name"], "phone": blank(lead["phone"]),
                "website": blank(lead["website"]), "address": blank(lead["address"]),
                "location": lead["location"], "email": blank(lead["email"]),
                "rating": None, "reviews": 0,
                "research": json.dumps({"source_urls": lead["source_urls"]})}

    def _web_results(self, leads: list, want: int, saved: dict) -> dict:
        def where(lead):
            return lead["address"] if lead["address"] != _NOT_FOUND \
                else (lead["location"] or _NOT_FOUND)
        body = "\n\n".join(
            f"{i}. {lead['studio_name']}\n"
            f"   Location: {where(lead)}\n"
            f"   Phone: {lead['phone']}\n"
            f"   Email: {lead['email']}\n"
            f"   Website: {lead['website']}\n"
            f"   Source: {' , '.join(lead['source_urls'])}"
            for i, lead in enumerate(leads, start=1))
        short = (f" Only {len(leads)} of the {want} requested could be verified from the "
                 "web results — say so plainly; do not pad the list.") if len(leads) < want else ""
        if saved.get("disabled") or saved.get("error"):
            save_note = "These leads were NOT saved."
        else:
            skipped = saved.get("skipped") or []
            save_note = (f"Saved {saved.get('count', 0)} new lead(s) to the leads table"
                         + (f"; {len(skipped)} already on file were skipped." if skipped else "."))
        raw = (
            f"{_WEB_PREFIX} ({len(leads)} businesses from live web search — every field "
            "checked against its source page). Present these to Sir exactly as given, "
            "INCLUDING each Source link. A field marked \"not found\" stays \"not found\" — "
            "never fill, guess, or alter a name, phone, email, website, or address, and "
            f"never add leads.{short} {save_note}\n\n" + body
        )
        return self._final(raw, book="tavily", deliverable=True,
                           feedback="verified web leads")

    # ── PROSPECTING FALLBACK (Google Places) ──
    async def _run_places_prospecting(self, ctx: dict) -> dict:
        message = ctx.get("message") or ctx.get("query", "")
        region = self._detect_region(message)
        rows = await self._safe_places(message, region)

        print(f"[leadgen] prospecting query={message!r} region={region or 'auto'} "
              f"-> {len(rows)} real result(s)")

        if not rows:
            return self._places_not_found(message)

        # Cache the FULL set so a later "make a call sheet for those" exports every row.
        cache_leadgen_results(ctx.get("session_id", "default"), message, rows)

        if is_export_request(message):
            return self._export_call_sheet(message, rows)

        return self._places_results(message, rows)

    async def _safe_places(self, message: str, region: str | None) -> list:
        """Fetch verified Places rows for a query, fail-soft. Asserts Places is in the
        manifest first (the boundary is enforced at the fetch site). search_places is
        itself fail-soft, but a surprise error must never break cognition — return []."""
        self._assert_in_manifest(_PLACES)
        try:
            from tools.places import search_places
            return await search_places(message, region=region)
        except Exception as e:
            print(f"[leadgen] places lookup errored: {e}")
            return []

    def _detect_region(self, message: str) -> str | None:
        """ISO region code if the location is OBVIOUS, else None (let Places infer).
        Country mentions win; otherwise a small set of unambiguous foreign metros."""
        low = (message or "").lower()
        for pattern, code in _REGION_PATTERNS:
            if re.search(pattern, low):
                return code
        for city, code in _CITY_REGION.items():
            if re.search(rf"\b{re.escape(city)}\b", low):
                return code
        return None

    def _export_cached(self, ctx: dict) -> dict:
        """Export the session's cached prospecting set to an .xlsx call sheet. Titled
        from the ORIGINAL query (so the filename reflects 'studios in Mumbai', not the
        'make a call sheet' follow-up)."""
        session_id = ctx.get("session_id", "default")
        cached = get_cached_leadgen(session_id)
        rows = cached["rows"]
        print(f"[leadgen] export follow-up — cached set for session={session_id!r}: "
              f"{len(rows)} rows (original query={cached['query']!r})")
        return self._export_call_sheet(cached["query"], rows)

    def _places_not_found(self, message: str) -> dict:
        """Honest not-found — Places ran and returned nothing (or has no key). NO
        fabrication. deliverable=False so cognition relays it plainly and offers to
        broaden / rename."""
        query = (message or "").strip()
        flag = (
            f"{_NOT_FOUND_PREFIX} Google Places returned no businesses for \"{query}\". "
            "Tell Sir plainly that nothing was found. Do NOT fabricate any names, phones, "
            "ratings, websites, or addresses. Suggest he broaden the search (wider area or "
            "a simpler/more common category) or rename it (a different keyword)."
        )
        return self._final(flag, book="none", deliverable=False,
                           feedback="places ran, zero results — not-found flag")

    def _places_results(self, message: str, rows: list) -> dict:
        """Format the VERIFIED Places rows verbatim as the material. No book call —
        cognition presents and judges fit; she never invents a field."""
        body = "\n\n".join(self._format_row(i, r) for i, r in enumerate(rows, start=1))
        raw = (
            f"{_PLACES_PREFIX} ({len(rows)} businesses from Google Places — VERIFIED data. "
            "Present these to Sir exactly as given and judge fit only from what's here. "
            "NEVER add, drop, or alter a name, phone, rating, website, or address, and never "
            "invent extra leads):\n\n" + body
        )
        return self._final(raw, book="google_places", deliverable=True,
                           feedback="verified places rows")

    def _format_row(self, i: int, r: dict) -> str:
        rating = r.get("rating")
        rating_str = f"{rating} ({r.get('reviews', 0)} reviews)" if rating is not None \
            else f"unrated ({r.get('reviews', 0)} reviews)"
        return (
            f"{i}. {r.get('name', '').strip()}\n"
            f"   Phone: {r.get('phone') or '—'}\n"
            f"   Website: {r.get('website') or '—'}\n"
            f"   Address: {r.get('address') or '—'}\n"
            f"   Rating: {rating_str}"
        )

    def _export_call_sheet(self, message: str, rows: list) -> dict:
        """Write the rows to a styled .xlsx call sheet and confirm — the file IS the
        deliverable, so cognition just names it and the count, no row dump."""
        try:
            from tools.excel_gen import make_call_sheet
            path = make_call_sheet(self._export_title(message), rows)
        except Exception as e:
            print(f"[leadgen] excel export failed: {e} — falling back to in-chat list")
            return self._places_results(message, rows)

        fname = os.path.basename(path)
        print(f"[leadgen] call sheet written: {path} ({len(rows)} rows)")
        raw = (
            f"{_EXPORT_PREFIX} {len(rows)} verified leads saved to {path}. Tell Sir plainly "
            f"the call sheet is ready — name the file ({fname}) and the lead count ({len(rows)}). "
            "Do NOT list the rows back or invent anything; the data is in the file."
        )
        return self._final(raw, book="google_places", deliverable=False,
                           feedback="call sheet exported", file_path=path)

    def _export_title(self, message: str) -> str:
        return " ".join((message or "").split()[:6]) or "leadgen"

    def _final(self, raw_text: str, book: str, deliverable: bool, feedback: str,
               file_path: str | None = None) -> dict:
        """Shape a finished teacher result (no book pass needed). file_path is set when
        a real file was written, so the orchestrator can surface it in the response."""
        return {
            "raw_text": raw_text,
            "book_used": book,
            "tokens": 0,
            "domain": self.domain,
            "deliverable": deliverable,
            "self_check": {"passes": True, "feedback": feedback},
            "file_path": file_path,
        }

    # ── SINGLE-BUSINESS RESEARCH (Tavily + book) ──
    async def _run_single_research(self, ctx: dict) -> dict:
        """Deep web-presence research on ONE named business.

        The book must NEVER invent a business. Two distinct empty cases:
          * search RAN and found nothing about it → short-circuit with a plain
            not-found flag (no book call, no fabrication).
          * search COULDN'T run (no key / error) → we don't know it doesn't exist,
            so the book works from its own knowledge and marks unverified facts
            'unknown'."""
        target = ctx.get("message") or ctx.get("query", "")
        searched, results = await self._gather_research(ctx)

        if searched:
            relevant = self._filter_relevant(target, results)
            if not relevant:
                print(f"[leadgen] research ran but no relevant results for {target!r} "
                      "-> not-found flag (no book call)")
                return self._research_not_found(target)
            ctx = {**ctx, "live_research": relevant}
        else:
            ctx = {**ctx, "live_research": []}

        return await super().run(ctx)

    async def _gather_research(self, ctx: dict) -> tuple:
        """(searched, results). `searched` is False ONLY when the search could not
        run at all (no key / request error) — distinct from running and finding
        nothing. That distinction stops us telling Sir a business 'may not exist'
        when really the search never ran."""
        target = ctx.get("message") or ctx.get("query", "")
        if not target.strip():
            return False, []
        self._assert_in_manifest(_TAVILY)
        try:
            from tools import tavily
            return True, await tavily.search(target)
        except Exception as e:
            print(f"[leadgen] live research unavailable: {e}")
            return False, []

    def _name_tokens(self, target: str) -> set:
        """Distinctive identity tokens from the target — the business's actual name,
        with trade descriptors and request filler stripped out."""
        toks = re.findall(r"[a-z0-9]+", (target or "").lower())
        return {t for t in toks if len(t) >= 3 and t not in _GENERIC_NAME_TOKENS}

    def _filter_relevant(self, target: str, results: list) -> list:
        """Keep only results that actually mention the business — a distinctive name
        token in the title or content. If the target has no distinctive token to
        match on, don't over-filter (better to let the book work than wrongly cry
        'not found')."""
        names = self._name_tokens(target)
        usable = [r for r in results if (r.get("title") or r.get("content"))]
        if not names:
            return usable
        kept = []
        for r in usable:
            blob = f"{r.get('title', '')} {r.get('content', '')}".lower()
            if any(n in blob for n in names):
                kept.append(r)
        return kept

    def _research_not_found(self, target: str) -> dict:
        """A clean not-found result for the single-business path — no book call,
        nothing fabricated."""
        query = (target or "").strip()
        flag = (
            f"{_NOT_FOUND_PREFIX} a live web search for \"{query}\" returned nothing about "
            "this business. It may not exist or have no web presence. Do NOT fabricate any "
            "services, contact, social, or fit details. Tell Sir plainly it wasn't found, and "
            "suggest he broaden the search or try a different / more exact name."
        )
        return self._final(flag, book="none", deliverable=False,
                           feedback="research ran, no relevant results — not-found flag")

    def _places_block(self, live_places: list) -> str:
        """Format verified Places rows as the factual list of businesses to profile — used
        on the combined path so the dossier is anchored to REAL directory data, not names
        the book might invent."""
        if not live_places:
            return ""
        lines = "\n".join(self._format_row(i, r) for i, r in enumerate(live_places, start=1))
        return ("\n\nVERIFIED LOCAL BUSINESSES (REAL Google Places rows — these are the "
                "businesses to profile; use their name/phone/website/address EXACTLY as given "
                "and never invent extra leads or alter a field):\n" + lines)

    def _research_block(self, live_research: list) -> str:
        """Format real web results as the factual basis for the dossier."""
        if not live_research:
            return ("\n\nLIVE WEB RESEARCH: none available (search returned nothing or "
                    "is not configured). Work from what you reliably know; mark anything "
                    "you cannot verify as \"unknown\" — do not invent it.")
        lines = "\n".join(
            f"- {h['title']} ({h['url']})\n  {h['content']}".rstrip()
            for h in live_research if h.get("title") or h.get("content")
        )
        return ("\n\nLIVE WEB RESEARCH (REAL results — use these as your factual basis; "
                "extract services/contact/social/fit from them, do not contradict or invent "
                "beyond them):\n" + lines)

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        target = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown leadgen context (ICP, past leads, what Sir cares about):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        places_block = self._places_block(ctx.get("live_places", []))
        research_block = self._research_block(ctx.get("live_research", []))

        # On the combined path multiple verified businesses may be listed above — profile
        # each one; otherwise this is a single named target and one dossier is right.
        per_business = ("\n\nIf multiple VERIFIED LOCAL BUSINESSES are listed above, return "
                        "one dossier block per business, each with the sections below.") \
            if places_block else ""

        return f"""You are a lead-research analyst. Produce a factual research dossier on the target below. \
Output RAW structured research only — no greeting, no opinion of your own voice, no outreach copy, no sign-off.

TARGET:
{target}{memory_block}{places_block}{research_block}{per_business}

Return the following sections. If a fact is unknown, write "unknown" — do not invent it.

SERVICES: what they offer / sell.
CONTACT: email, phone, website, physical location.
SOCIAL: instagram / linkedin / other handles and follower scale if known.
FIT SIGNALS: concrete evidence for or against this being a good lead — size, activity, recent posts, gaps a service could fill, budget cues.
NOTES: anything else materially useful for qualifying this lead.

Be specific and concise. Facts only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        # Final/true material (not-found flag, verified Places rows, export confirmation)
        # is a deliberate, correct outcome — never a quality failure. Pass it verbatim so
        # the looper doesn't re-ask a book and invent something.
        if text.startswith(_PASSTHROUGH_PREFIXES):
            return {"passes": True, "feedback": "final material — passed through"}

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real research"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of researching"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable research substance (services/contact/social/fit)"}

        return {"passes": True, "feedback": "on-topic research with substance"}
