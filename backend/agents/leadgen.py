import os
import re

from agents.base import Teacher

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
_EXPORT_PREFIX = "CALL SHEET EXPORTED:"     # xlsx written to outputs/
_PASSTHROUGH_PREFIXES = (_NOT_FOUND_PREFIX, _PLACES_PREFIX, _EXPORT_PREFIX)

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
# it. "tavily" is the shared pool's Tavily web-search tool (registry name "search");
# "places" is Google Places. Leadgen uses each tool's STRUCTURED layer (search_places /
# web_search return dicts) rather than the registry's string wrapper — it needs the raw
# fields (phone/website/address) for the call sheet — but the two names still name the
# same tools the registry exposes.
_PLACES = "places"
_TAVILY = "search"          # Tavily web search, as named in tools/registry.py
MANIFEST = (_PLACES, _TAVILY)

# Words that mean "hand me the leads as a file" — any of these in a message is an
# export request. Leadgen always exports the rich .xlsx call sheet (openpyxl).
_EXPORT_INTENT_SIGNALS = (
    "call sheet", "callsheet", "call-sheet", "spreadsheet", "sheet",
    "xlsx", "excel", "csv", "export", "download",
)


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
      * PROSPECTING — "find me <category> in <place>" → Google Places returns
        VERIFIED business rows (name/phone/website/address/rating/reviews). No book
        in the loop, so nothing can be invented. Empty → an honest not-found flag.
      * SINGLE-BUSINESS RESEARCH — deep web-presence dig on ONE named business →
        Tavily, then a book frames a dossier from the REAL results.
    Never writes the outreach, never adds personality."""

    domain = "leadgen"
    memory_ns = "mem_leadgen"
    deliverable = True  # lead research is inherently a structured deliverable

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

        # The brain PICKS its tool(s) for this task — rules only, no LLM call (quota).
        tools = self._select_tools(message)
        print(f"[leadgen] tool selection for {message!r}: {tools}")

        if tools == [_PLACES]:
            return await self._run_prospecting(ctx)      # local directory only
        if tools == [_TAVILY]:
            return await self._run_single_research(ctx)  # web dig on one business
        return await self._run_combined(ctx)             # both: local + web context

    def _select_tools(self, message: str) -> list[str]:
        """Rules-based (NO LLM — quota) pick of which manifest tool(s) this task needs:
          * a list/prospecting ask ("find studios in Mumbai")        → [places]
          * a deep dig on ONE named business ("research Luma Labs")   → [tavily]
          * an ask that needs a local directory AND web context      → [places, tavily]
            (e.g. "find studios in Mumbai and research their reputation")
        Anything ambiguous defaults to Places — the safe superset that returns real data
        or an honest nothing. The result is intersected with MANIFEST so the boundary is
        explicit: leadgen can never emit a tool outside it."""
        low = (message or "").lower()
        has_single = any(s in low for s in _SINGLE_RESEARCH_SIGNALS)
        has_list = (any(re.search(rf"\b{re.escape(v)}", low) for v in _LIST_VERBS)
                    or any(re.search(rf"\b{re.escape(n)}\b", low) for n in _PROSPECT_NOUNS))

        if has_single and has_list:
            selected = [_PLACES, _TAVILY]   # named-business research scoped to a local list
        elif has_single:
            selected = [_TAVILY]            # one named business, web presence only
        else:
            selected = [_PLACES]            # default / prospecting: real directory data

        return [t for t in selected if t in MANIFEST]

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

    # ── PROSPECTING (Google Places) ──
    async def _run_prospecting(self, ctx: dict) -> dict:
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
            from tools.search import web_search
            return True, await web_search(target)
        except Exception as e:
            print(f"[leadgen] live research unavailable: {e}")
            return False, []

    # ── COMBINED (Places + Tavily → book) ──
    async def _run_combined(self, ctx: dict) -> dict:
        """The ask needs a local directory AND web context (e.g. "find studios in Mumbai
        and research their reputation"). Fetch BOTH manifest tools, inject both real result
        sets, and let the book reason a dossier grounded in them. If BOTH come back empty →
        an honest not-found flag (no book call, nothing fabricated)."""
        message = ctx.get("message") or ctx.get("query", "")
        region = self._detect_region(message)

        rows = await self._safe_places(message, region)
        searched, results = await self._gather_research(ctx)
        relevant = self._filter_relevant(message, results) if searched else []

        print(f"[leadgen] combined query={message!r} -> {len(rows)} places row(s), "
              f"{len(relevant)} relevant web result(s)")

        if not rows and not relevant:
            return self._combined_not_found(message)

        # Real Places rows are still a cacheable lead set — a later "call sheet for those"
        # exports them without re-querying, same as the pure prospecting path.
        if rows:
            cache_leadgen_results(ctx.get("session_id", "default"), message, rows)

        ctx = {**ctx, "live_places": rows, "live_research": relevant}
        return await super().run(ctx)

    def _combined_not_found(self, message: str) -> dict:
        """Both Places and the web search ran and returned nothing usable. No book call,
        no fabrication — an honest not-found flag."""
        query = (message or "").strip()
        flag = (
            f"{_NOT_FOUND_PREFIX} neither Google Places nor a live web search returned "
            f"anything for \"{query}\". Tell Sir plainly nothing was found. Do NOT fabricate "
            "any names, contacts, ratings, websites, or web-presence details. Suggest he "
            "broaden the area/category or try different keywords."
        )
        return self._final(flag, book="none", deliverable=False,
                           feedback="places + web ran, zero results — not-found flag")

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
