import asyncio
import re

from core.router import SYSTEM_PROMPT, _extract_mood
from core.books import select_book, select_book_for_tier, select_fast_book, call_book, model_for
from core.profile import get_relevant_facts
from core.memory import save_message
from core.extractor import extract_and_store
from core.action_dispatch import get_cached_reply, record_reply
from core.provider_health import circuit_summary, open_providers
from core.trace import stage_of, mark

# Her generation never needs raw horsepower the way research does — it needs to
# reason and sound like herself. "good" keeps the free cloud books in play.
_GEN_SPEC = {"reasoning": "good"}

# Backstop for prompt scaffolding the model sometimes echoes AFTER her answer —
# "[Raw material analysis]", a bracketed analysis/reasoning header, then notes on
# mood/intent. Only her reply should survive, so cut from the first such marker to
# the end. Keyword-gated so a stray legit bracket in her reply isn't nuked. The mood
# tag is extracted (and removed) BEFORE this runs, so it's never the trigger here.
_ANALYSIS_RE = re.compile(
    r"\s*(?:"
    r"\[[^\]]*\b(?:raw material|analysis|reasoning|thinking|mood|intent|phrase|delivery|meta)\b[^\]]*\]"
    r"|\*{0,2}\s*raw material(?:\s+analysis)?\s*\*{0,2}\s*:"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_analysis(text: str) -> str:
    """Remove any trailing analysis/reasoning scaffolding the model leaked."""
    return _ANALYSIS_RE.sub("", text or "").strip()


# Quote-wrapping backstop — the action path especially comes back with the WHOLE reply
# sitting inside double quotes ("Reminder saved to drink water..."), as if she were
# quoting herself. Openers/closers checked as pairs so smart quotes are covered too.
_QUOTE_PAIRS = (('"', '"'), ("“", "”"))


def _strip_quote_wrap(text: str) -> str:
    """Unwrap a reply that is entirely one quoted span. Only strips when the SAME quote
    opens and closes the whole string AND neither quote char appears inside it — so a
    reply that legitimately quotes something ('He said "no", Sir.') or that stitches two
    quoted fragments together is left exactly as it is."""
    t = (text or "").strip()
    for open_q, close_q in _QUOTE_PAIRS:
        if len(t) >= 2 and t.startswith(open_q) and t.endswith(close_q):
            inner = t[1:-1]
            if open_q not in inner and close_q not in inner:
                return inner.strip()
    return t


# ── Empty-section guard (deliverables) ──
# The ensemble combine is done by cognition's model, not code: it's handed two source
# blocks and told to keep the section structure. Under that load it sometimes prints the
# section skeleton (OVERVIEW:, KEY CONCEPTS:, …) but drops the body — bare labels with no
# content beneath. This is the deterministic backstop: after her reply is generated, drop
# any deliverable section header that has nothing under it, so a bare label can NEVER
# render. A header that DOES have content (inline or on the lines before the next header)
# is left untouched, so a full deliverable passes through unchanged.
#
# A "section header" here is an ALL-CAPS-style label ending in a colon (OVERVIEW:,
# KEY CONCEPTS:, FIT SIGNALS:, SOURCE/URL:, WHY-FIT: — the labels every deliverable teacher
# emits), optionally wrapped in markdown emphasis/bullets. The uppercase-only label keeps
# it from ever matching her lowercase prose ("Here's the shape, Sir:"). group(2) is any
# content sitting inline after the colon.
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?:[*_#>]+\s*|-\s+)?"          # optional leading emphasis/bullet marks
    r"([A-Z][A-Z0-9 &/'’\-]{1,38}?)"     # the label — uppercase run, no lowercase prose
    r"\**\s*:\s*(.*?)\s*$"               # colon (allowing a trailing **) + inline content
)


def _has_real_content(s: str) -> bool:
    """True when a section body carries actual text — not just leftover emphasis/bullet/
    dash punctuation or whitespace the model left behind under an empty header."""
    return re.sub(r"[*_#>\-\s]+", "", s or "") != ""


def _drop_empty_sections(text: str) -> str:
    """Remove every deliverable section header that has no content under it. A header is
    kept iff it has real text either inline after the colon OR on the lines up to the next
    header. Non-header prose (her framing line, list items, ROLE N group labels) is never
    matched, so a fully-filled deliverable renders unchanged."""
    lines = (text or "").split("\n")

    headers = []  # (line_index, inline_content)
    for i, ln in enumerate(lines):
        m = _SECTION_HEADER_RE.match(ln)
        if m:
            headers.append((i, m.group(2)))

    if not headers:
        return text

    drop = set()
    for k, (i, inline) in enumerate(headers):
        end = headers[k + 1][0] if k + 1 < len(headers) else len(lines)
        body = inline + "\n" + "\n".join(lines[i + 1:end])
        if not _has_real_content(body):
            drop.update(range(i, end))  # the bare header + the blank span it owns

    if not drop:
        return text
    return "\n".join(ln for j, ln in enumerate(lines) if j not in drop).strip()


# ── Request-aware output sizing ──
# The deliverable/list tiers below only fire when a TEACHER set the flag. A casual
# "find me 10 garages" hits no teacher, so without this it fell to the 300 cap and
# died mid-list. So size the cap to what Sir actually ASKED for, deterministically
# (no extra LLM): an explicitly-sized list must never truncate.
_PER_ITEM_TOKENS = 120   # rough room for one list item (name + a line of detail)

# ── Output token caps by tier ──
# Sized so a NORMAL answer FINISHES. The old 300 default cut real explanations (CAP
# theorem, caching strategy) off mid-sentence, and the 80 trivial cap clipped anything
# with a second thought. These are OUTPUT caps only — the model stops when it's done, so
# a larger cap just prevents truncation; it is NOT a target length (her voice stays terse).
_CAP_BY_COMPLEXITY = {"trivial": 150, "simple": 512, "complex": 1500}
_CAP_DELIVERABLE = 2048  # a structured deliverable / explicit list — never clip it
# Nemotron Super/Ultra have HUGE context — a strong/frontier turn must not be pinned at a
# Groq-ish size. Scale its answer cap up so real multi-part reasoning lands in full.
_CAP_NEMOTRON = 4096

_ROOMY_FLOOR = _CAP_DELIVERABLE   # an unsized list still gets the full deliverable floor
_ROOMY_CEILING = _CAP_NEMOTRON    # hard ceiling so a huge "list 500" can't drain quota
_LEADGEN_CEILING = 8000  # leadgen lists are VERIFIED real data — never truncate; size to fit

# A count tied to a list ask: "10 garages", "top 5 ideas", "find me 15", "list 20".
# The noun/verb context keeps it from sizing on incidental numbers in stray prose.
_COUNT_RE = re.compile(
    r"\b(?:top\s+)?(\d{1,3})\s+[a-z]"                       # "10 garages", "top 5 ideas"
    r"|\b(?:give|find|get|show|name|list|fetch)\s+(?:me\s+)?(\d{1,3})\b",  # "find me 10"
    re.IGNORECASE,
)
# A bare list/enumeration ask with no number ("list the …", "every …", "steps to …").
_LIST_VERB_RE = re.compile(
    r"\b(list|enumerate|itemi[sz]e|rundown|breakdown|every|all\s+the|"
    r"steps?\s+to|bullet)\b",
    re.IGNORECASE,
)


def _request_cap(message: str) -> int | None:
    """If Sir explicitly sized a list/long answer, return a token cap scaled to it so
    it can't truncate; else None. ~120 tokens/item when a count is given, floored at
    the deliverable size and ceilinged to protect quota. Deterministic, no LLM."""
    msg = message or ""
    m = _COUNT_RE.search(msg)
    n = int(next(g for g in m.groups() if g)) if m else None
    if n is None and not _LIST_VERB_RE.search(msg):
        return None
    if n is None:
        return _ROOMY_FLOOR
    return max(_ROOMY_FLOOR, min(n * _PER_ITEM_TOKENS, _ROOMY_CEILING))


def _output_cap(ctx: dict, material, req_cap: int | None) -> int:
    """The hard OUTPUT-length cap for this turn, sized to what the turn actually is —
    complexity tier, deliverable/list flag, an explicit request size, and the reasoning
    tier (Nemotron gets Nemotron-scale room, never a Groq-ish cap). Deterministic, no LLM.

    Order: leadgen verified list → deliverable/list → general-path gen_tier → complexity;
    then Nemotron scaling and the explicit request size raise (never lower) the result."""
    # A leadgen deliverable is a VERIFIED Places list that must NEVER be cut off. Size the
    # cap to the material itself (≈chars/3 + headroom) so every real row survives, up to a
    # generous ceiling — independent of the modest prose caps below.
    if ctx.get("domain") == "leadgen" and ctx.get("deliverable") and material:
        sized = int(len(material) / 3) + 500
        return max(_CAP_DELIVERABLE, req_cap or 0, min(sized, _LEADGEN_CEILING))

    gen_tier = ctx.get("gen_tier")
    # A deliverable or an explicit 'list' action is structured — give it the deliverable
    # floor so the block never clips. (task_block framing keys off ctx['deliverable']
    # separately; this only sizes the cap.)
    if ctx.get("deliverable") or ctx.get("action") == "list":
        base = _CAP_DELIVERABLE
    # General path (domain=none): gen_tier already folded complexity/loop_worthy in —
    # 'strong' is complex-grade reasoning, 'fast' a real short answer, 'fast_lane' small talk.
    elif gen_tier == "strong":
        base = _CAP_BY_COMPLEXITY["complex"]
    elif gen_tier == "fast":
        base = _CAP_BY_COMPLEXITY["simple"]
    elif gen_tier == "fast_lane":
        base = _CAP_BY_COMPLEXITY["trivial"]
    else:
        # Teacher/action path (no gen_tier): size off the triage complexity. Unknown →
        # 'simple', so a real answer never falls back to a small-talk cap.
        base = _CAP_BY_COMPLEXITY.get(ctx.get("complexity"), _CAP_BY_COMPLEXITY["simple"])

    # Nemotron scaling: a 'strong' (Super) or loop_worthy/frontier (Ultra) turn runs on a
    # huge-context book — let the answer breathe rather than cap it at a Groq-ish size.
    if gen_tier == "strong" or ctx.get("loop_worthy"):
        base = max(base, _CAP_NEMOTRON)

    # An explicitly-sized list ask ("find me 10 garages") can only RAISE the cap.
    return max(base, req_cap or 0)


# ── List-type deliverable detection (leads, jobs, any row-per-item material) ──
# The generic deliverable task_block below was written for HEADER-style dossiers
# (SERVICES:/CONTACT:/SOCIAL:/...) and only asks the model to "keep the section
# structure" — nothing tells it to preserve line breaks between discrete items, so
# a 10-row Places list (leadgen._format_row: "1. Name\n   Phone: ...\n   ...") was
# free to melt into one prose paragraph. Two-or-more "N. " line starts is the
# signal a teacher already emitted a row-per-item list (as opposed to prose or
# ALL-CAPS section headers), regardless of which domain produced it.
_LIST_ROW_RE = re.compile(r"^\s*\d{1,3}\.\s+\S", re.MULTILINE)


def _is_list_material(material: str) -> bool:
    return len(_LIST_ROW_RE.findall(material or "")) >= 2


def _format_history(history: list) -> str:
    if not history:
        return "(nothing yet)"
    lines = []
    for turn in history:
        who = "Sir" if turn.get("role") == "user" else "You"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)


# ── Local-path history window ──
# The local book (ollama llama3.2, 3B) has a small effective context and latches onto
# whatever dominates its prompt. A 4-message (2-exchange) window still bled: with secret
# mode's studios turn immediately followed by a workout ask, the studios turn WAS the
# most recent exchange, so any window that includes "the last exchange" still hands the
# 3B "Studio Nine, ..." right before the new question — it doesn't matter how tight the
# window is if the offending turn is inside it. So the LOCAL/secret path gets NO stored
# history at all: only the live message (surfaced separately below) drives the reply.
# This is "last 1 turn max" in the sense that matters — the one turn the 3B sees is the
# CURRENT one, not anything retrieved from prior storage. Cloud books are completely
# unaffected — this changes nothing about WHAT may be injected (privacy guard), only how
# much prior conversation the 3B is ever shown.
_LOCAL_HISTORY_MSGS = 0


# ── Local path: a bare handover of sensitive data ──
# "my pan number is 1bsjfdaa2" is not a question. There is nothing to answer, no
# material, no task — and that is exactly what broke. The full prompt below hands the
# 3B a persona, a memory block, a RAW MATERIAL slot reading "(none — you're working
# from yourself here)" and a task line telling it to "decide how to handle this for
# Sir". Given a blank to fill and no facts to fill it with, a 3B fills it: it invented
# a corrected PAN ("The correct pan number is 1BSJFDAA2-01"), claimed a lookup that
# never happened, drifted to a shipment notification, and proposed verifying the
# number. Every one of those is the model answering a question nobody asked.
#
# So this turn gets its OWN prompt: short, taught by example rather than by rules, and
# — the load-bearing part — carrying none of the message. See the note on
# _LOCAL_DISCLOSURE_PROMPT below for why the value and the label are both withheld.
#
# Deliberately excluded too: the persona block, retrieved memory and conversation
# history. They are the rest of the confabulation surface (the invented "shipment
# notification" has the shape of history bleed). The voice is carried by the examples
# instead, which is all a one-line acknowledgement needs.
#
# Scope: LOCAL path only (see local_only in _cognition_pass), no material, no action.
# This changes no routing, no tier and no vault behaviour — the turn was already
# secret, already ollama-only, and core.extractor already vaults it. This only fixes
# what she SAYS back.

# The labels that make a value sensitive are Sir's own, in privacy.local, already
# parsed by triage. Reused rather than copied: a label he adds there has to work here
# too, and a second list would drift silently. An unreadable/missing privacy.local
# yields an empty tuple, which simply means this path never fires — the fail-safe
# direction, since the old prompt still runs.
try:
    from core.triage import _RULES as _PRIVACY_RULES
    _SENSITIVE_LABELS = tuple(
        re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
        for w in (_PRIVACY_RULES.get("patterns") or ()))
except Exception as e:  # pragma: no cover — never break cognition over a rules file
    print(f"[cognition] privacy labels unavailable ({e}) — local disclosure path off")
    _SENSITIVE_LABELS = ()

# Anything that makes the turn a REQUEST rather than a handover. Any hit and the
# normal prompt runs, so "can you check my pan 1234" keeps its existing behaviour.
# Note what is NOT here: save / store / note / remember. "Remember my pan is X" is
# still a handover, and the answer to it is still "noted, it's held".
_TASK_RE = re.compile(
    r"\b(what|whats|when|where|why|how|who|which|whose|can you|could you|would you|"
    r"will you|please|find|look ?up|check|verify|confirm|validate|tell|show|give|"
    r"send|explain|help|make|write|draft|calculate|compare|list|search|fix|update)\b",
    re.IGNORECASE)

# The handover shape: a marker, then a value carrying a digit, and the value ENDS the
# message — "…is 1bsjfdaa2", "…number: 4821 9930 1174". The end anchor is what keeps
# "my bank balance dropped in 2024" out: a value mentioned in passing isn't handed
# over. The digit is what keeps "my bank account is closed" out.
_HANDOVER_RE = re.compile(
    r"(?:\bis\b|\bare\b|\bwas\b|=|:)\s*"        # marker
    r"(?=[\w@.\-]*\d)"                          # the value carries a digit
    r"[\w@.\-]{3,}(?:[ \-][\w@.\-]{2,})*"       # value, possibly grouped (4821 9930)
    r"\s*[.!]?\s*$",                            # …and it ends the message
    re.IGNORECASE)

# A credential's value need not carry a digit — "my password is correcthorse" is still
# a handover. These labels have no non-disclosure reading, so the digit test is
# dropped for them (and only them).
_CREDENTIAL_RE = re.compile(
    r"\b(password|passwd|passphrase|pin|otp|cvv|api key|secret key|token)\b",
    re.IGNORECASE)
_HANDOVER_ANY_RE = re.compile(r"(?:\bis\b|\bare\b|=|:)\s*\S{3,}\s*[.!]?\s*$",
                              re.IGNORECASE)

_MAX_DISCLOSURE_WORDS = 20   # a handover is one short line; anything longer is a turn


def _is_bare_disclosure(message: str) -> bool:
    """True when the message is ONLY a piece of sensitive data being handed over —
    a sensitive label, a value at the end of the line, and nothing asked."""
    msg = (message or "").strip()
    if not msg or len(msg.split()) > _MAX_DISCLOSURE_WORDS:
        return False
    if "?" in msg or _TASK_RE.search(msg):
        return False
    if not any(label.search(msg) for label in _SENSITIVE_LABELS):
        return False
    if _HANDOVER_RE.search(msg):
        return True
    return bool(_CREDENTIAL_RE.search(msg) and _HANDOVER_ANY_RE.search(msg))


# THE PROMPT HAS NO MESSAGE SLOT. That is the fix, not an oversight.
#
# The value cannot be read back, corrected, completed or "verified" because it is
# never in the prompt to begin with. No instruction can be as reliable as an absence
# on a 3B: told "never repeat the value" while holding the value, it still produced
# "The correct pan number is 1BSJFDAA2-01". Nothing here is information-dependent —
# the only true answer to a bare handover is a one-line acknowledgement — so the
# model is given the situation and none of the content.
#
# The label is withheld for a separate, measured reason: llama3.2's safety training
# fires on the WORDS "password", "otp", "account number", not just on a value. With
# the label present (even with the value redacted) roughly 40% of runs came back
# "I cannot store sensitive information such as passwords" — a refusal that is both
# wrong and unhelpful, since core.extractor has ALREADY vaulted the turn. Naming the
# vault write as done, with nothing sensitive in view, removes the trigger entirely:
# 18/18 clean runs, no refusals.
#
# Three demonstrations, one line of instruction. A 3B imitates far better than it
# obeys, and every example shows the same three things — acknowledge, say it's held,
# stop.
_LOCAL_DISCLOSURE_PROMPT = """You are AIOS, Sir's assistant, running on his own machine. He has just filed a private note in his own encrypted vault. You cannot see what is in it. It is already saved — nothing is being asked of you.

Say ONE short line, in your voice, confirming it is filed. Never ask a question.

Sir: [filed a private note]
You: Noted, Sir. It's in the vault.
[mood: neutral]

Sir: [filed a private note]
You: Filed, Sir. Locked away.
[mood: neutral]

Sir: [filed a private note]
You: Got it — safe with me, Sir.
[mood: neutral]

Sir: [filed a private note]
You:"""


def _format_material(raw_material):
    """Normalise teacher output into a text block. Returns (block, is_ensemble).
    A list means either a per-task book ensemble (agents.base.Teacher._ensemble_pass,
    fired only when the brain set ensemble=true for this task) or a multi-domain
    fan-out (Slice 8) handed back >1 raw output — she is the combiner either way."""
    if raw_material is None:
        return None, False

    if isinstance(raw_material, list):
        texts = []
        for item in raw_material:
            t = item.get("raw_text", "") if isinstance(item, dict) else str(item)
            if t and t.strip():
                texts.append(t.strip())
        if not texts:
            return None, False
        if len(texts) == 1:
            return texts[0], False
        block = "\n\n".join(
            f"── Source {chr(65 + i)} ──\n{t}" for i, t in enumerate(texts)
        )
        return block, True

    if isinstance(raw_material, dict):
        t = (raw_material.get("raw_text") or "").strip()
        return (t or None), False

    t = str(raw_material).strip()
    return (t or None), False


def _build_prompt(ctx: dict, material: str, is_ensemble: bool,
                  local_history: bool = False) -> str:
    message = ctx.get("message") or ctx.get("query", "")

    # her_memory_of_user — core personal memory (mem_core == the existing
    # long_term_memory store, read via profile.get_relevant_facts).
    # Retrieval guard: if this turn will be answered on a cloud book, only public
    # facts may be injected. secret turns run local (ollama), so private is allowed
    # there. The cloud guard must hold regardless — default to the safe side.
    cloud_bound = ctx.get("sensitivity") != "secret"
    facts = get_relevant_facts(message, cloud_bound=cloud_bound)
    memory_block = "(nothing relevant)"
    if facts:
        memory_block = "\n".join(f"- {f}" for f in facts)

    # local_history → this turn answers on the 3B local book: tight window (see
    # _LOCAL_HISTORY_MSGS) so a stale topic can't bleed into the new answer. Cloud
    # books keep the full session history unchanged. Sliced explicitly rather than via
    # history[-_LOCAL_HISTORY_MSGS:] — a negative-zero slice (history[-0:]) is a classic
    # Python trap that silently returns the WHOLE list, not zero items.
    history = ctx.get("history", [])
    if local_history and len(history) > _LOCAL_HISTORY_MSGS:
        print(f"[cognition] local path — history trimmed to last "
              f"{_LOCAL_HISTORY_MSGS} of {len(history)} messages")
        history = history[len(history) - _LOCAL_HISTORY_MSGS:] if _LOCAL_HISTORY_MSGS > 0 else []
    history_block = _format_history(history)

    # ── Action path: no prior conversation. ──
    # An action confirmation's only truth is the RAW MATERIAL below — the REAL tool
    # result for THIS turn. Prior turns add nothing to it, and they are exactly how the
    # previous confirmation ("Reminder to drink water in 10 minutes, Sir done.") got
    # replayed as the answer to a NEW reminder: the model copies the nearest matching
    # line it can see. Every other path keeps its history window untouched.
    if ctx.get("action_result") is not None:
        history_block = ("(withheld — this turn confirms an action; the raw material "
                         "below is the only truth for it)")

    if material is None:
        material_block = "(none — you're working from yourself here)"
    elif is_ensemble:
        material_block = (
            "Two independent research passes ran on this. They may overlap or "
            "conflict. You are the one who reconciles them — there is no separate "
            "combiner. Weigh both, resolve disagreements, keep what's solid:\n\n"
            + material
        )
    else:
        material_block = material

    if ctx.get("action_result") is not None:
        # The RAW MATERIAL above is the TRUE result of an action that already ran.
        # Branch on the REAL ok flag so she can never narrate a success the tool did
        # not return — this is what stops "Email sent successfully" on an ok:False.
        if ctx["action_result"].get("ok"):
            task_block = (
                "You just performed a real action for Sir and it SUCCEEDED. The RAW "
                "MATERIAL above is the ACTUAL result — confirm it in your voice, minimally "
                "and truthfully, inventing nothing beyond it. Confirm ONLY what THIS "
                "material states: never an earlier action's task, time or details. "
                "ACTOR CHECK (get this right — it inverts easily): HE asked, YOU acted. "
                "He did not remind/set/log/save/mark/clear anything — YOU did, for HIM. "
                "Say it as 'I've set/logged/saved/cleared ... for you' — NEVER as 'you "
                "reminded/asked/told me to ...' or any phrasing where Sir is the one who "
                "performed the action on you. "
                "IMPORTANT: if the material "
                "says the action is STAGED / NOT yet sent (e.g. an email draft awaiting "
                "confirmation), present it and ASK for confirmation — do NOT say it was "
                "sent or done."
            )
        else:
            task_block = (
                "The action you attempted for Sir did NOT succeed — the RAW MATERIAL above "
                "says ACTION NOT DONE. You MUST NOT claim it worked, sent, saved, or "
                "completed. Do not use words like 'sent', 'done', or 'successfully'. Do NOT "
                "restate, paraphrase, or summarize his request back to him as if you're "
                "acknowledging or handling it — that reads as agreement when nothing "
                "happened. State plainly, in your voice, what actually happened (usually: "
                "you can't do that) and relay the exact ask / next step from the material — "
                "nothing more."
            )
    elif ctx.get("deliverable") and _is_list_material(material):
        task_block = (
            "This is a deliverable LIST for Sir — discrete items (leads, jobs, whatever the "
            "material is), not a narrative. Open with ONE line in your voice framing it. Then "
            "render the items as a MARKDOWN NUMBERED LIST, exactly one item per number: the "
            "item's name in **bold** on its own line, then its remaining fields (phone, "
            "address, rating, website, salary, whatever the material actually gives) each on "
            "their OWN line directly beneath it, with a blank line between items. Use real "
            "line breaks — NEVER run two items together, and NEVER run two fields of the same "
            "item together, in one sentence or paragraph. Carry every real field from the "
            "material verbatim (never invent or drop one); only omit a field if the material "
            "has none for it. Do not pad, do not add commentary per item. End with one line "
            "only if a real next step exists."
        )
    elif ctx.get("deliverable"):
        task_block = (
            "This is a deliverable for Sir. Open with ONE line in your voice framing it "
            "(e.g. what this is / your read on it). Then present the material as a CLEAN, "
            "SCANNABLE STRUCTURED BLOCK — keep the section structure from the raw material "
            "(headers, fields), do not melt it into a paragraph, do not pad. If TWO research "
            "passes are shown above, MERGE them into ONE set of sections — fill each section "
            "with the combined facts from BOTH passes; never print the section set twice and "
            "never leave a header empty because the two disagreed. FILL every header you "
            "print with its real content from the material. If a section genuinely has no "
            "content, OMIT that header entirely — never print a bare label with nothing under "
            "it. Tighten and format it well; remove filler and any 'unknown' noise that adds "
            "nothing. End with one line only if a real next step exists. The structure IS the "
            "value — present it, don't dissolve it."
        )
    else:
        task_block = (
            "Reason as yourself. Decide how to handle this for Sir, how to present it, whether "
            "to push back, how to deliver. The material above is input to your thinking, not "
            "your answer — never hand it back raw, never narrate that you researched it. Speak "
            "in your own voice, as yourself."
        )

    return f"""── HOW YOU ANSWER (this binds hardest — over everything below) ──
This is your voice. Match it exactly:
  Sir: "Wow."                         You: "Right? Took long enough, Sir."
  Sir: "How are you?"                 You: "Sharp as ever. You?"
  Sir: "Can we start working now?"    You: "Ready when you are, Sir."
  Sir: "Thanks."                      You: "Always, Sir."

Rules:
- 1–2 sentences by default. One line when one line is the truth.
- Dry, warm, anticipates him. You land it and move on.
- No trailing question unless it genuinely earns one. Don't end every line with a question.
- No coaching, no life-advice tone, no hedging. No "I'd like to clarify", no "I want to make sure", no unprompted clarifying questions — read the room and answer. If it's truly ambiguous, take your best read and go.
- Expand past two sentences ONLY for a real deliverable — a list, real steps, structure he asked for. Length is earned, never default.
- You call him "Sir". Confirmations are minimal: "Done." "On it." "Noted." Nothing more unless he needs more.
- Output ONLY your reply. Never your reasoning, never section headers, never analysis of the raw material or of his mood/intent, never meta-commentary about how you're answering. The thinking is internal; the words you say are all he sees.

{SYSTEM_PROMPT}

── WHAT YOU KNOW ABOUT SIR (carry quietly, do not recite) ──
{memory_block}

── RECENT CONVERSATION ──
{history_block}

── RAW MATERIAL (input to your thinking — NOT your answer) ──
{material_block}

── SIR'S MESSAGE ──
{message}

── YOUR TASK ──
{task_block}

── OUTPUT (binds last) ──
Reply with ONLY the words you say to Sir, then the single mood tag the system asked for — nothing else. Do NOT emit any analysis, section header, or bracketed label (e.g. "[Raw material analysis]", "[Raw material]"), and do NOT explain your reasoning, his mood, or his intent. Your reasoning is internal; he sees only the answer."""


async def cognition_pass(ctx: dict, raw_material=None) -> dict:
    """THE mind — traced. The `cognition` span covers the whole pass (including the
    persist + extract tail); the individual book attempts inside it record themselves
    as nested `book` stages. Observation only: the reply is passed through untouched,
    and stage_of is inert when ctx carries no trace."""
    with stage_of(ctx, "cognition") as st:
        reply = await _cognition_pass(ctx, raw_material, st)
        st.set(book=reply.get("provider_used"), mood=reply.get("mood"))
        return reply


async def _cognition_pass(ctx: dict, raw_material, st) -> dict:
    """THE mind. Reasons over the teacher's raw material and speaks HER response
    in HER voice. The raw material is input to her thinking, never her answer."""
    # ── STALE ACTION GUARD ──
    # action_result is stamped by the orchestrator with the trace_id of the request that
    # produced it. If the stamp isn't THIS turn's trace, the result belongs to another
    # turn — drop it AND the material rendered from it, so she falls through to a plain
    # reply and can never confirm an action that didn't happen on this turn. ctx is
    # copied rather than mutated: the caller's dict is its own request's record.
    action_result = ctx.get("action_result")
    if action_result is not None and action_result.get("trace_id") != ctx.get("trace_id"):
        print(f"[cognition] WARNING: stale action_result — stamped "
              f"{action_result.get('trace_id')}, this turn is {ctx.get('trace_id')}. "
              f"Dropped; no action is referenced in this reply.")
        ctx = {k: v for k, v in ctx.items() if k not in ("action_result", "action")}
        raw_material = None
        action_result = None  # keep this var in sync — the replay check below must
                               # not act on a result that was just dropped as stale

    # ── IDEMPOTENT REPLAY ──
    # action_dispatch already stopped the TOOL from writing twice (run_action's own
    # cache — see core/action_dispatch.py). That alone still let a duplicate come back
    # worded two different ways: cognition phrases a fresh confirmation from an LLM on
    # every call, even when the underlying fact (the reused tool result) is identical.
    # A guarded action_result carries "_idem_key" and, on a replay, "_idem_replay":
    # True (both set by run_action). If a full reply was already recorded for that
    # exact key, hand it back byte-for-byte — no material, no prompt, no book call,
    # nothing persisted a second time for a request that never really happened twice.
    if action_result is not None and action_result.get("_idem_replay") and action_result.get("_idem_key"):
        cached_reply = get_cached_reply(action_result["_idem_key"])
        if cached_reply is not None:
            print(f"[cognition] idempotent replay {action_result['_idem_key'][:12]} — "
                  "reusing the stored reply verbatim, no new book call")
            st.set(idem_replay=True)
            mark(ctx, idempotency_hit=True)
            return cached_reply

    material, is_ensemble = _format_material(raw_material)

    # ── CLOUD GUARD (covers ALL retrieval: documents, memory, future tools) ──
    # Any RETRIEVED material injected this turn carries its sensitivity in
    # ctx["material_tier"]. Per the LLD the cloud guard must filter BOTH private AND
    # secret: retrieved private/secret content must NEVER reach a cloud book. So we
    # force the WHOLE pass local (ollama) when it fires — book selection AND the
    # fact extractor (which is itself a cloud call on the response). This is the one
    # guard layer; tools only tag ctx["material_tier"], they don't re-route.
    # Computed BEFORE the prompt is built: the local (3B) path needs to know it's
    # local so it can take the tight history window.
    base_tier = ctx.get("sensitivity", "public")
    material_tier = ctx.get("material_tier")
    local_only = base_tier == "secret" or material_tier in ("private", "secret")
    if material_tier in ("private", "secret"):
        print(f"[guard] retrieved tier={material_tier} -> local-only")
    # 'secret' is the system's local-only selector (ollama for books, vault for the
    # extractor, never cloud). Route the whole turn through it when the guard fires.
    route_tier = "secret" if local_only else base_tier

    # local_only ⇒ every book candidate is ollama (select_book*/select_fast_book all
    # return ["ollama"] for tier 'secret') — trim history for the 3B. Cloud turns
    # keep the full window.
    #
    # One exception, LOCAL PATH ONLY: a bare handover of sensitive data with no task
    # attached gets the short, example-led prompt above instead. The full prompt gives
    # a 3B a blank to fill and no facts to fill it with, and it invents (see
    # _is_bare_disclosure). Guarded on material/action being absent so a teacher turn
    # or an action confirmation is never diverted. The cloud path cannot reach this.
    disclosure = (local_only and material is None
                  and ctx.get("action_result") is None
                  and _is_bare_disclosure(ctx.get("message") or ctx.get("query", "")))
    if disclosure:
        print("[cognition] local path — bare sensitive handover, no task: "
              "acknowledge-and-hold prompt (message withheld from the model)")
        prompt = _LOCAL_DISCLOSURE_PROMPT
    else:
        prompt = _build_prompt(ctx, material, is_ensemble, local_history=local_only)

    # Hard cap on output length, independent of what the model wants to do. Sized to the
    # turn (complexity tier / deliverable / explicit request size / reasoning tier) so a
    # NORMAL answer finishes — the old 300 default cut explanations off mid-sentence.
    # _output_cap owns the ordering; see its docstring.
    req_cap = _request_cap(ctx.get("message") or ctx.get("query", ""))
    max_tokens = _output_cap(ctx, material, req_cap)
    print(f"[cognition] max_tokens={max_tokens} complexity={ctx.get('complexity')} "
          f"gen_tier={ctx.get('gen_tier')} deliverable={bool(ctx.get('deliverable'))} "
          f"action={ctx.get('action')} domain={ctx.get('domain')} req_cap={req_cap}")

    # Book selection. The general (domain=none) path is sized by the orchestrator and
    # handed down as ctx['gen_tier'] (fast_lane / fast / strong) so a real knowledge
    # question can't fall to the 8B just because the 3B triage under-rated it:
    #   - fast_lane → Groq's 8B (sub-second small-talk lane). The few-shot voice
    #     examples carry it; teacher/deliverable paths never reach here.
    #   - fast / strong → the tier book pool (groq 70b, or nemotron_super → groq →
    #     cerebras), honoring the same availability/fallback rules as the teachers.
    # Teacher/action turns set no gen_tier and keep the original behavior unchanged.
    gen_tier = ctx.get("gen_tier")
    fast_lane = (gen_tier == "fast_lane") or (
        gen_tier is None and ctx.get("complexity") == "trivial"
        and not ctx.get("deliverable"))
    if fast_lane:
        books = select_fast_book(route_tier)
    elif gen_tier in ("fast", "strong"):
        # gen_tier is ALREADY the final tier — the orchestrator folded complexity /
        # loop_worthy into it (complex/loop_worthy → 'strong'). Do NOT pass them again
        # or the upgrade signal would double-fire (strong→frontier→Ultra).
        books = select_book_for_tier(gen_tier, route_tier)
    else:
        books = select_book(_GEN_SPEC, route_tier,
                            complexity=ctx.get("complexity"), loop_worthy=ctx.get("loop_worthy"))

    # Everything the trace needs to explain this turn's routing, recorded before the
    # first call so a total book failure still leaves the decision visible.
    #
    # circuits/skipped are the OTHER half of that explanation: without them a trace
    # showing 'cognition ran on cerebras' gives no hint that groq was skipped because
    # its circuit was open, and the routing reads as an unexplained downgrade. Both
    # are None when every provider is healthy, so a normal turn's meta is unchanged.
    # Names and states only — see the whitelist in core/trace.py.
    st.set(gen_tier=gen_tier, fast_lane=fast_lane, lane="fast" if fast_lane else "full",
           local_only=local_only, material_tier=material_tier, max_tokens=max_tokens,
           deliverable=bool(ctx.get("deliverable")), domain=ctx.get("domain"),
           circuits=circuit_summary(), skipped=",".join(open_providers()) or None)
    mark(ctx, fast_lane=fast_lane, deliverable=bool(ctx.get("deliverable")))

    result = None
    for book in books:
        try:
            with stage_of(ctx, "book", book=book, model=model_for(book),
                          max_tokens=max_tokens) as bst:
                result = await call_book(prompt, book, max_tokens=max_tokens)
                bst.set(tokens_in=result.get("tokens_in"),
                        tokens_out=result.get("tokens_out"))
            break
        except Exception:
            continue  # call_book logged / benched it; fall to next candidate

    if result is None:
        return {
            "response": "Something's off with my connection, Sir. Give me a moment.",
            "mood": "neutral",
            "provider_used": "none",
            "tokens": 0,
        }

    # Surface fast vs full so latency wins are visible per turn. max_tokens is the cap we
    # imposed; tokens is what the call actually used (prompt + completion).
    print(f"[cognition] lane={'fast' if fast_lane else 'full'} "
          f"book={result['book_used']} model={model_for(result['book_used'])} "
          f"max_tokens={max_tokens} tokens={result['tokens']}")

    # Mood first (pulls + removes the [mood: x] tag), then strip any leaked analysis
    # scaffolding so only her spoken reply remains.
    response, mood = _extract_mood(result["raw_text"])
    response = _strip_analysis(response)
    # …then unwrap a reply the model handed back entirely inside quotes.
    response = _strip_quote_wrap(response)
    # Deliverables only: drop any section header the combine left empty (bare 'CONTEXT:'
    # with no body) so a hollow skeleton can never render. Full sections pass untouched.
    if ctx.get("deliverable"):
        response = _drop_empty_sections(response)

    # Persist the turn + mine durable facts — same path /chat uses, off the loop.
    message = ctx.get("message") or ctx.get("query", "")
    session_id = ctx.get("session_id", "default")
    # route_tier carries the guard: private/secret → 'secret' → extractor vaults the
    # turn locally instead of sending the (private) response to the cloud extractor.
    await asyncio.to_thread(save_message, session_id, "user", message)
    await asyncio.to_thread(save_message, session_id, "assistant", response)
    await asyncio.to_thread(extract_and_store, message, response, route_tier)

    reply = {
        "response": response,
        "mood": mood,
        "provider_used": result["book_used"],
        "tokens": result["tokens"],
    }

    # Record the FULL reply for a guarded (write) action, so a LATER duplicate of the
    # exact same request — still inside the idempotency TTL — gets this exact text
    # back instead of a fresh, possibly differently-worded paraphrase of the same
    # fact. Only ever set on a genuinely FRESH execution (_idem_replay is False),
    # never on a replay itself — replaying a replay would just re-store what's
    # already there.
    if action_result is not None and action_result.get("_idem_key") and not action_result.get("_idem_replay"):
        record_reply(action_result["_idem_key"], reply)

    return reply
