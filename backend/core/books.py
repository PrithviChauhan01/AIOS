import os
import re
import time
import asyncio
from config import Config

# ── Capability map — exact values from LLD §4 ──
CAPABILITY_MAP = {
    "groq":     {"speed": "high", "reasoning": "good",   "context": "med",   "cost": "free", "tier": "cloud"},
    "cerebras": {"speed": "high", "reasoning": "good",   "context": "med",   "cost": "free", "tier": "cloud"},
    "mistral":  {"speed": "med",  "reasoning": "good",   "context": "huge",  "cost": "free", "tier": "cloud"},
    "gpt4o":    {"speed": "med",  "reasoning": "best",   "context": "large", "cost": "ltd",  "tier": "cloud"},
    "gemini":   {"speed": "med",  "reasoning": "strong", "context": "huge",  "cost": "ltd",  "tier": "cloud"},
    # NVIDIA Nemotron 3 (OpenAI-compatible via NVIDIA NIM) — high-reasoning, free, cloud.
    "nemotron_super": {"speed": "med", "reasoning": "best",     "context": "huge", "cost": "free", "tier": "cloud"},
    "nemotron_ultra": {"speed": "low", "reasoning": "frontier", "context": "huge", "cost": "free", "tier": "cloud"},
    "ollama":   {"speed": "low",  "reasoning": "weak",   "context": "small", "cost": "free", "tier": "local"},
}

# ── Ordinal scales for fit comparison ──
_SPEED     = {"low": 0, "med": 1, "high": 2}
_REASONING = {"weak": 0, "good": 1, "strong": 2, "best": 3, "frontier": 4}
_CONTEXT   = {"small": 0, "med": 1, "large": 2, "huge": 3}
_COST      = {"free": 0, "ltd": 1}  # lower is cheaper, cheaper preferred

_MODELS = {
    "groq":     "llama-3.3-70b-versatile",
    "groq_fast":"llama-3.1-8b-instant",
    "cerebras":"gpt-oss-120b",
    "mistral":  "mistral-large-latest",
    "gpt4o":    "gpt-4o",
    "gemini":   "gemini-1.5-pro",
    "nemotron_super": "nvidia/nemotron-3-super-120b-a12b",
    "nemotron_ultra": "nvidia/nemotron-3-ultra-550b-a55b",
    "ollama":   "llama3.2",
}

def model_for(book: str) -> str:
    """The concrete model id behind a book — for logging fast vs full."""
    return _MODELS.get(book, book)

_MAX_TOKENS = 1024

# ── Rate-limit cooldown state ──
_COOLDOWN = {}            # book -> unix ts until which it is benched
_COOLDOWN_SECONDS = 60

# Nemotron is the DEFAULT reasoning book; Groq is its quota-fallback. A quota/429 from
# EITHER Nemotron benches BOTH (super + ultra) for a longer window so we don't hammer a
# rate-limited free tier turn after turn — selection auto-demotes to Groq until it lapses.
_NEMOTRON_BOOKS = ("nemotron_super", "nemotron_ultra")
_NEMOTRON_COOLDOWN_SECONDS = 5 * 60
_nemotron_cooldown_until = 0.0   # unix ts; while > now, Nemotron is skipped in selection

def _nemotron_cooling() -> bool:
    return _nemotron_cooldown_until > time.time()

def _available(book: str) -> bool:
    if book in _NEMOTRON_BOOKS and _nemotron_cooling():
        return False
    return _COOLDOWN.get(book, 0) < time.time()

def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return ("429" in s or "rate limit" in s or "rate_limit" in s
            or "too many requests" in s or "quota" in s)

# ── Selection ──
def _meets(book: str, spec: dict) -> bool:
    caps = CAPABILITY_MAP[book]
    for field, scale in (("reasoning", _REASONING), ("context", _CONTEXT), ("speed", _SPEED)):
        if field in spec:
            if scale[caps[field]] < scale[spec[field]]:
                return False
    if "cost" in spec:  # cost is a ceiling, not a floor — cheaper or equal passes
        if _COST[caps["cost"]] > _COST[spec["cost"]]:
            return False
    return True

def _fit_key(book: str, spec: dict):
    caps = CAPABILITY_MAP[book]
    # tightest fit first: minimise over-provisioning on reasoning/context, then prefer
    # cheaper, then faster (speed is the final tiebreak per spec — fit, then speed).
    overprovision = 0
    for field, scale in (("reasoning", _REASONING), ("context", _CONTEXT)):
        if field in spec:
            overprovision += max(0, scale[caps[field]] - scale[spec[field]])
    return (overprovision, _COST[caps["cost"]], -_SPEED[caps["speed"]])

# ── Hard-reasoning preference ──
# Some turns want raw reasoning horsepower regardless of how modest the capability
# spec is. loop_worthy (the hardest reasoning) prefers the 550B Nemotron Ultra; a
# merely complex turn prefers the 120B Nemotron Super. Either way the tail is the
# proven free chain: groq 70b → cerebras. Books that are benched/unavailable are
# dropped here and the caller simply falls through to the next one.
def _hard_reasoning_order(loop_worthy, complexity) -> list:
    if loop_worthy:
        return ["nemotron_ultra", "nemotron_super", "groq", "cerebras"]
    if complexity == "complex":
        return ["nemotron_super", "groq", "cerebras"]
    return []

def select_book(capability_spec: dict, sensitivity_tier: str,
                complexity: str = None, loop_worthy: bool = None) -> list:
    """Return books ordered best→worst. Caller calls call_book down the list,
    falling back to the next candidate on failure.

    complexity / loop_worthy steer the hard-reasoning preference (Nemotron). They may
    be passed explicitly or ride inside capability_spec — explicit args win."""
    # secret → ollama only, hard filter, no exception (cooldown does not apply: nowhere else to go)
    if sensitivity_tier == "secret":
        return ["ollama"]

    if complexity is None:
        complexity = capability_spec.get("complexity")
    if loop_worthy is None:
        loop_worthy = capability_spec.get("loop_worthy")

    pool = [b for b in CAPABILITY_MAP if _available(b)]
    if not pool:  # everyone benched — relax and let cooldowns sort themselves out
        pool = list(CAPABILITY_MAP)

    candidates = [b for b in pool if _meets(b, capability_spec)]
    if candidates:
        ordered = sorted(candidates, key=lambda b: _fit_key(b, capability_spec))
    else:
        # nothing fully fits — best-effort: strongest available first
        print(f"[books] no book meets spec {capability_spec}; falling back to strongest available")
        ordered = sorted(pool, key=lambda b: (
            -_REASONING[CAPABILITY_MAP[b]["reasoning"]],
            -_CONTEXT[CAPABILITY_MAP[b]["context"]],
            -_SPEED[CAPABILITY_MAP[b]["speed"]],
        ))

    # Promote the hard-reasoning chain to the front when the turn calls for it. Only
    # books that are actually available (not benched) lead; the rest of the fit-ordered
    # list trails as further fallback.
    front = [b for b in _hard_reasoning_order(loop_worthy, complexity) if b in pool]
    if front:
        ordered = front + [b for b in ordered if b not in front]
    return ordered

# ── Reasoning tiers — teachers declare one; selection honors it ──
# A teacher names the reasoning tier it wants and we map that to an ordered book pool
# with the proven free chain (groq 70b → cerebras) as the tail. This REPLACES the old
# dependence on triage's complexity guess to fire Nemotron: a 'strong' teacher gets
# Nemotron Super on the FIRST pass regardless of how the local 3B triage rated the turn.
# Availability/cooldown aware — benched books are dropped and the caller falls through.
# Nemotron is the DEFAULT for real reasoning (Super leads, Ultra is the reasoning
# fallback); Groq 70b → Cerebras is the quota-fallback tail once Nemotron is exhausted
# or benched. 'fast' stays Groq-only — formats/light reasoning don't need Nemotron.
_TIER_BOOKS = {
    "fast":     ["groq", "cerebras"],                                          # formats/light reasoning
    "strong":   ["nemotron_super", "nemotron_ultra", "groq", "cerebras"],      # real reasoning
    "frontier": ["nemotron_super", "nemotron_ultra", "groq", "cerebras"],      # hardest
}
_TIER_ORDER = ["fast", "strong", "frontier"]
_DEFAULT_TIER = "fast"


def upgrade_tier(tier: str, complexity: str = None, loop_worthy: bool = None) -> str:
    """Triage complexity is an UPGRADE-ONLY signal. A 'complex' or loop_worthy turn
    bumps the teacher's declared tier ONE level up (fast→strong→frontier). It can
    never pull a tier BELOW the teacher's declared floor — we only ever raise, so an
    under-rating triage can't strand a reasoning task on Groq."""
    if tier not in _TIER_ORDER:
        tier = _DEFAULT_TIER
    if loop_worthy or complexity == "complex":
        return _TIER_ORDER[min(_TIER_ORDER.index(tier) + 1, len(_TIER_ORDER) - 1)]
    return tier


def select_book_for_tier(tier: str, sensitivity_tier: str,
                         complexity: str = None, loop_worthy: bool = None) -> list:
    """Map a teacher's reasoning tier → ordered book list (best→worst), after applying
    the upgrade signal and the same availability/cooldown filter select_book uses.
    Privacy is absolute and overrides the tier entirely: secret → ollama only."""
    # secret → ollama only, hard filter (privacy beats tier — identical to select_book)
    if sensitivity_tier == "secret":
        return ["ollama"]
    eff = upgrade_tier(tier, complexity, loop_worthy)
    order = _TIER_BOOKS.get(eff, _TIER_BOOKS[_DEFAULT_TIER])
    # Nemotron on cooldown (recent quota/429) → demote to Groq for this turn. _available
    # already drops the benched Nemotron books; this just surfaces it in the log.
    if _nemotron_cooling() and any(b in _NEMOTRON_BOOKS for b in order):
        print("[books] nemotron cooling → groq")
    avail = [b for b in order if _available(b)]
    return avail if avail else order  # all benched → keep order; cooldowns lapse on retry


# ── Ensemble book pairing — two DIFFERENT books for a compulsory teacher ensemble ──
# A book's "family": both Nemotrons share one endpoint/quota, so they count as one for
# ensemble diversity — pairing them would be two calls to the same provider (and both
# die together on a 429). Collapsing them lets the second slot fall to a genuinely
# independent book (Groq), which is the whole point of running two passes.
def _book_family(book: str) -> str:
    if book in _NEMOTRON_BOOKS:
        return "nemotron"
    if book in ("groq", "groq_fast"):
        return "groq"
    return book


def select_ensemble_books(tier: str, sensitivity_tier: str,
                          complexity: str = None, loop_worthy: bool = None) -> list:
    """Two DISTINCT books (different families) for a compulsory teacher ensemble, best→worst.
    The teacher's tier book leads; the second slot is the next-best book from a DIFFERENT
    family (e.g. nemotron_super + groq — never nemotron twice, never the same book twice),
    so the two research passes are genuinely independent. Returns fewer than two only when
    a single family is all that's available (everything else benched) — the caller then
    degrades to a single pass. secret → ['ollama'] (one local book; caller must not ensemble)."""
    ordered = select_book_for_tier(tier, sensitivity_tier,
                                   complexity=complexity, loop_worthy=loop_worthy)
    picked, seen = [], set()
    for b in ordered:
        fam = _book_family(b)
        if fam in seen:
            continue
        picked.append(b)
        seen.add(fam)
        if len(picked) == 2:
            break
    return picked


# ── Fast lane ──
# Trivial / short-circuit small-talk needs her voice, not reasoning horsepower.
# This routes it to Groq's small 8B model (sub-second) with the normal good-
# reasoning chain as fallback if the fast book is benched or fails. 'groq_fast' is
# deliberately NOT in CAPABILITY_MAP, so the generic selector never reaches it —
# the 70B stays the default for teacher/deliverable work. Secret stays local-only.
def select_fast_book(sensitivity_tier: str) -> list:
    if sensitivity_tier == "secret":
        return ["ollama"]
    books = []
    if _available("groq_fast"):
        books.append("groq_fast")
    for b in select_book({"reasoning": "good"}, sensitivity_tier):
        if b not in books:
            books.append(b)
    return books

# ── Provider clients (lazy so the module imports without every SDK/key present) ──
_clients = {}

def _groq_client():
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(api_key=Config.GROQ_API_KEY)
    return _clients["groq"]

def _cerebras_client():
    if "cerebras" not in _clients:
        from cerebras.cloud.sdk import Cerebras
        _clients["cerebras"] = Cerebras(api_key=Config.CEREBRAS_API_KEY)
    return _clients["cerebras"]

def _mistral_client():
    if "mistral" not in _clients:
        from mistralai import Mistral
        _clients["mistral"] = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
    return _clients["mistral"]

def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    return _clients["openai"]

# NVIDIA NIM is OpenAI-compatible — same SDK, just a different base_url + key.
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

def _nvidia_client():
    if "nvidia" not in _clients:
        from openai import OpenAI
        _clients["nvidia"] = OpenAI(api_key=Config.NVIDIA_API_KEY, base_url=_NVIDIA_BASE_URL)
    return _clients["nvidia"]

# ── Per-book synchronous calls (run off-thread by call_book) ──
# max_tokens is per-call: cognition hard-caps length (trivial ~60, deliverable
# ~900). Defaults to _MAX_TOKENS for callers that don't care (e.g. teachers).
def _call_groq(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _groq_client().chat.completions.create(
        model=_MODELS["groq"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

def _call_groq_fast(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _groq_client().chat.completions.create(
        model=_MODELS["groq_fast"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

def _call_cerebras(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _cerebras_client().chat.completions.create(
        model=_MODELS["cerebras"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

def _call_mistral(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _mistral_client().chat.complete(
        model=_MODELS["mistral"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

def _call_gpt4o(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _openai_client().chat.completions.create(
        model=_MODELS["gpt4o"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

# Nemotron Super — standard chat completion, no reasoning-budget params.
def _call_nemotron_super(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _nvidia_client().chat.completions.create(
        model=_MODELS["nemotron_super"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens)
    return r.choices[0].message.content, r.usage.total_tokens

# Nemotron Ultra — a REASONING model. It thinks in message.reasoning_content and
# answers in message.content. We:
#   * turn thinking on + give it a real budget (extra_body),
#   * pass the caller's cap as the FINAL-ANSWER budget and add the reasoning budget ON
#     TOP — the API's max_tokens covers reasoning + content together, so if we only sent
#     max(cap, budget) a long think would eat the window and the visible answer would cut
#     off mid-sentence ("Let me…"). Adding them keeps the caller's cap purely for content.
#   * return ONLY message.content — the reasoning trace is captured for logs and
#     DROPPED, never handed to cognition/the user.
_ULTRA_REASONING_BUDGET = 16384
_ULTRA_MIN_ANSWER_TOKENS = 512   # floor so a tiny caller cap can't starve the visible answer
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>\s*")

def _call_nemotron_ultra(prompt: str, max_tokens: int = _MAX_TOKENS):
    # max_tokens is the FINAL-ANSWER budget; reasoning gets its own budget on top.
    answer_cap = max(max_tokens, _ULTRA_MIN_ANSWER_TOKENS)
    api_max = _ULTRA_REASONING_BUDGET + answer_cap
    r = _nvidia_client().chat.completions.create(
        model=_MODELS["nemotron_ultra"], messages=[{"role": "user", "content": prompt}],
        max_tokens=api_max,
        extra_body={"chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": _ULTRA_REASONING_BUDGET})
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        # Internal only — log its size, never its content, and never return it.
        print(f"[books] nemotron_ultra reasoning trace dropped ({len(reasoning)} chars, internal)")
    answer = msg.content or ""
    # Defensive: if a <think>…</think> block ever leaks into content, strip it so the
    # reply can never carry the thinking trace.
    answer = _THINK_BLOCK_RE.sub("", answer).strip()
    return answer, r.usage.total_tokens

def _call_gemini(prompt: str, max_tokens: int = _MAX_TOKENS):
    import google.generativeai as genai
    genai.configure(api_key=Config.GOOGLE_AI_API_KEY)
    model = genai.GenerativeModel(_MODELS["gemini"])
    r = model.generate_content(
        prompt, generation_config={"max_output_tokens": max_tokens})
    tokens = getattr(getattr(r, "usage_metadata", None), "total_token_count", 0)
    return r.text, tokens

def _call_ollama(prompt: str, max_tokens: int = _MAX_TOKENS):
    import ollama
    # num_predict caps the OUTPUT; num_ctx is the whole window (prompt + output).
    # The default num_ctx (2048) is smaller than a deliverable's prompt+material,
    # so the model silently drops the tail and the reply ends mid-sentence. Give it
    # a window big enough to hold the full prompt AND max_tokens of new output.
    r = ollama.chat(
        model=_MODELS["ollama"], messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "num_ctx": 8192})
    tokens = r.get("prompt_eval_count", 0) + r.get("eval_count", 0)
    return r["message"]["content"], tokens

_DISPATCH = {
    "groq": _call_groq,
    "groq_fast": _call_groq_fast,
    "cerebras": _call_cerebras,
    "mistral": _call_mistral,
    "gpt4o": _call_gpt4o,
    "gemini": _call_gemini,
    "nemotron_super": _call_nemotron_super,
    "nemotron_ultra": _call_nemotron_ultra,
    "ollama": _call_ollama,
}

# ── Invocation ──
async def call_book(prompt: str, book: str, max_tokens: int = _MAX_TOKENS) -> dict:
    fn = _DISPATCH.get(book)
    if fn is None:
        raise ValueError(f"unknown book: {book}")
    try:
        raw_text, tokens = await asyncio.to_thread(fn, prompt, max_tokens)
        return {"raw_text": raw_text, "book_used": book, "tokens": tokens}
    except Exception as e:
        if _is_rate_limit(e):
            _COOLDOWN[book] = time.time() + _COOLDOWN_SECONDS
            if book in _NEMOTRON_BOOKS:
                # Quota/429 from Nemotron → bench BOTH Nemotron books for the longer
                # window and auto-demote to Groq (the caller falls to the next candidate).
                global _nemotron_cooldown_until
                _nemotron_cooldown_until = time.time() + _NEMOTRON_COOLDOWN_SECONDS
                print(f"[books] nemotron quota → groq (cooldown {_NEMOTRON_COOLDOWN_SECONDS // 60}m)")
            else:
                print(f"[books] {book} rate-limited — benched {_COOLDOWN_SECONDS}s")
        else:
            print(f"[books] {book} failed: {e}")
        raise  # let caller fall back to the next candidate

# ── Ensemble ──
async def run_ensemble(prompt: str, capability_spec: dict, sensitivity_tier: str) -> list:
    """Run the top 2 candidates in parallel. Returns a list of raw results —
    combining/judging is cognition's job, not ours."""
    if not capability_spec.get("loop_worthy", True):
        # not loop-worthy: degrade to a single best-effort call
        for book in select_book(capability_spec, sensitivity_tier):
            try:
                return [await call_book(prompt, book)]
            except Exception:
                continue
        return []

    candidates = select_book(capability_spec, sensitivity_tier)[:2]
    results = await asyncio.gather(
        *(call_book(prompt, b) for b in candidates), return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]
