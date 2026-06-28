import os
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
    "ollama":   {"speed": "low",  "reasoning": "weak",   "context": "small", "cost": "free", "tier": "local"},
}

# ── Ordinal scales for fit comparison ──
_SPEED     = {"low": 0, "med": 1, "high": 2}
_REASONING = {"weak": 0, "good": 1, "strong": 2, "best": 3}
_CONTEXT   = {"small": 0, "med": 1, "large": 2, "huge": 3}
_COST      = {"free": 0, "ltd": 1}  # lower is cheaper, cheaper preferred

_MODELS = {
    "groq":     "llama-3.3-70b-versatile",
    "groq_fast":"llama-3.1-8b-instant",
    "cerebras":"gpt-oss-120b",
    "mistral":  "mistral-large-latest",
    "gpt4o":    "gpt-4o",
    "gemini":   "gemini-1.5-pro",
    "ollama":   "llama3.2",
}

def model_for(book: str) -> str:
    """The concrete model id behind a book — for logging fast vs full."""
    return _MODELS.get(book, book)

_MAX_TOKENS = 1024

# ── Rate-limit cooldown state ──
_COOLDOWN = {}            # book -> unix ts until which it is benched
_COOLDOWN_SECONDS = 60

def _available(book: str) -> bool:
    return _COOLDOWN.get(book, 0) < time.time()

def _is_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return "429" in s or "rate limit" in s or "rate_limit" in s or "too many requests" in s

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

def select_book(capability_spec: dict, sensitivity_tier: str) -> list:
    """Return books ordered best→worst. Caller calls call_book down the list,
    falling back to the next candidate on failure."""
    # secret → ollama only, hard filter, no exception (cooldown does not apply: nowhere else to go)
    if sensitivity_tier == "secret":
        return ["ollama"]

    pool = [b for b in CAPABILITY_MAP if _available(b)]
    if not pool:  # everyone benched — relax and let cooldowns sort themselves out
        pool = list(CAPABILITY_MAP)

    candidates = [b for b in pool if _meets(b, capability_spec)]
    if candidates:
        return sorted(candidates, key=lambda b: _fit_key(b, capability_spec))

    # nothing fully fits — best-effort: strongest available first
    print(f"[books] no book meets spec {capability_spec}; falling back to strongest available")
    return sorted(pool, key=lambda b: (
        -_REASONING[CAPABILITY_MAP[b]["reasoning"]],
        -_CONTEXT[CAPABILITY_MAP[b]["context"]],
        -_SPEED[CAPABILITY_MAP[b]["speed"]],
    ))

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
