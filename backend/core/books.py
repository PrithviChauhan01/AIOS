import os
import re
import time
import asyncio
from config import Config
# Every call below is bounded and retried by the shared policy — see core/net.py.
# max_retries=0 on each client makes tenacity the ONLY retry authority: the SDKs
# retry twice by default, which would silently multiply attempts (and hide the 429s
# the cooldown logic below depends on seeing).
from core.net import (CLOUD_TIMEOUT, LONG_TIMEOUT, timeout_for, with_retry,
                      ollama_client)
# Circuit breaker. Every _call_* below reports its outcome through net.with_retry's
# choke point; selection here asks whether a provider is worth offering at all.
from core import provider_health as health
from core.provider_health import provider_of

# ── Capability map — exact values from LLD §4 ──
CAPABILITY_MAP = {
    # Kimi K2.5 on Amazon Bedrock — the MAIN book. Paid (instance-role billed), hence 'ltd'.
    "bedrock_kimi": {"speed": "med", "reasoning": "frontier", "context": "huge", "cost": "ltd", "tier": "cloud"},
    "groq":     {"speed": "high", "reasoning": "good",   "context": "med",   "cost": "free", "tier": "cloud"},
    "cerebras": {"speed": "high", "reasoning": "good",   "context": "med",   "cost": "free", "tier": "cloud"},
    "mistral":  {"speed": "med",  "reasoning": "good",   "context": "huge",  "cost": "free", "tier": "cloud"},
    "gpt4o":    {"speed": "med",  "reasoning": "best",   "context": "large", "cost": "ltd",  "tier": "cloud"},
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
    "bedrock_kimi": "moonshotai.kimi-k2.5",
    # Groq retired the llama-3.x models on 2026-08-16 — gpt-oss replaces both tiers.
    "groq":     "openai/gpt-oss-120b",
    "groq_fast":"openai/gpt-oss-20b",
    "groq_qwen":"qwen/qwen3.8-27b",   # same Groq key, separate per-model quota
    "cerebras":"gpt-oss-120b",
    "mistral":  "mistral-large-latest",
    "gpt4o":    "gpt-4o",
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
    """SOFT bench — the rate-limit cooldown. Relaxable: when every book is cooling,
    selection ignores it rather than answering nothing (see _pool)."""
    if book in _NEMOTRON_BOOKS and _nemotron_cooling():
        return False
    return _COOLDOWN.get(book, 0) < time.time()


# ── Circuit gate ──
# Two benches, deliberately different in kind:
#
#   _available  — the rate-limit cooldown. A guess that a provider WILL refuse us
#                 shortly. Relaxed when everything is benched, because being wrong
#                 costs one 429 and being right saves a turn.
#   _closed     — the circuit breaker (core/provider_health.py). A MEASUREMENT that
#                 a provider is failing right now — a bad key, an exhausted quota,
#                 a dead endpoint. NEVER relaxed: relaxing it just spends the full
#                 timeout budget to rediscover what we already know.
#
# So the fallback ladder is: usable → circuit-closed-only (cooldowns relaxed) →
# the local book → nothing (the caller's loop finds no book and cognition returns
# its existing degradation line). A turn never escalates past the local book.
def _closed(book: str) -> bool:
    # An unconfigured book (no key / no SDK) is as dead as an open circuit — and,
    # like one, never relaxed. _CONFIGURED is computed once at import, below.
    return book in _CONFIGURED and health.allow_book(book)


def _usable(book: str) -> bool:
    return _available(book) and _closed(book)


def _local_fallback() -> list:
    """Every cloud circuit is open → the local book, which is a TIGHTENING (local,
    private) and never an escape. Empty when ollama's circuit is open too — there
    is genuinely nowhere left to go, and the caller degrades gracefully."""
    if _closed("ollama"):
        print("[books] all cloud circuits open → falling back to ollama (local)")
        return ["ollama"]
    print("[books] all circuits open, ollama included → no book available")
    return []


def _pool(books) -> list:
    """Apply both benches to a candidate list, relaxing ONLY the soft one."""
    usable = [b for b in books if _usable(b)]
    if usable:
        return usable
    # Everyone cooling — relax the cooldown (existing behaviour), never the circuit.
    return [b for b in books if _closed(b)]

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

# ── Main chain ──
# The fixed order every selector leads with: Kimi K2.5 on Bedrock → Groq gpt-oss-120b
# → Cerebras gpt-oss-120b → ollama (local, always LAST). Benched/disabled books are
# dropped and the caller simply falls through to the next one.
_MAIN_CHAIN = ["bedrock_kimi", "groq", "cerebras"]
_LOCAL_BOOK = "ollama"


def _with_local_tail(books) -> list:
    """ollama goes last — after every cloud book, never ahead of one — and only
    while its circuit is closed."""
    books = [b for b in books if b != _LOCAL_BOOK]
    if _closed(_LOCAL_BOOK):
        books.append(_LOCAL_BOOK)
    return books


def _why_unusable(book: str):
    """Why selection dropped a book, or None when it didn't. An unconfigured book
    returns None too: that was logged once at startup and must not repeat per turn."""
    if book not in _CONFIGURED:
        return None
    if not health.allow_book(book):
        return f"{provider_of(book)} circuit {health.circuit_of(provider_of(book))}"
    if not _available(book):
        return "rate-limit cooldown"
    return None


def _log_skipped(books, picked) -> None:
    for b in books:
        if b not in picked:
            why = _why_unusable(b)
            if why:
                print(f"[books] skipping {b} — {why}")

def select_book(capability_spec: dict, sensitivity_tier: str,
                complexity: str = None, loop_worthy: bool = None) -> list:
    """Return books ordered best→worst. Caller calls call_book down the list,
    falling back to the next candidate on failure.

    The main chain (bedrock_kimi → groq → cerebras) always leads, other fitting
    books trail, ollama is last. complexity / loop_worthy are still accepted for
    callers but no longer change the order."""
    # secret → ollama only, hard filter, no exception. Neither bench applies: there is
    # nowhere else to go BY DESIGN. If ollama's circuit is open the call is refused at
    # the choke point and the turn FAILS to the degradation line — a secret turn never
    # escapes to cloud because the local provider is unhealthy. Privacy sits above the
    # circuit, always.
    if sensitivity_tier == "secret":
        return ["ollama"]

    # Cooldowns relax when everything is benched; open circuits never do. ollama is
    # in CAPABILITY_MAP, so an all-cloud-open pool naturally leaves the local book as
    # the only candidate — the best-effort branch below then picks it.
    pool = _pool(CAPABILITY_MAP)
    if not pool:
        return _local_fallback()

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

    # The main chain leads whatever the spec. Only books that are actually available
    # (not benched) lead; the rest of the fit-ordered list trails, ollama last.
    front = [b for b in _MAIN_CHAIN if b in pool]
    _log_skipped(_MAIN_CHAIN, front)
    return _with_local_tail(front + [b for b in ordered if b not in front])

# ── Reasoning tiers — teachers declare one; selection honors it ──
# A teacher names the reasoning tier it wants and we map that to an ordered book pool.
# Every tier currently runs the main chain (Kimi K2.5 on Bedrock → groq → cerebras,
# ollama appended last); the tiers are kept so teachers can diverge again later.
# Availability/cooldown aware — benched books are dropped and the caller falls through.
_TIER_BOOKS = {
    "fast":     list(_MAIN_CHAIN),
    "strong":   list(_MAIN_CHAIN),
    "frontier": list(_MAIN_CHAIN),
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
    # secret → ollama only, hard filter (privacy beats tier AND circuit — identical to
    # select_book; an open local circuit fails the turn, it never widens it).
    if sensitivity_tier == "secret":
        return ["ollama"]
    eff = upgrade_tier(tier, complexity, loop_worthy)
    order = _TIER_BOOKS.get(eff, _TIER_BOOKS[_DEFAULT_TIER])
    avail = _pool(order)
    _log_skipped(order, avail)
    # A tier list is cloud-only, so an empty one means every cloud circuit in this
    # tier is open — that is exactly the fall-to-local case, not a reason to hand
    # back dead candidates. Otherwise ollama trails the cloud books as the last resort.
    return _with_local_tail(avail) if avail else _local_fallback()


# ── Ensemble book pairing — two DIFFERENT books when the brain decides ensemble=true ──
# A book's "family": both Nemotrons share one endpoint/quota, so they count as one for
# ensemble diversity — pairing them would be two calls to the same provider (and both
# die together on a 429). Collapsing them lets the second slot fall to a genuinely
# independent book (Groq), which is the whole point of running two passes.
# provider_of draws exactly this partition ({groq, groq_fast} → one, {nemotron_super,
# nemotron_ultra} → one, everything else its own) and is the same notion for the same
# reason — one key, one endpoint, one quota, one failure. Sharing it keeps ensemble
# diversity and circuit sharing from ever drifting apart.
def _book_family(book: str) -> str:
    return provider_of(book)


def select_ensemble_books(tier: str, sensitivity_tier: str,
                          complexity: str = None, loop_worthy: bool = None) -> list:
    """Two DISTINCT books (different families), best→worst — the candidate pool for a
    teacher ensemble. This is NOT compulsory: the caller (agents.base.Teacher.run) only
    reaches this when the brain's per-task plan set ensemble=true for a task it
    judged genuinely benefits from two passes (complex reasoning, a high-stakes
    deliverable); a straightforward task gets a single book instead — the quota guard.
    The teacher's tier book leads; the second slot is the next-best book from a DIFFERENT
    family (e.g. nemotron_super + groq — never nemotron twice, never the same book twice),
    so the two research passes are genuinely independent. Returns fewer than two only when
    a single family is all that's available (everything else benched) — the caller then
    degrades to a single pass. secret → ['ollama'] (one local book; caller must not ensemble)."""
    ordered = select_book_for_tier(tier, sensitivity_tier,
                                   complexity=complexity, loop_worthy=loop_worthy)
    # The local tail is a last resort, not an ensemble partner — unless it is all there is.
    ordered = [b for b in ordered if b != _LOCAL_BOOK] or ordered
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
    if _usable("groq_fast"):
        books.append("groq_fast")
    for b in select_book({"reasoning": "good"}, sensitivity_tier):
        if b not in books:
            books.append(b)
    return books

# ── Provider clients (lazy so the module imports without every SDK/key present) ──
_clients = {}

def groq_client():
    # Public: core.brain runs the routing brain (Groq 70b) on this same shared client, so
    # there is exactly one Groq connection pool in the process.
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(api_key=Config.GROQ_API_KEY,
                                timeout=CLOUD_TIMEOUT, max_retries=0)
    return _clients["groq"]

def _cerebras_client():
    if "cerebras" not in _clients:
        from cerebras.cloud.sdk import Cerebras
        _clients["cerebras"] = Cerebras(api_key=Config.CEREBRAS_API_KEY,
                                        timeout=CLOUD_TIMEOUT, max_retries=0)
    return _clients["cerebras"]

def _mistral_client():
    if "mistral" not in _clients:
        from mistralai import Mistral
        _clients["mistral"] = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""),
                                      timeout_ms=int(LONG_TIMEOUT.read * 1000))
    return _clients["mistral"]

def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""),
                                    timeout=CLOUD_TIMEOUT, max_retries=0)
    return _clients["openai"]

# NVIDIA NIM is OpenAI-compatible — same SDK, just a different base_url + key.
# Public: core.brain runs the routing brain (Qwen) on the same endpoint and shares this
# lazily-built client, so there is exactly one NIM connection pool in the process.
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

def nvidia_client():
    if "nvidia" not in _clients:
        from openai import OpenAI
        # Nemotron is the slow, long-answer endpoint — it gets the long read budget as
        # its client default; per-call overrides still apply below.
        _clients["nvidia"] = OpenAI(api_key=Config.NVIDIA_API_KEY, base_url=_NVIDIA_BASE_URL,
                                    timeout=LONG_TIMEOUT, max_retries=0)
    return _clients["nvidia"]

# Amazon Bedrock — credentials come from boto3's default chain (the EC2 instance role
# on the server; nothing in .env). The long read budget is the client default because
# Kimi K2.5 reasons before it answers. botocore's own retries are off: with_retry is
# the only retry authority, exactly as for the SDK clients above.
_BEDROCK_REGION = "ap-south-1"

def bedrock_client():
    if "bedrock" not in _clients:
        import boto3
        from botocore.config import Config as BotoConfig
        _clients["bedrock"] = boto3.client(
            "bedrock-runtime", region_name=_BEDROCK_REGION,
            config=BotoConfig(connect_timeout=LONG_TIMEOUT.connect,
                              read_timeout=LONG_TIMEOUT.read,
                              retries={"total_max_attempts": 1, "mode": "standard"}))
    return _clients["bedrock"]


def _bedrock_converse(prompt: str, max_tokens: int) -> dict:
    return bedrock_client().converse(
        modelId=_MODELS["bedrock_kimi"],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens})

# ── Deterministic reachability probe (/health/deep) ──
# Lists the provider's MODELS: a real HTTP round trip on the real key, no completion,
# no tokens, no model asked to judge anything. Two deliberate properties:
#
#   READ-ONLY w.r.t. routing — a ping never reports to provider_health, so checking
#   health can never open or close a circuit. A diagnostic that changes what it
#   measures is worthless, and an operator refreshing a dashboard must not be able to
#   bench a provider.
#
#   OPT-IN at the endpoint — these are real API calls against the same free tiers the
#   assistant runs on. /health/deep answers from recorded call outcomes by default and
#   only pings when explicitly asked to.
_PING_CLIENTS = {
    # bedrock-runtime has no model list — the ping is a tiny converse. Opt-in at
    # /health/deep, so it only costs when asked.
    "bedrock": lambda: _bedrock_converse("ping", _BEDROCK_PROBE_TOKENS),
    "groq": lambda: groq_client().models.list(),
    "cerebras": lambda: _cerebras_client().models.list(),
    "nvidia": lambda: nvidia_client().models.list(),
    "openai": lambda: _openai_client().models.list(),
    "mistral": lambda: _mistral_client().models.list(),
    "ollama": lambda: ollama_client().list(),
}

# Which key each provider needs — reported so a "down" provider that simply has no
# key configured reads as unconfigured rather than broken.
_PROVIDER_KEY = {
    # No key: the instance role. "Configured" means the startup probe passed.
    "bedrock": lambda: "iam-role" if "bedrock_kimi" in _CONFIGURED else "",
    "groq": lambda: Config.GROQ_API_KEY,
    "cerebras": lambda: Config.CEREBRAS_API_KEY,
    "nvidia": lambda: Config.NVIDIA_API_KEY,
    "openai": lambda: os.getenv("OPENAI_API_KEY", ""),
    "mistral": lambda: os.getenv("MISTRAL_API_KEY", ""),
    "ollama": lambda: "local",  # no key — the daemon is either up or it isn't
}


def key_configured(provider: str) -> bool:
    getter = _PROVIDER_KEY.get(provider)
    return bool(getter and (getter() or "").strip())


# ── Startup configuration gate ──
# A book whose key or SDK is missing would fail on every single call (gpt4o and
# mistral on most installs). Decide once at import, log once, and keep it out of
# selection entirely instead of spending a turn rediscovering it.
_PROVIDER_PACKAGE = {
    "bedrock": "boto3",
    "groq": "groq",
    "cerebras": "cerebras.cloud.sdk",
    "nvidia": "openai",
    "openai": "openai",
    "mistral": "mistralai",
    "ollama": "ollama",
}


def _package_present(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # parent package missing (cerebras.cloud.sdk)
        return False


def _configured_books() -> set:
    ok = set()
    for book in _MODELS:
        provider = provider_of(book)
        if provider == "bedrock":
            continue  # no key to check — probe_bedrock() decides it at startup
        if not key_configured(provider):
            print(f"[books] {book} disabled — no API key for {provider}")
        elif not _package_present(_PROVIDER_PACKAGE.get(provider, provider)):
            print(f"[books] {book} disabled — package for {provider} not installed")
        else:
            ok.add(book)
    return ok


_CONFIGURED = _configured_books()


# ── Bedrock startup probe ──
# The laptop has no AWS credentials; the server has an instance role. Probe ONCE at
# startup (main.py's lifespan) with a tiny real converse — that proves credentials,
# region and model access in one go. Pass → bedrock_kimi joins _CONFIGURED. Fail →
# it stays out for the life of the process, logged once, and the chain starts at groq.
# Like ping_provider, the probe never reports to provider_health.
_BEDROCK_PROBE_TOKENS = 16
_bedrock_probe_result = None


def probe_bedrock() -> bool:
    """BLOCKING, idempotent. Returns whether bedrock_kimi is enabled."""
    global _bedrock_probe_result
    if _bedrock_probe_result is not None:
        return _bedrock_probe_result
    book, reason = "bedrock_kimi", None
    t0 = time.monotonic()
    if not _package_present("boto3"):
        reason = "boto3 not installed"
    else:
        try:
            import boto3
            if boto3.Session().get_credentials() is None:
                reason = "no AWS credentials in the default chain"
            else:
                _bedrock_converse("ping", _BEDROCK_PROBE_TOKENS)
        except Exception as e:
            reason = f"probe failed — {type(e).__name__}: {e}"[:240]
    _bedrock_probe_result = reason is None
    if reason:
        _CONFIGURED.discard(book)
        print(f"[books] {book} disabled — {reason}")
    else:
        _CONFIGURED.add(book)
        print(f"[books] {book} enabled — {_MODELS[book]} in {_BEDROCK_REGION} "
              f"answered the probe ({int((time.monotonic() - t0) * 1000)}ms)")
    return _bedrock_probe_result


def ping_provider(provider: str) -> dict:
    """BLOCKING reachability check for one provider (run it off the event loop).
    Returns {ok, ms, error} — measured, never inferred."""
    fn = _PING_CLIENTS.get(provider)
    if fn is None:
        return {"ok": False, "ms": None, "error": "no probe for this provider"}
    if not key_configured(provider):
        return {"ok": False, "ms": None, "error": "no key configured"}
    t0 = time.monotonic()
    try:
        fn()
        return {"ok": True, "ms": int((time.monotonic() - t0) * 1000), "error": None}
    except Exception as e:
        return {"ok": False, "ms": int((time.monotonic() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}"[:200]}


# ── Usage reporting ──
# Every provider already returns a usage object on the SAME response; this just reads
# it. Nothing here estimates or tokenizes — a provider that reports no prompt/
# completion split yields None for that half and the trace column stays NULL.
def _usage(u) -> tuple:
    """(total, prompt, completion) exactly as the provider reported them."""
    return (getattr(u, "total_tokens", None) or 0,
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None))


# ── Per-book synchronous calls (run off-thread by call_book) ──
# max_tokens is per-call: cognition hard-caps length (trivial ~60, deliverable
# ~900). Defaults to _MAX_TOKENS for callers that don't care (e.g. teachers).
#
# Each is wrapped by @with_retry (2 attempts, backoff+jitter, retryable failures
# only) and passes an explicit per-call timeout sized to its output — a deliverable
# asking for 4096 tokens legitimately needs longer than a 150-token confirmation.
# A retry re-runs THIS function against THIS book, so it can never cross a tier.
class BedrockError(Exception):
    """A botocore ClientError re-raised with the HTTP status as `status_code` — the
    shape net.py reads. ClientError keeps it in a dict (response['ResponseMetadata']),
    which would hide throttling (429) and AccessDenied (403) from retry and circuit."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# Kimi K2.5 reasons before it answers and the reasoning counts against maxTokens, so a
# tiny caller cap (the 150-token fast lane) could be spent thinking with no answer left.
# Floor the cap; reasoningContent blocks are dropped, only text blocks are returned.
_KIMI_MIN_TOKENS = 1024

@with_retry(provider="bedrock")
def _call_bedrock_kimi(prompt: str, max_tokens: int = _MAX_TOKENS):
    try:
        r = _bedrock_converse(prompt, max(max_tokens, _KIMI_MIN_TOKENS))
    except Exception as e:
        resp = getattr(e, "response", None)
        if not isinstance(resp, dict):
            raise  # transport failure (EndpointConnectionError, ReadTimeoutError …)
        err = resp.get("Error", {})
        raise BedrockError(f"{err.get('Code')}: {err.get('Message')}",
                           resp.get("ResponseMetadata", {}).get("HTTPStatusCode")) from e
    blocks = r["output"]["message"]["content"]
    answer = "".join(b["text"] for b in blocks if "text" in b)
    answer = _THINK_BLOCK_RE.sub("", answer).strip()
    u = r.get("usage", {})
    return answer, u.get("totalTokens") or 0, u.get("inputTokens"), u.get("outputTokens")

@with_retry(provider="groq")
def _call_groq(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = groq_client().chat.completions.create(
        model=_MODELS["groq"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout_for(max_tokens), reasoning_effort="low")
    return (r.choices[0].message.content, *_usage(r.usage))

# gpt-oss is a reasoning model and its thinking counts against max_tokens — "low"
# keeps a 150-token fast-lane cap from being spent before the answer starts.
@with_retry(provider="groq")
def _call_groq_fast(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = groq_client().chat.completions.create(
        model=_MODELS["groq_fast"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout_for(max_tokens), reasoning_effort="low")
    return (r.choices[0].message.content, *_usage(r.usage))

# Qwen on Groq — the alternative when gpt-oss-120b is rate-limited (Groq quotas are
# per model). Thinking off: it is a fallback voice, not a reasoning pass.
@with_retry(provider="groq")
def _call_groq_qwen(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = groq_client().chat.completions.create(
        model=_MODELS["groq_qwen"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout_for(max_tokens), reasoning_effort="none")
    return (r.choices[0].message.content, *_usage(r.usage))

@with_retry(provider="cerebras")
def _call_cerebras(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _cerebras_client().chat.completions.create(
        model=_MODELS["cerebras"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout_for(max_tokens))
    return (r.choices[0].message.content, *_usage(r.usage))

@with_retry(provider="mistral")
def _call_mistral(prompt: str, max_tokens: int = _MAX_TOKENS):
    # The Mistral SDK takes its timeout on the client (timeout_ms), not per call.
    r = _mistral_client().chat.complete(
        model=_MODELS["mistral"], messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return (r.choices[0].message.content, *_usage(r.usage))

@with_retry(provider="openai")
def _call_gpt4o(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = _openai_client().chat.completions.create(
        model=_MODELS["gpt4o"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=timeout_for(max_tokens))
    return (r.choices[0].message.content, *_usage(r.usage))

# Nemotron Super — thinks by DEFAULT, and the thinking counts against max_tokens: on a
# 150-token fast-lane cap it spent the whole budget reasoning and returned
# content=None (the public-mode empty bubble). Thinking off; Ultra is the reasoning book.
@with_retry(provider="nvidia")
def _call_nemotron_super(prompt: str, max_tokens: int = _MAX_TOKENS):
    r = nvidia_client().chat.completions.create(
        model=_MODELS["nemotron_super"], messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=LONG_TIMEOUT,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    answer = _THINK_BLOCK_RE.sub("", r.choices[0].message.content or "").strip()
    return (answer, *_usage(r.usage))

# Nemotron Ultra — a REASONING model. It thinks in message.reasoning_content and
# answers in message.content. We:
#   * turn thinking on (extra_body) and size max_tokens for a real reasoning budget,
#   * pass the caller's cap as the FINAL-ANSWER budget and add the reasoning budget ON
#     TOP — the API's max_tokens covers reasoning + content together, so if we only sent
#     max(cap, budget) a long think would eat the window and the visible answer would cut
#     off mid-sentence ("Let me…"). Adding them keeps the caller's cap purely for content.
#   * return ONLY message.content — the reasoning trace is captured for logs and
#     DROPPED, never handed to cognition/the user.
_ULTRA_REASONING_BUDGET = 16384
_ULTRA_MIN_ANSWER_TOKENS = 512   # floor so a tiny caller cap can't starve the visible answer
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>\s*")

@with_retry(provider="nvidia")
def _call_nemotron_ultra(prompt: str, max_tokens: int = _MAX_TOKENS):
    # max_tokens is the FINAL-ANSWER budget; reasoning gets its own budget on top.
    answer_cap = max(max_tokens, _ULTRA_MIN_ANSWER_TOKENS)
    api_max = _ULTRA_REASONING_BUDGET + answer_cap
    r = nvidia_client().chat.completions.create(
        model=_MODELS["nemotron_ultra"], messages=[{"role": "user", "content": prompt}],
        max_tokens=api_max,
        # Always the long budget: even a small answer cap sits behind a 16k reasoning pass.
        timeout=LONG_TIMEOUT,
        # No separate thinking-budget param: the endpoint doesn't take it. The budget
        # is enforced by api_max above (reasoning + answer share max_tokens).
        extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        # Internal only — log its size, never its content, and never return it.
        print(f"[books] nemotron_ultra reasoning trace dropped ({len(reasoning)} chars, internal)")
    answer = msg.content or ""
    # Defensive: if a <think>…</think> block ever leaks into content, strip it so the
    # reply can never carry the thinking trace.
    answer = _THINK_BLOCK_RE.sub("", answer).strip()
    return (answer, *_usage(r.usage))

@with_retry(provider="ollama")
def _call_ollama(prompt: str, max_tokens: int = _MAX_TOKENS):
    # num_predict caps the OUTPUT; num_ctx is the whole window (prompt + output).
    # The default num_ctx (2048) is smaller than a deliverable's prompt+material,
    # so the model silently drops the tail and the reply ends mid-sentence. Give it
    # a window big enough to hold the full prompt AND max_tokens of new output.
    # The shared client carries the LOCAL (120s) read budget — a local 3B is slow,
    # not hung, and this is the ONLY book a secret turn is allowed to use.
    r = ollama_client().chat(
        model=_MODELS["ollama"], messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "num_ctx": 8192})
    # ollama names the same two numbers differently — prompt_eval_count is the input
    # side, eval_count the generated side.
    prompt_tokens = r.get("prompt_eval_count")
    completion_tokens = r.get("eval_count")
    tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return r["message"]["content"], tokens, prompt_tokens, completion_tokens

_DISPATCH = {
    "bedrock_kimi": _call_bedrock_kimi,
    "groq": _call_groq,
    "groq_fast": _call_groq_fast,
    "groq_qwen": _call_groq_qwen,
    "cerebras": _call_cerebras,
    "mistral": _call_mistral,
    "gpt4o": _call_gpt4o,
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
        # tokens is the total (unchanged); tokens_in/tokens_out are the provider's own
        # prompt/completion split, or None where a provider doesn't report one.
        raw_text, tokens, tokens_in, tokens_out = await asyncio.to_thread(
            fn, prompt, max_tokens)
        if (raw_text or "").strip():
            print(f"[books] answered by {book} ({model_for(book)}, {tokens} tokens)")
        else:
            print(f"[books] {book} returned an empty reply → caller falls back")
        return {"raw_text": raw_text, "book_used": book, "tokens": tokens,
                "tokens_in": tokens_in, "tokens_out": tokens_out}
    except health.CircuitOpen as e:
        # Refused before the wire, not failed on it. Nothing to bench (the circuit
        # IS the bench) and nothing to record — this call never happened.
        print(f"[books] {book} skipped — {e} → falling back to next book")
        raise  # caller falls to the next candidate, exactly as for a failure
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
                print(f"[books] {book} rate-limited — benched {_COOLDOWN_SECONDS}s"
                      " → falling back to next book")
        else:
            print(f"[books] {book} failed ({type(e).__name__}: {e}) → falling back to next book")
        raise  # let caller fall back to the next candidate
