"""Outbound provider policy — one place that decides how long a call may take and
when it may be tried again.

Before this, every provider call was unbounded: the SDKs default to no read
timeout on some paths, so a hung endpoint held the request open forever and the
turn never came back. A hang is worse than a failure — a failure falls through to
the next book in the chain within seconds, a hang just stops.

Two rules, applied identically to every call site (books, brain, extractor,
triage, action extraction, email composition, STT):

  TIMEOUT — connect 5s everywhere (a handshake slower than that is a dead
  endpoint, not a busy one). The read budget is what varies: 30s for a normal
  cloud answer, 60s for a deliverable/long-output call, 120s for the local
  ollama book, which is genuinely slow rather than hung.

  RETRY — at most 2 attempts, exponential backoff with jitter, and ONLY for
  failures that a second attempt could plausibly fix: timeouts, connection
  errors, 429, 5xx. Every other 4xx (bad key, malformed request, permission) is
  returned on the first attempt — retrying a rejected request just doubles the
  latency before the same answer.

  HEALTH — a call decorated `@with_retry(provider="groq")` reports its outcome
  (ok/fail, wall-clock latency, HTTP status) to core/provider_health.py, and is
  refused outright when that provider's circuit is open. This is the SINGLE choke
  point for both: every provider call in the system is already wrapped by this
  decorator, so naming the provider is the whole wiring — no timing, counting or
  circuit logic is scattered across call sites.

Two invariants worth stating explicitly:

  PRIVACY — a retry re-invokes the SAME function against the SAME book. It never
  selects a different provider, so it cannot move a turn onto a tier the router
  didn't choose: a secret/private turn retries ollama→ollama, never ollama→cloud.
  Book selection stays entirely in core/books.py. The health gate here is the same
  shape: it can only ever REFUSE a call, never redirect one.

  FALLBACK — `reraise=True` means the caller sees the ORIGINAL provider
  exception after the last attempt, not a tenacity wrapper. Every existing
  handler (call_book's rate-limit cooldown, the next-book loop in cognition, the
  fail-soft try/excepts in triage/extractor) keeps working untouched; retry is
  invisible to them apart from happening later. A refused call raises
  provider_health.CircuitOpen, which those same handlers treat as any other
  provider failure — fall to the next candidate.
"""

import functools
import re
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core import provider_health as health

CONNECT_S = 5.0
READ_S = 30.0
READ_LONG_S = 60.0       # deliverable / long-output cloud calls
READ_LOCAL_S = 120.0     # ollama: slow, not hung

MAX_ATTEMPTS = 2


def _timeout(read_s: float) -> httpx.Timeout:
    # write shares the read budget (a large prompt upload is the mirror of a large
    # response); pool shares connect — waiting for a free connection that long means
    # the pool is wedged.
    return httpx.Timeout(connect=CONNECT_S, read=read_s, write=read_s, pool=CONNECT_S)


CLOUD_TIMEOUT = _timeout(READ_S)
LONG_TIMEOUT = _timeout(READ_LONG_S)
LOCAL_TIMEOUT = _timeout(READ_LOCAL_S)

# Output size above which a call is treated as a long one (deliverables, lists,
# nemotron-scale answers). The default per-call cap is 1024, so this only lifts the
# read budget for calls that genuinely intend to produce more.
LONG_OUTPUT_TOKENS = 1024


def timeout_for(max_tokens: int | None) -> httpx.Timeout:
    """The cloud read budget this call should get, from what it intends to generate."""
    return LONG_TIMEOUT if (max_tokens or 0) > LONG_OUTPUT_TOKENS else CLOUD_TIMEOUT


# Transport-level failures — nothing was answered, so a second attempt is honest.
_TRANSPORT_ERRORS = (
    httpx.TimeoutException,      # connect/read/write/pool timeouts
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def _status_of(exc: BaseException) -> int | None:
    """HTTP status behind an exception, whichever shape the provider SDK used.
    None when the failure never got as far as a response."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


# Statuses that several SDKs only ever admit to in the message text — a Cerebras or
# ollama error can carry no status attribute at all while plainly saying "rate limit
# reached". The circuit breaker keys its two immediate-open rules on 429/401/403, so
# it has to see those however they arrived. Word-bounded on the numerics so a model
# id or a token count can't be read as a status.
_RATE_LIMIT_RE = re.compile(
    r"\b429\b|rate[ _-]?limit|too many requests|quota", re.IGNORECASE)
_AUTH_RE = re.compile(
    r"\b401\b|\b403\b|unauthorized|forbidden|invalid[ _-]?api[ _-]?key|"
    r"authentication", re.IGNORECASE)


def status_of(exc: BaseException) -> int | None:
    """The status a failure REPRESENTS: the SDK's own field where there is one, and
    the status the message admits to where there isn't. Used by the health choke
    point below — `_status_of` stays the strict, structural reading that retry
    decisions are made on."""
    status = _status_of(exc)
    if status is not None:
        return status
    text = str(exc)
    if _RATE_LIMIT_RE.search(text):
        return 429
    if _AUTH_RE.search(text):
        return 401
    return None


def is_retryable(exc: BaseException) -> bool:
    """True only for failures a second attempt could plausibly fix.

    Deliberately provider-agnostic: it reads the exception's shape rather than
    importing groq/openai/cerebras error classes, so core/books.py can keep
    importing those SDKs lazily and a provider that isn't installed never breaks
    this module."""
    # A refused call never happened. There is nothing to retry, and retrying would
    # burn the half-open probe budget on a provider we just declined to dial.
    if isinstance(exc, health.CircuitOpen):
        return False

    if isinstance(exc, _TRANSPORT_ERRORS):
        return True

    status = _status_of(exc)
    if status is not None:
        # 429 = ask again shortly; 5xx = their side, may pass. Every other 4xx is a
        # rejected request (auth, bad body, permission) — fail fast, don't repeat it.
        return status == 429 or status >= 500

    # No status: the SDKs wrap transport failures in their own APIConnectionError /
    # APITimeoutError and chain the httpx cause, so look through it.
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, _TRANSPORT_ERRORS):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


def _log_retry(state) -> None:
    exc = state.outcome.exception() if state.outcome else None
    fn = getattr(state.fn, "__name__", "provider call")
    print(f"[net] {fn} attempt {state.attempt_number}/{MAX_ATTEMPTS} failed "
          f"({type(exc).__name__}: {exc}) — retrying")


def _retried(fn):
    return retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=4.0, jitter=0.5),
        retry=retry_if_exception(is_retryable),
        before_sleep=_log_retry,
        reraise=True,  # callers keep seeing the provider's own exception
    )(fn)


def _recorded(fn, provider: str):
    """Report ONE attempt — wall-clock latency and its outcome. Inside the retry
    wrapper on purpose: each attempt is a real dial, so three failing attempts are
    three failures and trip the breaker as promptly as three failing calls would."""

    @functools.wraps(fn)
    def attempt(*args, **kwargs):
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            health.record_fail(provider, int((time.monotonic() - t0) * 1000),
                               status_of(e))
            raise
        health.record_ok(provider, int((time.monotonic() - t0) * 1000))
        return result

    return attempt


def _gated(attempt, retried, provider: str):
    """Check the circuit ONCE per logical call, OUTSIDE the retry wrapper.

    Outside matters. With the check inside, a 429 on attempt 1 would trip the
    breaker and attempt 2 would then raise CircuitOpen — the caller would see the
    breaker's exception instead of the provider's 429, and every handler that reads
    the original error (call_book's rate-limit cooldown, and the Nemotron 5-minute
    bench in particular) would quietly stop firing. The FALLBACK invariant in this
    module's docstring is load-bearing: a caller always sees the provider's own
    exception.

    A HALF-OPEN probe runs the UNRETRIED attempt — 'exactly one probe call' means
    exactly one dial, not one call that may quietly dial twice."""

    @functools.wraps(attempt)
    def call(*args, **kwargs):
        claim = health.begin_call(provider)
        if not claim:
            # Refused, not failed: nothing is recorded, because nothing happened.
            raise health.CircuitOpen(provider, health.circuit_of(provider))
        target = attempt if claim == health.PROBE else retried
        return target(*args, **kwargs)

    return call


def with_retry(fn=None, *, provider: str | None = None):
    """Wrap a BLOCKING provider call with the shared retry policy. Sync on purpose:
    every provider call in this codebase runs inside asyncio.to_thread, so the
    backoff sleeps on the worker thread and never blocks the event loop.

    Usable bare (`@with_retry`) or with a provider (`@with_retry(provider="groq")`).
    Naming the provider — the KEY/ENDPOINT, not the book; see
    provider_health.provider_of — is the entire wiring for health tracking and the
    circuit breaker. Without it the call is bounded and retried exactly as before
    and reports nothing, so an auxiliary call site can opt in later by adding one
    keyword and nothing else."""
    def decorate(f):
        if not provider:
            return _retried(f)
        attempt = _recorded(f, provider)
        return _gated(attempt, _retried(attempt), provider)

    return decorate(fn) if fn is not None else decorate


# ── Local model (ollama) ──
# One client for the whole process, built with the local read budget. The
# module-level ollama.chat() helper builds its own client with NO read timeout,
# which is exactly the unbounded call this module exists to remove.
_ollama_client = None


def ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(timeout=LOCAL_TIMEOUT)
    return _ollama_client


@with_retry(provider="ollama")
def ollama_chat(model: str, messages: list, options: dict = None, **kwargs):
    """Bounded, retried replacement for ollama.chat(). Same arguments, same return
    value — callers keep their existing fail-safe try/except around it."""
    return ollama_client().chat(model=model, messages=messages,
                                options=options or {}, **kwargs)
