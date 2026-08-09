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

Two invariants worth stating explicitly:

  PRIVACY — a retry re-invokes the SAME function against the SAME book. It never
  selects a different provider, so it cannot move a turn onto a tier the router
  didn't choose: a secret/private turn retries ollama→ollama, never ollama→cloud.
  Book selection stays entirely in core/books.py.

  FALLBACK — `reraise=True` means the caller sees the ORIGINAL provider
  exception after the last attempt, not a tenacity wrapper. Every existing
  handler (call_book's rate-limit cooldown, the next-book loop in cognition, the
  fail-soft try/excepts in triage/extractor) keeps working untouched; retry is
  invisible to them apart from happening later.
"""

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

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


def is_retryable(exc: BaseException) -> bool:
    """True only for failures a second attempt could plausibly fix.

    Deliberately provider-agnostic: it reads the exception's shape rather than
    importing groq/openai/cerebras error classes, so core/books.py can keep
    importing those SDKs lazily and a provider that isn't installed never breaks
    this module."""
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


def with_retry(fn):
    """Wrap a BLOCKING provider call with the shared retry policy. Sync on purpose:
    every provider call in this codebase runs inside asyncio.to_thread, so the
    backoff sleeps on the worker thread and never blocks the event loop."""
    return retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=4.0, jitter=0.5),
        retry=retry_if_exception(is_retryable),
        before_sleep=_log_retry,
        reraise=True,  # callers keep seeing the provider's own exception
    )(fn)


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


@with_retry
def ollama_chat(model: str, messages: list, options: dict = None, **kwargs):
    """Bounded, retried replacement for ollama.chat(). Same arguments, same return
    value — callers keep their existing fail-safe try/except around it."""
    return ollama_client().chat(model=model, messages=messages,
                                options=options or {}, **kwargs)
