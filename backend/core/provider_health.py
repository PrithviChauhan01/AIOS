"""Provider health + circuit breaker — the one place that knows whether a provider
is worth calling right now.

core/net.py already bounds a single call (timeout) and repeats one that could
plausibly succeed (retry). Neither has a memory: a provider whose key was revoked,
whose free tier is exhausted, or whose endpoint is simply gone was re-dialled on
every turn, paid the full connect+read budget every time, and only then fell to the
next book. Two attempts × five seconds × every book in the chain is a turn that
takes half a minute to say "something's off with my connection".

This module is that memory. Every provider call reports its outcome here (see the
single choke point in core/net.py), and book selection asks here before it offers a
candidate.

  STATE is per PROVIDER, not per book. groq and groq_fast are one key on one
  endpoint; nemotron_super and nemotron_ultra are one NVIDIA NIM account. A 401 on
  one IS a 401 on the other, so they share a circuit — see provider_of().

  CIRCUIT is the standard three-state breaker:
    closed    — normal. Failures accumulate; 3 CONSECUTIVE ones trip it.
    open      — the provider is skipped in selection for OPEN_SECONDS. Nothing is
                dialled, so a dead provider costs zero latency instead of a full
                timeout budget per turn.
    half_open — the cooldown lapsed. EXACTLY ONE probe call is allowed through
                (begin_call hands out a single token). It succeeds → closed and
                every counter resets; it fails → open again with the timer
                restarted. One probe, not a thundering herd.

  Two failures skip the 3-strike count and open immediately, because a second
  attempt cannot possibly help:
    401/403 — the key is bad. Retrying a rejected credential is pure latency.
    429     — the quota is spent. Hammering it is what keeps it spent.

  IN MEMORY ONLY. No table, no file. State is per process and starts empty, which
  is the correct default: a restart re-measures rather than inheriting a verdict
  about a provider that may well have recovered while we were down.

Three invariants worth stating explicitly:

  PRIVACY IS ABOVE THIS MODULE. A circuit can only ever REMOVE a candidate, never
  add one. core/books.py resolves the privacy tier FIRST (secret → ['ollama'],
  full stop) and consults this module afterwards, so an open local circuit on a
  secret turn means that turn FAILS — it does not escape to a cloud book. There is
  no code path here that can widen a tier, and there must never be one.

  NEVER RAISES TO THE USER. begin_call() refusing a call raises CircuitOpen at the
  choke point, which is an ordinary provider exception to every caller: books'
  fallback loop moves to the next candidate, and when the whole chain is exhausted
  cognition returns its existing degradation line. Nothing new reaches the user.

  OBSERVATION, NOT ASSESSMENT. Everything reported here is measured from real
  calls — outcomes, statuses, wall-clock latencies. No model is asked to judge
  whether a provider "seems healthy".
"""

import threading
import time

# ── Circuit states ──
CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

# 3 consecutive failures trip a healthy provider. Below 3, a single blip (one 5xx,
# one flaky connection) is exactly what net.py's retry is for and should not bench
# a provider that is otherwise answering.
FAIL_THRESHOLD = 3

# How long an open circuit stays shut before the probe. Long enough that a
# rate-limit window or a provider incident has a chance to pass, short enough that
# a recovered provider is back in rotation within a couple of turns.
OPEN_SECONDS = 120.0

# Rolling latency window per provider. ~50 samples is enough for a stable median
# and small enough that a provider that got slow an hour ago doesn't still say so.
LATENCY_WINDOW = 50

# A half-open probe holds its token until it reports back. If a probe never reports
# (killed thread, a call that outlived every timeout), the token would strand the
# provider in half_open forever. This releases it — comfortably longer than the
# longest read budget in net.py (120s local), so it can only ever fire on a probe
# that genuinely vanished.
_PROBE_MAX_S = 180.0

# Statuses that open the circuit on the FIRST failure. See the module docstring:
# retrying any of these is guaranteed to reproduce them.
_AUTH_STATUSES = (401, 403)
_RATE_LIMIT_STATUS = 429

# ── book → provider ──
# The provider is the thing that actually fails: one key, one endpoint, one quota.
# Books sharing one are deliberately collapsed onto a single circuit.
_PROVIDER_OF_BOOK = {
    "groq": "groq",
    "groq_fast": "groq",          # same api.groq.com key as groq
    "cerebras": "cerebras",
    "mistral": "mistral",
    "gpt4o": "openai",
    "nemotron_super": "nvidia",   # both Nemotrons are one NVIDIA NIM account
    "nemotron_ultra": "nvidia",
    "ollama": "ollama",
}

# Providers that live outside this process. `ollama` is deliberately NOT here: it is
# the local fallback and the only book a secret turn may use.
CLOUD_PROVIDERS = ("groq", "cerebras", "mistral", "openai", "nvidia")
LOCAL_PROVIDER = "ollama"


def provider_of(book: str) -> str:
    """The provider (key/endpoint/quota) behind a book. Unknown names map to
    themselves, so a book added to core.books without an entry here still gets its
    own circuit rather than silently sharing someone else's."""
    return _PROVIDER_OF_BOOK.get(book, book)


def books_of(provider: str) -> list:
    """Every book one circuit governs — the inverse of provider_of. Includes books
    outside CAPABILITY_MAP (groq_fast is reached only through the fast lane, but it
    dies with the same key)."""
    return sorted(b for b, p in _PROVIDER_OF_BOOK.items() if p == provider)


class CircuitOpen(Exception):
    """Raised by the choke point INSTEAD of dialling a provider whose circuit is
    open. Deliberately a plain exception: every existing caller already handles a
    provider raising (books' next-candidate loop, the teacher's fallback list,
    cognition's degradation return), so this needs no new handling anywhere.

    net.is_retryable returns False for it — there is nothing to retry."""

    def __init__(self, provider: str, state: str = OPEN):
        self.provider = provider
        self.state = state
        super().__init__(f"{provider} circuit {state} — call skipped")


class _State:
    __slots__ = ("provider", "last_ok_ts", "last_fail_ts", "consecutive_fails",
                 "total_calls", "total_fails", "latencies", "circuit", "opened_at",
                 "probe_started_at", "last_status", "last_reason")

    def __init__(self, provider: str):
        self.provider = provider
        self.last_ok_ts = None
        self.last_fail_ts = None
        self.consecutive_fails = 0
        self.total_calls = 0
        self.total_fails = 0
        self.latencies = []          # rolling, capped at LATENCY_WINDOW
        self.circuit = CLOSED
        self.opened_at = None
        self.probe_started_at = None  # set while a half-open probe is in flight
        self.last_status = None
        self.last_reason = None


_STATES: dict[str, _State] = {}
_LOCK = threading.Lock()   # provider calls run on asyncio.to_thread workers — real
                           # concurrency, so every mutation below holds this.


def _log(provider: str, frm: str, to: str, reason: str) -> None:
    print(f"[health] {provider} circuit {frm} -> {to} ({reason})")


def _state(provider: str) -> _State:
    """The state row for a provider. Caller MUST hold _LOCK."""
    st = _STATES.get(provider)
    if st is None:
        st = _State(provider)
        _STATES[provider] = st
    return st


def _refresh(st: _State, now: float) -> None:
    """Lazy time-driven transitions. Caller MUST hold _LOCK.

    There is no timer thread: an open circuit becomes half-open the first time
    anyone LOOKS at it after the cooldown, which is the same thing from every
    caller's point of view and costs nothing while the process is idle."""
    if st.circuit == OPEN and st.opened_at is not None:
        if now - st.opened_at >= OPEN_SECONDS:
            st.circuit = HALF_OPEN
            st.probe_started_at = None
            _log(st.provider, OPEN, HALF_OPEN,
                 f"{OPEN_SECONDS:.0f}s elapsed — one probe call allowed")
    if (st.circuit == HALF_OPEN and st.probe_started_at is not None
            and now - st.probe_started_at >= _PROBE_MAX_S):
        # The probe never reported back. Release the token rather than strand the
        # provider — the next caller gets a fresh probe.
        print(f"[health] {st.provider} half-open probe never reported "
              f"({_PROBE_MAX_S:.0f}s) — releasing the probe slot")
        st.probe_started_at = None


def _open(st: _State, now: float, reason: str) -> None:
    """Trip (or re-trip) the circuit. Caller MUST hold _LOCK."""
    was = st.circuit
    st.circuit = OPEN
    st.opened_at = now
    st.probe_started_at = None
    st.last_reason = reason
    _log(st.provider, was, OPEN, reason)


# ── Selection gate ──
def allow(provider: str) -> bool:
    """May book selection OFFER this provider as a candidate?

    True when the circuit is closed, or half-open with its probe still unspent.
    False when it is open (the 120s skip) or half-open with a probe already in
    flight. Read-only apart from the lazy open→half_open transition: selection must
    not consume the probe token for a book it may never actually call."""
    now = time.monotonic()
    with _LOCK:
        st = _state(provider)
        _refresh(st, now)
        if st.circuit == CLOSED:
            return True
        if st.circuit == HALF_OPEN:
            return st.probe_started_at is None
        return False


def allow_book(book: str) -> bool:
    """allow() keyed by BOOK name — the form core.books selects on."""
    return allow(provider_of(book))


# ── Call gate (the choke point calls these) ──
# What begin_call hands back. Both are truthy; None means refused. The choke point
# treats PROBE differently — a probe is ONE dial, never a retried call.
NORMAL = "normal"
PROBE = "probe"


def begin_call(provider: str) -> str | None:
    """Claim the right to make one call. None → the caller must NOT dial.

    This is where 'exactly one probe' is enforced: in half-open the first caller
    takes the token and every other caller is refused until that probe reports."""
    now = time.monotonic()
    with _LOCK:
        st = _state(provider)
        _refresh(st, now)
        if st.circuit == CLOSED:
            return NORMAL
        if st.circuit == HALF_OPEN and st.probe_started_at is None:
            st.probe_started_at = now
            print(f"[health] {st.provider} half-open — probing with one call")
            return PROBE
        return None


def record_ok(provider: str, latency_ms: int | None = None) -> None:
    """A call came back. Closes a half-open circuit and clears the failure streak."""
    now = time.monotonic()
    with _LOCK:
        st = _state(provider)
        st.total_calls += 1
        st.last_ok_ts = time.time()
        st.consecutive_fails = 0
        st.last_status = 200
        if latency_ms is not None:
            st.latencies.append(int(latency_ms))
            del st.latencies[:-LATENCY_WINDOW]
        if st.circuit != CLOSED:
            # The probe succeeded — full reset. The counters described the outage;
            # keeping them would misreport a provider that is now answering.
            _log(st.provider, st.circuit, CLOSED, "probe succeeded — counters reset")
            st.circuit = CLOSED
            st.opened_at = None
            st.total_fails = 0
            st.last_reason = None
        st.probe_started_at = None


def record_fail(provider: str, latency_ms: int | None = None,
                status: int | None = None) -> None:
    """A call failed. Trips the circuit on an auth error, a 429, a failed probe, or
    the FAIL_THRESHOLD-th consecutive failure.

    latency is recorded for failures too — a 30s timeout is a real measurement of
    how the provider is behaving, and hiding it would make p50 flatter than the
    truth."""
    now = time.monotonic()
    with _LOCK:
        st = _state(provider)
        st.total_calls += 1
        st.total_fails += 1
        st.consecutive_fails += 1
        st.last_fail_ts = time.time()
        st.last_status = status
        if latency_ms is not None:
            st.latencies.append(int(latency_ms))
            del st.latencies[:-LATENCY_WINDOW]

        was_probing = st.circuit == HALF_OPEN and st.probe_started_at is not None
        st.probe_started_at = None

        if was_probing:
            _open(st, now, f"probe failed (status={status}) — {OPEN_SECONDS:.0f}s again")
        elif status in _AUTH_STATUSES:
            _open(st, now, f"auth error {status} — the key is bad, retrying is pointless")
        elif status == _RATE_LIMIT_STATUS:
            _open(st, now, "429 — quota exhausted")
        elif st.circuit == CLOSED and st.consecutive_fails >= FAIL_THRESHOLD:
            _open(st, now, f"{st.consecutive_fails} consecutive failures")


# ── Reads ──
def _p50(latencies: list) -> int | None:
    if not latencies:
        return None
    ordered = sorted(latencies)
    return int(ordered[len(ordered) // 2])


def circuit_of(provider: str) -> str:
    """Current circuit state, applying any due transition first."""
    now = time.monotonic()
    with _LOCK:
        st = _state(provider)
        _refresh(st, now)
        return st.circuit


def open_providers() -> list:
    """Providers selection is currently SKIPPING — open, or half-open with the one
    probe already spent. This is exactly the set core.books drops from a candidate
    list, which is what makes it the honest 'skipped' value for a trace."""
    now = time.monotonic()
    out = []
    with _LOCK:
        for name, st in _STATES.items():
            _refresh(st, now)
            if st.circuit == OPEN or (st.circuit == HALF_OPEN
                                      and st.probe_started_at is not None):
                out.append(name)
    return sorted(out)


def circuit_summary() -> str | None:
    """Non-closed circuits as 'provider:state', comma-joined — or None when every
    provider is healthy (the common case, and a None keeps it out of trace meta
    entirely rather than writing 'all fine' on every row)."""
    now = time.monotonic()
    parts = []
    with _LOCK:
        for name, st in sorted(_STATES.items()):
            _refresh(st, now)
            if st.circuit != CLOSED:
                parts.append(f"{name}:{st.circuit}")
    return ",".join(parts) or None


def snapshot() -> dict:
    """Per-provider health for /health/deep. Measured facts only.

    Timestamps are unix epoch seconds (time.time), not the monotonic clock the
    circuit timers use — a caller needs a wall time it can compare to its own."""
    now = time.monotonic()
    out = {}
    with _LOCK:
        for name, st in _STATES.items():
            _refresh(st, now)
            out[name] = {
                "circuit": st.circuit,
                "consecutive_fails": st.consecutive_fails,
                "p50_latency_ms": _p50(st.latencies),
                "last_ok_ts": st.last_ok_ts,
                "last_fail_ts": st.last_fail_ts,
                "total_calls": st.total_calls,
                "total_fails": st.total_fails,
                "opened_at_age_s": (round(now - st.opened_at, 1)
                                    if st.opened_at is not None else None),
                "last_status": st.last_status,
                "reason": st.last_reason,
                "samples": len(st.latencies),
            }
    return out


def reset(provider: str | None = None) -> None:
    """Drop recorded state — one provider, or all of it. For tests and for a manual
    'I fixed the key, try again now' without a restart. Nothing in the request path
    calls this."""
    with _LOCK:
        if provider is None:
            _STATES.clear()
        else:
            _STATES.pop(provider, None)
