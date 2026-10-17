"""Per-IP sliding-window rate limit for public mode.

Two windows, both must pass: Config.PUBLIC_RATE_SHORT (20 / 10 min) and
Config.PUBLIC_RATE_DAILY (100 / day). In-memory, so limits reset on restart — fine
for a single-process demo server. Behind a reverse proxy, the client IP is whatever
uvicorn resolved (its --proxy-headers / --forwarded-allow-ips settings); this module
never parses X-Forwarded-For itself, since that header is client-controlled.
"""

import threading
import time
from collections import defaultdict, deque

from config import Config

_lock = threading.Lock()
_hits: "defaultdict[str, deque]" = defaultdict(deque)  # ip → hit timestamps, oldest first
_clock = time.monotonic  # swapped in tests
_SWEEP_AT = 10000  # drop IPs idle for a full day once this many are tracked


def check(ip: str) -> tuple[bool, int, str | None]:
    """Record a hit for `ip` if allowed. Returns (allowed, retry_after_s, message).
    A refused request is NOT recorded, so hammering a 429 doesn't extend the wait."""
    now = _clock()
    short_n, short_s = Config.PUBLIC_RATE_SHORT
    day_n, day_s = Config.PUBLIC_RATE_DAILY
    with _lock:
        if len(_hits) > _SWEEP_AT:
            for key in [k for k, v in _hits.items() if not v or now - v[-1] > day_s]:
                del _hits[key]
        hits = _hits[ip or "unknown"]
        while hits and now - hits[0] > day_s:
            hits.popleft()

        if len(hits) >= day_n:
            retry = int(day_s - (now - hits[0])) + 1
            return False, retry, (
                "You've hit today's message limit for the AIOS demo. "
                "Thanks for trying it — come back tomorrow!")

        recent = [t for t in hits if now - t <= short_s]
        if len(recent) >= short_n:
            retry = int(short_s - (now - recent[0])) + 1
            minutes = max(1, round(retry / 60))
            return False, retry, (
                "You're sending messages a little fast. Please take a breather and "
                f"try again in about {minutes} minute{'s' if minutes != 1 else ''}.")

        hits.append(now)
        return True, 0, None


def reset() -> None:
    """Forget every IP (tests)."""
    with _lock:
        _hits.clear()
