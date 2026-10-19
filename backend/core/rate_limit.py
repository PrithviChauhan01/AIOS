"""Per-IP sliding-window rate limit for public mode.

Config.PUBLIC_RATE_DAILY (5 messages / rolling 24h). In-memory, so limits reset on
restart — fine for a single-process demo server. Behind a reverse proxy, the client
IP is whatever uvicorn resolved (its --proxy-headers / --forwarded-allow-ips
settings); this module never parses X-Forwarded-For itself, since that header is
client-controlled.
"""

import ipaddress
import threading
import time
from collections import defaultdict, deque

from config import Config

_lock = threading.Lock()
_hits: "defaultdict[str, deque]" = defaultdict(deque)  # ip → hit timestamps, oldest first
_clock = time.monotonic  # swapped in tests
_SWEEP_AT = 10000  # drop IPs idle for a full day once this many are tracked

LIMIT_MESSAGE = ("You've used all your free AIOS messages for today. "
                 "Thanks for trying it — come back tomorrow!")


def _bucket(ip: str) -> str:
    """IPv4 → the address; IPv6 → its /64. One IPv6 subscriber typically holds a
    whole /64, so keying on the full address would let them mint fresh quotas."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip or "unknown"
    if addr.version == 6:
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def _prune(ip: str, now: float) -> deque:
    """The live hits for `ip`. Caller holds _lock."""
    _, window = Config.PUBLIC_RATE_DAILY
    if len(_hits) > _SWEEP_AT:
        for key in [k for k, v in _hits.items() if not v or now - v[-1] > window]:
            del _hits[key]
    hits = _hits[_bucket(ip)]
    while hits and now - hits[0] > window:
        hits.popleft()
    return hits


def check(ip: str) -> tuple[bool, int, int, str | None]:
    """Record a hit for `ip` if allowed. Returns (allowed, remaining, retry_after_s,
    message). `remaining` is what's left AFTER this hit. A refused request is NOT
    recorded, so hammering a 429 doesn't extend the wait."""
    now = _clock()
    limit, window = Config.PUBLIC_RATE_DAILY
    with _lock:
        hits = _prune(ip, now)
        if len(hits) >= limit:
            retry = int(window - (now - hits[0])) + 1
            return False, 0, retry, LIMIT_MESSAGE
        hits.append(now)
        return True, limit - len(hits), 0, None


def remaining(ip: str) -> int:
    """Messages left for `ip` right now, without consuming one."""
    limit, _ = Config.PUBLIC_RATE_DAILY
    with _lock:
        return max(0, limit - len(_prune(ip, _clock())))


def reset() -> None:
    """Forget every IP (tests)."""
    with _lock:
        _hits.clear()
