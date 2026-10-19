"""Owner login for POST /token (personal mode): bcrypt password + 6-digit TOTP.

Both factors are always evaluated — no short-circuit — so response timing doesn't
reveal which one was wrong, and the caller only ever learns pass/fail. Every
comparison is constant-time (bcrypt.checkpw, hmac.compare_digest).

Lockout: 3 failures inside 15 minutes locks that client IP for 15 minutes. There is
deliberately no global lock — it would let anyone lock the owner out from anywhere.
While locked, even correct credentials are refused without being checked.

A TOTP code is single-use: once a time-step's code has logged in, that step and
every earlier one are spent, so a shoulder-surfed code can't be replayed inside its
validity window.

State is in-memory (single process); a restart clears lockouts.
"""

import hmac
import re
import threading
import time
from collections import deque

import bcrypt
import pyotp

from config import Config

MAX_FAILS = 3
LOCK_S = 15 * 60
_STEP_S = 30
_CODE_RE = re.compile(r"^\d{6}$")
# Checked when no hash is configured, so a misconfigured server costs the same time.
_DUMMY_HASH = bcrypt.hashpw(b"aios-unconfigured", bcrypt.gensalt())

_lock = threading.Lock()
_fails: dict[str, deque] = {}       # key → failure timestamps inside the window
_locked_until: dict[str, float] = {}
_last_step = -1                     # highest TOTP time-step already used to log in
_clock = time.time                  # swapped in tests


def _keys(ip: str) -> tuple[str, str]:
    return (f"ip:{ip or 'unknown'}",)


def _is_locked(key: str, now: float) -> bool:
    until = _locked_until.get(key)
    if until is None:
        return False
    if now >= until:
        del _locked_until[key]
        return False
    return True


def _record_failure(key: str, now: float) -> None:
    q = _fails.setdefault(key, deque())
    q.append(now)
    while q and now - q[0] > LOCK_S:
        q.popleft()
    if len(q) >= MAX_FAILS:
        _locked_until[key] = now + LOCK_S
        q.clear()


def _check_password(password: str) -> bool:
    stored = (Config.OWNER_PASSWORD_HASH or "").encode("utf-8")
    try:
        candidate = (password or "").encode("utf-8")[:72]  # bcrypt's input limit
        if not stored:
            bcrypt.checkpw(candidate, _DUMMY_HASH)
            return False
        return bcrypt.checkpw(candidate, stored)
    except ValueError:  # malformed stored hash
        return False


def _check_totp(code: str, now: float) -> int | None:
    """The matching time-step for `code` (±1 step of clock drift), or None.
    All three candidates are compared; a spent step never matches."""
    secret = Config.OWNER_TOTP_SECRET or ""
    code = (code or "").strip()
    if not secret or not _CODE_RE.match(code):
        return None
    try:
        totp = pyotp.TOTP(secret)
    except Exception:
        return None
    step = int(now // _STEP_S)
    match = None
    for s in (step - 1, step, step + 1):
        expected = totp.at(s * _STEP_S)
        if hmac.compare_digest(expected, code) and s > _last_step:
            match = s
    return match


def attempt(ip: str, password: str, code: str) -> str:
    """'ok' | 'invalid' | 'locked'. Serialised under one lock so concurrent guesses
    can't all slip past the counter before the first failure is recorded."""
    global _last_step
    with _lock:
        now = _clock()
        keys = _keys(ip)
        if any(_is_locked(k, now) for k in keys):
            return "locked"
        pw_ok = _check_password(password)
        step = _check_totp(code, now)
        if pw_ok and step is not None:
            _last_step = step
            for k in keys:
                _fails.pop(k, None)
            return "ok"
        for k in keys:
            _record_failure(k, now)
        return "invalid"


def reset() -> None:
    """Clear lockouts and the spent-code marker (tests)."""
    global _last_step
    with _lock:
        _fails.clear()
        _locked_until.clear()
        _last_step = -1
