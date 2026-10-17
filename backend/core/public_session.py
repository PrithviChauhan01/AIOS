"""In-memory session history for public mode.

A public visitor's conversation lives here and nowhere else — never SQLite, never
Chroma. Each session is a short rolling window of turns that expires after
Config.PUBLIC_SESSION_IDLE_S of inactivity; a restart drops everything, which is
correct for anonymous traffic.

Session ids are server-issued (secrets.token_urlsafe) and act as bearer handles: a
client that sends an unknown or expired id is simply issued a fresh one, so a
visitor can never pick an id and can never land in someone else's history.
"""

import secrets
import threading
import time
from collections import OrderedDict

from config import Config

_HISTORY_MSGS = 20       # same window personal mode loads from SQLite
_MAX_SESSIONS = 5000     # memory bound; least-recently-used sessions are dropped first

_lock = threading.Lock()
# session_id → (last_seen_monotonic, [ {role, content}, ... ]), oldest first
_sessions: "OrderedDict[str, tuple[float, list]]" = OrderedDict()
_clock = time.monotonic  # swapped in tests


def _sweep(now: float) -> None:
    """Drop idle-expired sessions. OrderedDict is kept in last-seen order, so the
    scan stops at the first live one. Caller holds _lock."""
    idle = Config.PUBLIC_SESSION_IDLE_S
    while _sessions:
        sid, (seen, _) = next(iter(_sessions.items()))
        if now - seen <= idle:
            break
        del _sessions[sid]


def resolve(session_id: str | None) -> str:
    """The live session for this id, or a freshly issued one. Touches last-seen."""
    now = _clock()
    with _lock:
        _sweep(now)
        if session_id and session_id in _sessions:
            _, history = _sessions.pop(session_id)
            _sessions[session_id] = (now, history)
            return session_id
        sid = secrets.token_urlsafe(16)
        _sessions[sid] = (now, [])
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
        return sid


def get_history(session_id: str) -> list:
    with _lock:
        _sweep(_clock())
        entry = _sessions.get(session_id)
        return [dict(t) for t in entry[1]] if entry else []


def append_turn(session_id: str, user_message: str, reply: str) -> None:
    """Record one exchange and refresh last-seen. A session that expired mid-turn is
    not resurrected — the reply still goes out, the history just starts over."""
    now = _clock()
    with _lock:
        entry = _sessions.pop(session_id, None)
        if entry is None:
            return
        history = entry[1]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        del history[:-_HISTORY_MSGS]
        _sessions[session_id] = (now, history)


def reset() -> None:
    """Drop every session (tests)."""
    with _lock:
        _sessions.clear()
