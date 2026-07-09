import re

# ── Per-session brainstorm / secret toggle ──
# Secret mode is an EXPLICIT, user-driven posture, OFF by default. When a session
# turns it ON, every turn for that session is forced local-only (ollama) regardless
# of domain — the orchestrator applies the override. This REPLACES the old auto-clamp
# where triage tagged domain=brainstorm and silently forced secret on every idea
# query (too eager, blocked legit cloud reasoning).
#
# State is in-memory, keyed by session_id. Single-user local app: a server restart
# resets the toggle to its safe explicit default (OFF). This flag is ONLY the
# brainstorm clamp's replacement — the STRUCTURAL privacy layer (hardcoded
# PAN/password/account regex in triage) is independent of it and always forces
# secret. The toggle can never weaken that layer.
_SECRET_MODE: dict[str, bool] = {}


def is_secret_mode(session_id: str) -> bool:
    return _SECRET_MODE.get(session_id, False)


def set_secret_mode(session_id: str, on: bool) -> None:
    _SECRET_MODE[session_id] = on


# Toggle phrases. OFF is checked first because "secret mode off" contains the ON
# phrase "secret mode" — order resolves that overlap deterministically.
_ON_PHRASES = (
    "brainstorm mode",
    "secret mode",
    "go private",
    "this is secret",
)
_OFF_PHRASES = (
    "exit brainstorm",
    "secret mode off",
    "normal mode",
    "you can go online",
)


def _normalize(message: str) -> str:
    """Lowercase and collapse punctuation so 'Secret mode, please.' still matches."""
    return re.sub(r"[^a-z0-9 ]+", " ", (message or "").lower()).strip()


def detect_toggle(message: str) -> str | None:
    """Return 'on' / 'off' if the message is a secret-mode toggle command, else None.
    OFF is matched first so 'secret mode off' isn't read as the ON phrase 'secret mode'."""
    norm = _normalize(message)
    if not norm:
        return None
    if any(p in norm for p in _OFF_PHRASES):
        return "off"
    if any(p in norm for p in _ON_PHRASES):
        return "on"
    return None
