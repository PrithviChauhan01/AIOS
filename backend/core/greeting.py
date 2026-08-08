"""Bare-greeting short circuit.

"hey" came back as "Not busy, doing some light maintenance on the systems." —
nothing in the session asked that, and nothing in the system had been doing
maintenance. That is the trivial fast lane (llama-3.1-8b under a small output
cap) confabulating: given almost no context and a prompt full of persona, an 8B
invents a plausible-sounding status report. There is no answer to "hey" worth a
model call, so the fix is to not make one.

This module is a pure recogniser: it matches a message that is ONLY a greeting /
thanks / acknowledgement / sign-off and hands back her fixed line for it. Anything
carrying an actual request — "hey can you find me studios" — does NOT match and
routes normally, because the match is against the WHOLE normalised message, never
a prefix or a substring.

Her lines come straight from the persona (core/router.py:SYSTEM_PROMPT — "When
he's casual, you're easy and warm. 'Hey, Sir.' Not a status report", "If he says
'hello,' you say hello", confirmations are minimal: "Done." "On it." "Noted.")
and the voice examples in cognition's prompt (Sir: "Thanks." → You: "Always,
Sir."). The greeting itself is time-of-day aware, which is the one thing about a
greeting that legitimately varies.

Nothing here touches triage, routing, privacy tiers or the cognition prompt — it
sits in front of all of them, and only for messages that carry no request at all.
"""

import re
from datetime import datetime

# Doubled/elongated letters are collapsed on BOTH sides before matching, so one
# canonical spelling covers every stretched variant Sir actually types:
# "heyyy" → "hey", "hii" → "hi", "okkk" → "ok", "hello" → "helo" (and the
# canonical "hello" collapses to "helo" too, so they still meet).
_RUN_RE = re.compile(r"(.)\1+")
# Everything that isn't a letter, digit or space: punctuation, emoji, "!!!".
_STRIP_RE = re.compile(r"[^a-z0-9 ]+")
# Trailing address to her — "hey aios", "morning there", "thanks sir".
_ADDRESS_RE = re.compile(r"\s+(sir|aios|there|man|buddy)$")

_GREET = (
    "hi", "hey", "helo", "hello", "yo", "sup", "whats up", "what's up", "wassup",
    "howdy", "hiya", "greetings", "good morning", "morning", "good afternoon",
    "afternoon", "good evening", "evening", "gm", "hey you", "hi again",
    "long time", "you there", "you up",
)
_THANKS = (
    "thanks", "thank you", "thanks a lot", "thanks so much", "thank you so much",
    "thx", "ty", "cheers", "appreciate it", "much appreciated", "nice one",
    "thanks again",
)
_ACK = (
    "ok", "okay", "k", "kk", "cool", "nice", "great", "got it", "gotcha",
    "understood", "sure", "alright", "all right", "right", "fine", "sounds good",
    "perfect", "noted", "makes sense", "fair enough", "good", "yep", "yup",
    "mhm", "indeed",
)
_BYE = (
    "bye", "goodbye", "bye bye", "see ya", "see you", "see you later", "later",
    "cya", "good night", "goodnight", "night", "gn", "take care", "im off",
    "i'm off", "signing off", "thats all", "that's all",
)


def _collapse(text: str) -> str:
    return _RUN_RE.sub(r"\1", text)


def _normalise(message: str) -> str:
    """Lowercase, drop punctuation/emoji, squeeze whitespace, drop a trailing form
    of address, then collapse letter runs. The result is compared WHOLE."""
    t = _STRIP_RE.sub(" ", (message or "").lower())
    t = " ".join(t.split())
    t = _ADDRESS_RE.sub("", t).strip()
    return _collapse(t)


# Canonical phrases collapsed the same way, so both sides of the comparison are
# in the same form. Built once at import.
_KINDS = {
    "greet": {_collapse(p) for p in _GREET},
    "thanks": {_collapse(p) for p in _THANKS},
    "ack": {_collapse(p) for p in _ACK},
    "bye": {_collapse(p) for p in _BYE},
}

# A hard ceiling on what can even be considered: a bare greeting is a few words.
# Belt and braces on top of the whole-string match.
_MAX_WORDS = 4


def detect_greeting(message: str) -> str | None:
    """'greet' | 'thanks' | 'ack' | 'bye' when the message is ONLY that, else None.
    "hey can you find me studios" returns None — it is not in any phrase set."""
    norm = _normalise(message)
    if not norm or len(norm.split()) > _MAX_WORDS:
        return None
    for kind, phrases in _KINDS.items():
        if norm in phrases:
            return kind
    return None


def _time_of_day(now: datetime = None) -> str:
    h = (now or datetime.now()).hour
    if h < 12:
        return "Morning"
    if h < 17:
        return "Afternoon"
    return "Evening"


def greeting_reply(kind: str, now: datetime = None) -> tuple:
    """(text, mood) for a detected kind — her line, no model involved. The mood is
    set here explicitly, so this path always carries a real one."""
    now = now or datetime.now()
    if kind == "greet":
        return f"{_time_of_day(now)}, Sir.", "warm"
    if kind == "thanks":
        return "Always, Sir.", "warm"
    if kind == "bye":
        late = now.hour >= 21 or now.hour < 5
        return ("Goodnight, Sir." if late else "Later, Sir."), "warm"
    return "Noted, Sir.", "neutral"  # ack
