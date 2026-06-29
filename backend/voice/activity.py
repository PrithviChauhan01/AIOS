"""Cross-process voice-activity flag.

The voice loop runs as its OWN process (`python -m voice.loop`); the reminder
scheduler runs inside the FastAPI backend process. They can't share memory, so
"is voice active right now?" is signalled through a heartbeat file in the temp
dir that both processes agree on.

While listening, the loop calls mark_active() each tick (a fresh mtime). The
scheduler checks is_active() before speaking a reminder — if the heartbeat is
fresh, voice is live and it speaks; otherwise it stays silent (toast only).
mark_inactive() clears the flag on a clean toggle-off; the freshness window also
covers the case where the loop process dies without cleaning up.
"""

import os
import tempfile
import time

_FLAG = os.path.join(tempfile.gettempdir(), "aios_voice_active")

# A heartbeat older than this means voice is no longer live (e.g. loop crashed).
# Comfortably larger than one capture cycle so a long VAD wait doesn't read stale.
_FRESH_SECONDS = 60


def mark_active() -> None:
    try:
        with open(_FLAG, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def mark_inactive() -> None:
    try:
        os.remove(_FLAG)
    except OSError:
        pass


def is_active() -> bool:
    try:
        return (time.time() - os.path.getmtime(_FLAG)) < _FRESH_SECONDS
    except OSError:
        return False
