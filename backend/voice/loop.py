"""AIOS voice loop — TOGGLE mode. Terminal-first, captions printed.

Press Enter to toggle listening ON. While ON, the loop runs hands-free:

    capture one utterance (VAD until silence) → transcribe
      ├ text is "stop" / "stop listening" → toggle OFF → idle
      └ else → "You: {text}" → handle_message → "AIOS: {response}" → speak → capture again

Press Enter again any time → toggle OFF → idle until pressed again. Ctrl+C exits clean.

A daemon thread reads Enter and flips a threading.Event; the async capture loop checks
that flag between utterances. handle_message is async; blocking mic/STT/TTS work is pushed
to threads so the event loop stays free.

Run from backend/:  python -m voice.loop

Remote mode: if AIOS_API_URL and AIOS_TOKEN are both set, each utterance is POSTed
to {AIOS_API_URL}/chat with a Bearer token instead of calling handle_message
in-process — so the mic/speaker can stay on the laptop while the backend runs elsewhere.
"""

import asyncio
import os
import sys
import threading
import time

import numpy as np
import requests

import config  # noqa: F401 — loads backend/.env so AIOS_API_URL/AIOS_TOKEN are visible
from voice.speaker import speak
from voice import activity
from voice import listener
from voice.listener import capture
from voice.transcriber import transcribe

_STOP_WORDS = ("stop", "stop listening")

_API_URL = os.getenv("AIOS_API_URL", "").rstrip("/")
_TOKEN = os.getenv("AIOS_TOKEN", "")
_REMOTE_TIMEOUT_S = 180  # covers a slow local-book turn on the server side


def _remote_chat(text: str) -> dict:
    """Blocking POST to the remote backend's /chat. Same result shape as handle_message."""
    r = requests.post(
        f"{_API_URL}/chat",
        json={"message": text, "session_id": "voice", "voice": False},
        headers={"Authorization": f"Bearer {_TOKEN}"},
        timeout=_REMOTE_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


async def _chat(text: str) -> dict:
    if _API_URL and _TOKEN:
        try:
            return await asyncio.to_thread(_remote_chat, text)
        except Exception as e:
            print(f"[loop] remote /chat failed: {type(e).__name__}: {e}")
            return {"response": "I couldn't reach the server, Sir.", "mood": "neutral"}
    from core.orchestrator import handle_message  # in-process: loads the full backend
    return await handle_message(text, session_id="voice", voice_flag=False)


def _enter_watcher(toggle):
    """Blocking input() loop on a daemon thread — each Enter press flips the toggle."""
    while True:
        try:
            input()
        except EOFError:
            return
        toggle()


async def _run():
    listening = threading.Event()

    def toggle():
        if listening.is_set():
            listening.clear()
            activity.mark_inactive()  # tell the reminder scheduler voice is off
            print("[loop] listening OFF — press Enter to resume.")
        else:
            listening.set()
            print("[loop] listening ON — speak, or say 'stop' to pause.")

    threading.Thread(target=_enter_watcher, args=(toggle,), daemon=True).start()
    if _API_URL and _TOKEN:
        print(f"[loop] remote mode — POSTing to {_API_URL}/chat")
    print("[loop] AIOS voice loop ready. Press Enter to start/stop listening. Ctrl+C to exit.")

    while True:
        if not listening.is_set():
            await asyncio.sleep(0.1)
            continue

        activity.mark_active()  # heartbeat — reminders may speak while we're live

        audio = await asyncio.to_thread(capture)
        if not listening.is_set():
            continue  # toggled off mid-capture — discard

        # capture diagnostics (set by listener._capture_speech on the call above)
        diag = dict(listener.last_capture)
        vad_wait_ms = diag.get("vad_wait_ms", 0.0)
        capture_ms = diag.get("capture_ms", 0.0)
        amp = float(np.abs(audio).max()) if audio.size else 0.0
        if not diag.get("vad_fired"):
            vad_status = "no-speech"
        elif diag.get("timed_out"):
            vad_status = "timed-out"
        else:
            vad_status = "fired"

        t0 = time.perf_counter()
        text_groq = await asyncio.to_thread(transcribe, audio, diag.get("raw_rms"), False)
        text_local = await asyncio.to_thread(transcribe, audio, diag.get("raw_rms"), True)
        print(f"[GROQ ] {text_groq!r}")
        print(f"[LOCAL] {text_local!r}")
        text = text_groq
        transcribe_ms = (time.perf_counter() - t0) * 1000.0

        if not text:
            print(
                f"[t] capture={capture_ms:.0f}ms vad_wait={vad_wait_ms:.0f}ms "
                f"transcribe={transcribe_ms:.0f}ms cognition=0ms | "
                f"amp={amp:.3f} vad={vad_status} -> EMPTY (amp={amp:.3f})"
            )
            continue

        if text.strip().lower().strip(" .!?,") in _STOP_WORDS:
            print(
                f"[t] capture={capture_ms:.0f}ms vad_wait={vad_wait_ms:.0f}ms "
                f"transcribe={transcribe_ms:.0f}ms cognition=0ms | "
                f"amp={amp:.3f} vad={vad_status} -> STOP"
            )
            listening.clear()
            activity.mark_inactive()  # tell the reminder scheduler voice is off
            print("[loop] listening OFF — press Enter to resume.")
            continue

        print(f"You: {text}")
        t0 = time.perf_counter()
        result = await _chat(text)
        cognition_ms = (time.perf_counter() - t0) * 1000.0
        response = result.get("response", "")
        mood = result.get("mood", "neutral")
        print(f"AIOS: {response}")
        await asyncio.to_thread(speak, response, mood)
        print(
            f"[t] capture={capture_ms:.0f}ms vad_wait={vad_wait_ms:.0f}ms "
            f"transcribe={transcribe_ms:.0f}ms cognition={cognition_ms:.0f}ms | "
            f"amp={amp:.3f} vad={vad_status}"
        )


def main():
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[loop] shutting down. Goodbye, Sir.")
        sys.exit(0)


if __name__ == "__main__":
    main()
