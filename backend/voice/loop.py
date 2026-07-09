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
"""

import asyncio
import sys
import threading
import time

import numpy as np

from core.orchestrator import handle_message
from voice.speaker import speak
from voice import activity
from voice import listener
from voice.listener import capture
from voice.transcriber import transcribe

_STOP_WORDS = ("stop", "stop listening")


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
        result = await handle_message(text, session_id="voice", voice_flag=False)
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
