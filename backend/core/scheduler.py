"""Reminder firing layer.

A BackgroundScheduler ticks every 30s inside the FastAPI process (started from
the lifespan). Each tick pulls reminders that have come due and delivers them
laptop-locally on two channels:

  1. The configured notifier (core/notifier.py; NOTIFIER=toast|email|log).
  2. Kokoro speech — ONLY if the voice loop is currently live (voice.activity).

After firing: recurring reminders (a recognized `repeat` cadence) get their
due_at advanced to the next occurrence; one-shots are marked done=1.

Everything here is best-effort: a failed notification or TTS must never stop the tick
or crash the backend. No phone/cloud delivery — that's post-Oracle.
"""

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from core import notifier
from tools.reminders import (
    due_reminders,
    complete_reminder,
    reschedule_reminder,
    next_occurrence,
)

_TICK_SECONDS = 30
_scheduler = None


def test_toast() -> None:
    """Standalone toast smoke test — runnable OUTSIDE the scheduler thread:

        python -c "from core.scheduler import test_toast; test_toast()"

    Confirms whether toasts render at all on this machine, independent of the
    BackgroundScheduler. If this shows a popup but a fired reminder doesn't, the
    problem is thread/timing. If this shows nothing either, it's Windows
    notification settings (see the ToastEnabled note), not the scheduler."""
    print("[reminder] test_toast: sending a test notification...")
    notifier.notify("AIOS", "test")
    if notifier.get_notifier().name == "toast" and notifier._toasts_enabled() is not False:
        print(
            "[reminder] test_toast: sent. If no popup appeared, check Focus "
            "Assist / Do Not Disturb and this app id's per-app notification "
            "setting in Settings > System > Notifications."
        )


def _maybe_speak(text: str) -> None:
    """Speak only if the voice loop is live — otherwise the backend would load
    Kokoro and grab the audio device for nothing."""
    try:
        from voice.activity import is_active
        if not is_active():
            return
        from voice.speaker import speak
        speak(text, "neutral")
    except Exception as e:
        print(f"[reminder] voice delivery failed: {e}")


# ── Tick ──
def _tick() -> None:
    # Per-tick diagnostics (tick now / total_pending / per-row DUE verdict) and
    # the scheduler DB path now live inside due_reminders() in tools/reminders.py,
    # so they print on every tick from a single source.
    try:
        due = due_reminders()
    except Exception as e:
        print(f"[reminder] tick query failed: {e}")
        return

    now = datetime.now()
    for r in due:
        rid, title, repeat = r["id"], r["title"], r.get("repeat")
        spoken = f"Sir, reminder: {title}."
        notifier.notify("AIOS Reminder", title)
        _maybe_speak(spoken)
        print(f"[reminder] fired id={rid} title={title}")

        nxt = next_occurrence(r["due_at"], repeat, now) if repeat else None
        if nxt is not None:
            reschedule_reminder(rid, nxt)
            print(f"[reminder] rescheduled id={rid} -> {nxt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            complete_reminder(rid)


# ── Lifecycle (called from FastAPI lifespan) ──
def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    # [diag 3] Confirm the notification channel works at startup, before any
    # reminder is due — so a missing/broken backend shows up now, not silently later.
    notifier.check()

    sched = BackgroundScheduler(daemon=True)
    # coalesce + max_instances=1: if the process was asleep, run the tick once on
    # resume rather than stacking missed ticks. next_run_time=now → fire promptly.
    sched.add_job(
        _tick, "interval", seconds=_TICK_SECONDS, id="reminder_tick",
        next_run_time=datetime.now(), coalesce=True, max_instances=1,
    )
    sched.start()
    _scheduler = sched
    print(f"[reminder] scheduler started — ticking every {_TICK_SECONDS}s")
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        print("[reminder] scheduler stopped")
