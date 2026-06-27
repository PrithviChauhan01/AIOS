"""Reminder firing layer.

A BackgroundScheduler ticks every 30s inside the FastAPI process (started from
the lifespan). Each tick pulls reminders that have come due and delivers them
laptop-locally on two channels:

  1. A Windows toast (win11toast) — pops even when the terminal is minimized.
  2. Kokoro speech — ONLY if the voice loop is currently live (voice.activity).

After firing: recurring reminders (a recognized `repeat` cadence) get their
due_at advanced to the next occurrence; one-shots are marked done=1.

Everything here is best-effort: a failed toast or TTS must never stop the tick
or crash the backend. No phone/cloud delivery — that's post-Oracle.
"""

import os
import traceback
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from tools.reminders import (
    due_reminders,
    complete_reminder,
    reschedule_reminder,
    next_occurrence,
)

_TICK_SECONDS = 30
_scheduler = None

# win11toast's default app_id is 'Python', which has NO registered Start-Menu
# identity — so Windows accepts notifier.show() (no exception) and then SILENTLY
# DROPS the banner. Routing through the built-in PowerShell AppUserModelID gives
# the toast a registered identity Windows will actually render. Override with
# AIOS_TOAST_APP_ID if you register a branded "AIOS" shortcut later.
_TOAST_APP_ID = os.getenv(
    "AIOS_TOAST_APP_ID",
    r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe",
)


def _toasts_enabled():
    """Read HKCU\\...\\PushNotifications\\ToastEnabled. 0 means Windows is
    suppressing ALL banners (the Notifications master toggle is off, or Focus
    Assist is on) — show() still succeeds but nothing pops. Returns True/False,
    or None when the value isn't present (Windows then defaults to enabled)."""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\PushNotifications",
        ) as k:
            val, _ = winreg.QueryValueEx(k, "ToastEnabled")
            return bool(val)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _warn_if_suppressed() -> None:
    """If Windows is globally suppressing banners, say so loudly — otherwise a
    'toast SENT' line with no popup looks like a code bug when it's a settings
    issue. Only prints when we can confirm ToastEnabled=0."""
    if _toasts_enabled() is False:
        print(
            "[reminder] NOTE: Windows ToastEnabled=0 -- notification banners are "
            "turned OFF. The toast was accepted but will NOT pop. Re-enable at "
            "Settings > System > Notifications, and turn OFF Focus Assist / Do "
            "Not Disturb."
        )


# ── Delivery channels (each fully best-effort) ──
def _toast(title: str, body: str) -> None:
    try:
        from win11toast import notify  # non-blocking fire-and-forget
        notify(title, body, app_id=_TOAST_APP_ID)
        print(f"[reminder] toast SENT (app_id={_TOAST_APP_ID!r})")
        _warn_if_suppressed()
    except Exception as e:
        print(f"[reminder] toast FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_toast() -> None:
    """Standalone toast smoke test — runnable OUTSIDE the scheduler thread:

        python -c "from core.scheduler import test_toast; test_toast()"

    Confirms whether toasts render at all on this machine, independent of the
    BackgroundScheduler. If this shows a popup but a fired reminder doesn't, the
    problem is thread/timing. If this shows nothing either, it's Windows
    notification settings (see the ToastEnabled note), not the scheduler."""
    print("[reminder] test_toast: sending a test notification...")
    _toast("AIOS", "test")
    if _toasts_enabled() is not False:
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
        _toast("AIOS Reminder", title)
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

    # [diag 3] Confirm the toast channel imports at startup, before any reminder
    # is due — so a missing/broken toast backend shows up now, not silently later.
    try:
        from win11toast import notify  # noqa: F401
        print(f"[reminder] toast backend=win11toast ready (app_id={_TOAST_APP_ID!r})")
        if _toasts_enabled() is False:
            print("[reminder] WARNING: ToastEnabled=0 — banners are OFF; reminders "
                  "will fire but not pop until notifications are re-enabled.")
    except Exception as e:
        print(f"[reminder] toast import failed: {e}")

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
