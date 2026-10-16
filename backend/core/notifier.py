"""Reminder notification channel — one interface, swappable backend.

Picked by Config.NOTIFIER (env NOTIFIER), default 'toast':

  toast — Windows toast via win11toast (laptop-local; the original behaviour).
  email — mail to Sir's own GMAIL_ADDRESS via tools/mailer.send_to_self. This
          skips the confirm gate, which is safe ONLY because the recipient is
          hardwired to self — no caller can aim it at anyone else.
  log   — print only. For headless servers with no other channel.

Every backend is best-effort: notify() never raises, so a failed delivery can
never stop the scheduler tick.
"""

import os
import traceback

from config import Config

# win11toast's default app_id is 'Python', which has NO registered Start-Menu
# identity — so Windows accepts notifier.show() (no exception) and then SILENTLY
# DROPS the banner. Routing through the built-in PowerShell AppUserModelID gives
# the toast a registered identity Windows will actually render. Override with
# AIOS_TOAST_APP_ID if you register a branded "AIOS" shortcut later.
_TOAST_APP_ID = os.getenv("AIOS_TOAST_APP_ID") or (
    r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
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


# ── Backends ──
class ToastNotifier:
    name = "toast"

    def notify(self, title: str, body: str) -> None:
        from win11toast import notify  # non-blocking fire-and-forget
        notify(title, body, app_id=_TOAST_APP_ID)
        print(f"[reminder] toast SENT (app_id={_TOAST_APP_ID!r})")
        _warn_if_suppressed()

    def check(self) -> None:
        from win11toast import notify  # noqa: F401
        print(f"[reminder] notifier=toast ready (app_id={_TOAST_APP_ID!r})")
        if _toasts_enabled() is False:
            print("[reminder] WARNING: ToastEnabled=0 — banners are OFF; reminders "
                  "will fire but not pop until notifications are re-enabled.")


class EmailNotifier:
    name = "email"

    def notify(self, title: str, body: str) -> None:
        from tools.mailer import send_to_self
        res = send_to_self(title, body)
        if res.get("ok"):
            print(f"[reminder] email SENT to self subject={title!r}")
        else:
            print(f"[reminder] email FAILED: {res.get('error')}")

    def check(self) -> None:
        import tools.mailer  # noqa: F401
        if not (Config.GMAIL_ADDRESS and Config.GMAIL_APP_PASSWORD):
            print("[reminder] WARNING: notifier=email but GMAIL_ADDRESS / "
                  "GMAIL_APP_PASSWORD are not set — reminders will not be delivered.")
        else:
            print(f"[reminder] notifier=email ready (to={Config.GMAIL_ADDRESS})")


class LogNotifier:
    name = "log"

    def notify(self, title: str, body: str) -> None:
        print(f"[reminder] NOTIFY {title}: {body}")

    def check(self) -> None:
        print("[reminder] notifier=log ready")


_BACKENDS = {"toast": ToastNotifier, "email": EmailNotifier, "log": LogNotifier}
_notifier = None


def get_notifier():
    global _notifier
    if _notifier is None:
        choice = (Config.NOTIFIER or "toast").strip().lower()
        cls = _BACKENDS.get(choice)
        if cls is None:
            print(f"[reminder] unknown NOTIFIER={choice!r} — falling back to log")
            cls = LogNotifier
        _notifier = cls()
    return _notifier


def notify(title: str, body: str) -> None:
    """Deliver one notification on the configured backend. Never raises."""
    n = get_notifier()
    try:
        n.notify(title, body)
    except Exception as e:
        print(f"[reminder] {n.name} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def check() -> None:
    """Startup diagnostic: confirm the backend is usable before any reminder is due."""
    n = get_notifier()
    try:
        n.check()
    except Exception as e:
        print(f"[reminder] notifier={n.name} import failed: {e}")
