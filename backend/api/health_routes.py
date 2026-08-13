"""Liveness (/health) and the deep, per-provider health report (/health/deep).

/health is unchanged: an unauthenticated "the process is up" probe.

/health/deep is the operator view — is a provider actually answering, and if not,
why. Two rules define what it is allowed to say:

  DETERMINISTIC. Every number here is measured. Circuit state, failure streaks,
  p50 latency and last-success timestamps come from the outcomes of REAL calls
  (core/provider_health.py, fed by the choke point in core/net.py); the checks come
  from real round trips (a SQLite query, a Chroma heartbeat, a models-list against
  the provider's own key). Nothing is estimated, and no model is ever asked whether
  something "looks healthy" — an LLM's opinion of an LLM provider is not a check.

  NON-MUTATING. Reading health never changes routing. The probes deliberately do
  not report to provider_health, so refreshing this endpoint cannot open or close a
  circuit, and a diagnostic can never become the cause of the thing it diagnoses.

Cloud probes are opt-in (?ping=true) because they are real API calls against the
same free tiers the assistant runs on. The default answer is built from the call
outcomes already recorded this process, which costs nothing and is a better signal
besides: it reflects the calls that actually matter. The local ollama daemon is
always probed — it is free, and it is the ONLY book a secret turn may use, so
whether it is up is the single most load-bearing fact on this page.
"""

import asyncio
import sqlite3
import time

from fastapi import APIRouter, Depends

from config import Config
from core import provider_health as health
from core.auth import require_auth
from core.books import key_configured, ping_provider
from core.provider_health import CLOUD_PROVIDERS, LOCAL_PROVIDER, books_of

router = APIRouter()
START_TIME = time.time()

# Cloud first, local last — the order an operator reads a fallback chain in.
_PROVIDERS = list(CLOUD_PROVIDERS) + [LOCAL_PROVIDER]


def _uptime() -> str:
    seconds = int(time.time() - START_TIME)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "vision": False,
        "memory": "connected",
        "uptime": _uptime(),
    }


# ── Deterministic local checks (blocking — run off the event loop) ──
def _check_sqlite() -> dict:
    """A real query against the real database file, not an os.path.exists()."""
    t0 = time.monotonic()
    try:
        conn = sqlite3.connect(Config.SQLITE_PATH, timeout=2.0)
        try:
            tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        finally:
            conn.close()
        return {"ok": True, "ms": int((time.monotonic() - t0) * 1000),
                "tables": tables, "path": Config.SQLITE_PATH, "error": None}
    except Exception as e:
        return {"ok": False, "ms": int((time.monotonic() - t0) * 1000),
                "path": Config.SQLITE_PATH, "error": f"{type(e).__name__}: {e}"[:200]}


def _check_chroma() -> dict:
    """Heartbeat plus the real size of the collection cognition actually reads."""
    t0 = time.monotonic()
    try:
        from db.chroma_init import get_chroma_client
        client = get_chroma_client()
        client.heartbeat()
        facts = client.get_or_create_collection("long_term_memory").count()
        return {"ok": True, "ms": int((time.monotonic() - t0) * 1000),
                "long_term_memory": facts, "error": None}
    except Exception as e:
        return {"ok": False, "ms": int((time.monotonic() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}"[:200]}


@router.get("/health/deep", dependencies=[Depends(require_auth)])
async def health_deep(ping: bool = False):
    """Per-provider health: circuit state, consecutive_fails, p50_latency_ms,
    last_ok_ts — plus the local store checks.

    ping=true additionally makes ONE real reachability call per configured cloud
    provider (a models list — no completion, no tokens). Off by default: those are
    live API calls against the same quotas the assistant needs.

    `status` is 'ok' when SQLite answers and at least one provider is both
    configured and not circuit-open; 'degraded' otherwise. Deterministic, and never
    a judgement call.
    """
    recorded = health.snapshot()

    # ollama is always probed — free, local, and the only book a secret turn can use.
    targets = _PROVIDERS if ping else [LOCAL_PROVIDER]
    checks, probes = await asyncio.gather(
        asyncio.gather(asyncio.to_thread(_check_sqlite),
                       asyncio.to_thread(_check_chroma)),
        asyncio.gather(*(asyncio.to_thread(ping_provider, p) for p in targets)),
    )
    sqlite_check, chroma_check = checks
    probe_of = dict(zip(targets, probes))

    providers = {}
    for name in _PROVIDERS:
        # A provider that has never been called has no recorded state — report the
        # empty circuit rather than omitting it, so the list is the full roster.
        stats = recorded.get(name, {
            "circuit": health.CLOSED, "consecutive_fails": 0, "p50_latency_ms": None,
            "last_ok_ts": None, "last_fail_ts": None, "total_calls": 0,
            "total_fails": 0, "opened_at_age_s": None, "last_status": None,
            "reason": None, "samples": 0,
        })
        providers[name] = {
            **stats,
            "books": books_of(name),
            "key_configured": key_configured(name),
            "probe": probe_of.get(name),  # None when not probed this call
        }

    live = [n for n, p in providers.items()
            if p["key_configured"] and p["circuit"] != health.OPEN]
    return {
        "status": "ok" if (sqlite_check["ok"] and live) else "degraded",
        "uptime": _uptime(),
        "probed": bool(ping),
        "checks": {"sqlite": sqlite_check, "chroma": chroma_check},
        "providers": providers,
        "open_circuits": health.open_providers(),
    }
