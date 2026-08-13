"""Trace spine — one row per request, N rows per stage.

Every turn already prints a `[orch:xxxxxxxx]` log line per stage. That id IS this
module's trace_id (core.orchestrator builds the Trace first and reads trace.trace_id
into ctx), so the logs and these tables are the same story told twice: once for a
human tailing stdout, once queryable after the fact.

Contract — all four of these are load-bearing:

  * PASSIVE. A Trace observes; it never decides anything. Nothing in routing,
    triage, privacy tiers, cognition or idempotency reads it.
  * BUFFERED. Stages accumulate in memory. There is exactly ONE database write per
    request — a single transaction holding 1 traces row + N trace_stages rows —
    issued from the orchestrator's finally block. No stage touches the DB.
  * FAIL-SOFT. flush() swallows every error and returns False. A trace failure must
    never fail or slow a request, so the stage context manager degrades to an inert
    no-op object rather than raising, and the orchestrator runs flush off the event
    loop thread.
  * METADATA ONLY. Message content, retrieved material, tool payloads and vault data
    NEVER reach these tables. Stage meta is filtered against ALLOWED_META_KEYS below
    — an unlisted key is DROPPED, not stored — so a careless call site cannot leak
    content, and a secret-tier turn writes exactly the same shape of metadata as a
    public one (labels, book names, counts) and none of its substance.

Token counts come from what providers actually report (core.books surfaces
usage.prompt_tokens / usage.completion_tokens, and ollama's prompt_eval_count /
eval_count, on every call_book result). Nothing here estimates or tokenizes; a
provider that reports no split leaves the column NULL.
"""

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from config import Config
from db.sqlite_init import TRACE_DDL, ensure_trace_schema

# ── Ambient request state ──
# Two things need to reach code that has no ctx to thread a trace through:
#
#   _STAGE_STACK — the seq of the stage currently open, so a stage opened INSIDE
#   another records it as parent_seq. `book` runs inside `cognition`, not after it;
#   a flat list of stages cannot say that, and seq alone reads as "cognition
#   finished, then a book ran". seq stays true start order AND parent_seq carries
#   containment, so the two never disagree.
#
#   _CURRENT_TRACE — the trace for the in-flight request, for tool call sites deep
#   in tools/ that are reached by several different paths (leadgen calls
#   tools.places.search_places directly, bypassing the registry).
#
# ContextVars are the right primitive here rather than instance state: asyncio
# copies the context per Task, so the concurrent fan-out teachers and the parallel
# ensemble book calls each get their OWN stack instead of trampling a shared one,
# and asyncio.to_thread carries the context onto the worker thread.
_STAGE_STACK: ContextVar[tuple] = ContextVar("aios_trace_stage_stack", default=())
_CURRENT_TRACE: ContextVar = ContextVar("aios_current_trace", default=None)

# ── Metadata whitelist ──
# The structural guarantee that no content lands in trace_stages.meta. A key that is
# not here is dropped at flush time with a warning — adding a new stage field is a
# deliberate act, never an accident. Nothing in this list can hold user text, tool
# payloads or retrieved material: they are classification labels, book/model names,
# counts and booleans.
ALLOWED_META_KEYS = frozenset({
    # entry
    "msg_len", "history_msgs", "voice",
    # triage
    "sensitivity", "complexity", "domain", "loop_worthy",
    # action_dispatch — the tool and verb that ran, never their arguments or result
    "tool", "verb", "ok", "detected", "idem_replay",
    # teacher
    "tier", "eff_tier", "ensemble", "deliverable", "self_check_passed", "books", "tools",
    # book
    "book", "model", "max_tokens",
    # looper
    "attempt", "attempts", "passed", "confidence",
    # cognition
    "gen_tier", "fast_lane", "lane", "local_only", "material_tier", "mood",
    # provider health — WHY a book chain looked the way it did. Provider names and
    # circuit states only ('groq:open'), never a key, an endpoint or an error body.
    "circuits", "skipped",
})

# Trace-level summary columns a caller may set via Trace.set(). ts_end, total_ms,
# teachers and the token totals are DERIVED at flush and are not settable.
_SUMMARY_FIELDS = (
    "sensitivity", "complexity", "domain", "teachers", "deliverable", "fast_lane",
    "secret_mode", "final_book", "loop_attempts", "tools_used", "idempotency_hit",
    "error",
)
# Stored 0/1 (nullable — None means "never reached that stage", which is not False).
_BOOL_SUMMARY_FIELDS = ("deliverable", "fast_lane", "secret_mode", "idempotency_hit")

_MAX_META_STR = 120     # a meta string value is a label; anything longer is a bug
_MAX_ERROR_STR = 200

# Sortable, SQLite-datetime-shaped, UTC. String comparison is a valid time
# comparison in this format, which is what the ?since= filter relies on.
_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)[:-3]


def _safe_err(err) -> str | None:
    """An exception → a short, single-line, TRUNCATED marker.

    Type name first because that is the part that is always safe and always useful.
    The message tail is capped hard: a provider SDK error can in principle echo part
    of what was sent, and 200 characters of a status line is worth having while a
    full error body is not."""
    if err is None:
        return None
    if isinstance(err, BaseException):
        text = f"{type(err).__name__}: {err}"
    else:
        text = str(err)
    return " ".join(text.split())[:_MAX_ERROR_STR] or None


def _safe_scalar(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(sorted(str(v)[:40] for v in value))[:_MAX_META_STR] or None
    return str(value)[:_MAX_META_STR]


def _safe_meta(meta: dict) -> str | None:
    """Stage meta → a JSON string of WHITELISTED keys only. Unlisted keys are dropped
    (with a warning) — see ALLOWED_META_KEYS."""
    if not meta:
        return None
    out = {}
    for key, value in meta.items():
        if key not in ALLOWED_META_KEYS:
            print(f"[trace] warning: meta key {key!r} is not whitelisted — dropped")
            continue
        out[key] = _safe_scalar(value)
    if not out:
        return None
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return None


class _Stage:
    """One buffered stage record. Mutated in place by its `with` body; written only
    by Trace.flush()."""

    __slots__ = ("stage", "seq", "parent_seq", "ts_start", "ts_end", "ms", "book",
                 "tokens_in", "tokens_out", "meta", "error", "_t0")

    def __init__(self, stage: str, seq: int, parent_seq, initial: dict):
        self.stage = stage
        self.seq = seq
        self.parent_seq = parent_seq
        self.ts_start = _now()
        self.ts_end = None
        self.ms = None
        self.book = None
        self.tokens_in = None
        self.tokens_out = None
        self.meta = {}
        self.error = None
        self._t0 = time.monotonic()
        if initial:
            self.set(**initial)

    def set(self, **fields):
        """Set stage columns; anything else goes to meta (and is whitelisted at
        flush). Never raises — a bad trace call must not break a turn."""
        try:
            for key, value in fields.items():
                if key in ("book", "tokens_in", "tokens_out"):
                    setattr(self, key, value)
                elif key == "error":
                    self.error = _safe_err(value)
                else:
                    self.meta[key] = value
        except Exception as e:  # pragma: no cover — belt and braces
            print(f"[trace] warning: stage.set failed ({e})")
        return self

    def close(self):
        if self.ts_end is None:
            self.ts_end = _now()
            self.ms = int((time.monotonic() - self._t0) * 1000)


class _NullStage:
    """Inert stand-in handed to call sites when there is no trace (direct teacher
    invocation, tests) or when opening a stage failed. Swallows everything."""

    def set(self, **fields):
        return self

    def close(self):
        pass


@contextmanager
def _null_stage():
    yield _NullStage()


class Trace:
    """One request's trace. Buffered in memory, flushed exactly once."""

    def __init__(self, session_id: str = "default", trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.session_id = session_id or "default"
        self.ts_start = _now()
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._stages: list[_Stage] = []
        self._seq = 0
        self._tools: set[str] = set()
        self._flushed = False
        self.summary = {field: None for field in _SUMMARY_FIELDS}
        # These three are DEFINITE from the moment a request starts, so they are
        # seeded rather than left NULL. NULL has to keep meaning "never determined"
        # (which is the right answer for e.g. deliverable on a turn that never
        # reached cognition) — it must NOT double as "zero". A turn where the looper
        # was skipped genuinely made 0 refinement attempts; a turn with no
        # idempotency replay genuinely had no hit; a turn that drew no tools
        # genuinely used none. tools_used is a comma-joined list, so its empty value
        # is the empty string.
        self.summary["loop_attempts"] = 0
        self.summary["idempotency_hit"] = False

    # ── Recording ──
    @contextmanager
    def stage(self, name: str, **initial):
        """Record one stage, nested under whichever stage is currently open.

        seq is assigned at OPEN, so it is true start order; parent_seq is the seq of
        the enclosing stage (None at top level), so `book` inside `cognition` reads
        as contained rather than as the step after it.

        An exception propagates UNCHANGED (the stage just records that it failed),
        so wrapping an existing call in a stage cannot alter control flow."""
        try:
            stack = _STAGE_STACK.get()
            with self._lock:
                self._seq += 1
                st = _Stage(name, self._seq, stack[-1] if stack else None, initial)
                self._stages.append(st)
            token = _STAGE_STACK.set(stack + (st.seq,))
        except Exception as e:  # pragma: no cover — tracing must never break a turn
            print(f"[trace] warning: could not open stage {name!r} ({e})")
            yield _NullStage()
            return
        try:
            yield st
        except BaseException as e:
            st.error = _safe_err(e)
            raise
        finally:
            try:
                _STAGE_STACK.reset(token)
            except ValueError:
                # Token minted in another context (a stage whose enter/exit straddle
                # a task boundary). Nothing to unwind here — the child context died
                # with its own copy of the stack.
                pass
            st.close()

    def set(self, **fields):
        """Set trace-level summary columns. Unknown fields are ignored."""
        try:
            for key, value in fields.items():
                if key not in _SUMMARY_FIELDS:
                    print(f"[trace] warning: unknown summary field {key!r} — ignored")
                    continue
                if key == "error":
                    value = _safe_err(value)
                self.summary[key] = value
        except Exception as e:  # pragma: no cover
            print(f"[trace] warning: trace.set failed ({e})")
        return self

    def add_tools(self, names) -> None:
        """Union tool names that ACTUALLY fired this turn into tools_used."""
        try:
            for name in (names or []):
                if name:
                    self._tools.add(str(name)[:40])
        except Exception as e:  # pragma: no cover
            print(f"[trace] warning: add_tools failed ({e})")

    # ── Derived summary ──
    def _teachers(self) -> str | None:
        seen = []
        for st in self._stages:
            if st.stage.startswith("teacher:"):
                domain = st.stage.split(":", 1)[1]
                if domain not in seen:
                    seen.append(domain)
        return ",".join(seen) or None

    def _token_totals(self) -> tuple[int | None, int | None]:
        """Summed from the `book` stages ONLY — they are the sole stages carrying
        provider-reported counts, so nothing is double counted by the teacher /
        cognition spans that contain them. None (not 0) when no book reported a
        split: a missing number is not a zero."""
        totals = [None, None]
        for idx, attr in enumerate(("tokens_in", "tokens_out")):
            values = [getattr(st, attr) for st in self._stages
                      if st.stage == "book" and isinstance(getattr(st, attr), int)]
            totals[idx] = sum(values) if values else None
        return totals[0], totals[1]

    # ── The one write ──
    def flush(self) -> bool:
        """Write this trace: ONE transaction, 1 traces row + N trace_stages rows.

        Never raises and never re-writes — a second call is a no-op. Returns True on
        a successful write, False if anything at all went wrong (the request is
        unaffected either way)."""
        try:
            with self._lock:
                if self._flushed:
                    return False
                self._flushed = True
                stages = list(self._stages)

            ts_end = _now()
            total_ms = int((time.monotonic() - self._t0) * 1000)
            tokens_in, tokens_out = self._token_totals()
            summary = dict(self.summary)
            # Never NULL: the empty string is the honest "no tool was invoked".
            explicit = summary.get("tools_used")
            if isinstance(explicit, (list, tuple, set)):
                self.add_tools(explicit)
            elif isinstance(explicit, str) and explicit:
                self.add_tools(explicit.split(","))
            tools_used = ",".join(sorted(self._tools))

            for field in _BOOL_SUMMARY_FIELDS:
                if summary.get(field) is not None:
                    summary[field] = int(bool(summary[field]))

            trace_row = (
                self.trace_id, self.session_id, self.ts_start, ts_end, total_ms,
                summary.get("sensitivity"), summary.get("complexity"),
                summary.get("domain"), self._teachers(), summary.get("deliverable"),
                summary.get("fast_lane"), summary.get("secret_mode"),
                summary.get("final_book"), tokens_in, tokens_out,
                int(summary.get("loop_attempts") or 0), tools_used,
                int(bool(summary.get("idempotency_hit"))), summary.get("error"),
            )
            stage_rows = [
                (self.trace_id, st.stage, st.seq, st.parent_seq, st.ts_start,
                 st.ts_end, st.ms, st.book, st.tokens_in, st.tokens_out,
                 _safe_meta(st.meta), st.error)
                for st in stages
            ]

            conn = sqlite3.connect(Config.SQLITE_PATH)
            try:
                # Self-contained like core.outcomes: works even if init_sqlite() has
                # not run in this process. executescript commits on its own, before
                # the single transaction below.
                ensure_trace_schema(conn)
                with conn:  # ONE transaction — both inserts land or neither does
                    conn.execute(
                        "INSERT OR REPLACE INTO traces "
                        "(trace_id, session_id, ts_start, ts_end, total_ms, sensitivity, "
                        " complexity, domain, teachers, deliverable, fast_lane, secret_mode, "
                        " final_book, tokens_in, tokens_out, loop_attempts, tools_used, "
                        " idempotency_hit, error) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        trace_row,
                    )
                    conn.execute("DELETE FROM trace_stages WHERE trace_id = ?",
                                 (self.trace_id,))
                    conn.executemany(
                        "INSERT INTO trace_stages "
                        "(trace_id, stage, seq, parent_seq, ts_start, ts_end, ms, book, "
                        " tokens_in, tokens_out, meta, error) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        stage_rows,
                    )
            finally:
                conn.close()
            return True
        except Exception as e:
            print(f"[trace] warning: flush failed ({e}) — trace dropped, request unaffected")
            return False


# ── Call-site helpers ──
# Stages instrument THEMSELVES through these: one line, no branching on whether a
# trace exists. A ctx with no trace (direct teacher invocation, a test) gets the
# inert no-op, so instrumentation never changes how a stage behaves.
def _trace_of(ctx) -> Trace | None:
    if not isinstance(ctx, dict):
        return None
    trace = ctx.get("trace")
    return trace if isinstance(trace, Trace) else None


def stage_of(ctx, name: str, **initial):
    """`with stage_of(ctx, "book", book=b) as st:` — the stage CM for ctx's trace,
    or an inert no-op when there is none."""
    trace = _trace_of(ctx)
    if trace is None:
        return _null_stage()
    return trace.stage(name, **initial)


def mark(ctx, **fields) -> None:
    """Set trace-level summary columns from inside a stage. No-op without a trace."""
    trace = _trace_of(ctx)
    if trace is not None:
        trace.set(**fields)


def mark_tools(ctx, names) -> None:
    """Record tools that actually fired this turn. No-op without a trace."""
    trace = _trace_of(ctx)
    if trace is not None:
        trace.add_tools(names)


@contextmanager
def activate(trace: Trace):
    """Make `trace` the ambient trace for this request, so call sites with no ctx
    (the tool layer) can record against it. Scoped to the calling task's context —
    concurrent requests never see each other's trace."""
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        try:
            _CURRENT_TRACE.reset(token)
        except ValueError:  # pragma: no cover — reset from a different context
            pass


def record_tool(*names) -> None:
    """Record tool NAMES against the ambient trace, from anywhere in the tool layer.

    This is the counterpart to mark_tools for code that never receives a ctx. The
    two are interchangeable and idempotent — tools_used is a set, so a tool reached
    through both the registry and its own structured entry point is recorded once.
    Names only: a tool's query and its results never touch a trace."""
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.add_tools(names)


# ── Reads (the /trace and /traces endpoints) ──
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


_TRACE_COLUMNS = (
    "trace_id, session_id, ts_start, ts_end, total_ms, sensitivity, complexity, "
    "domain, teachers, deliverable, fast_lane, secret_mode, final_book, tokens_in, "
    "tokens_out, loop_attempts, tools_used, idempotency_hit, error"
)


def get_trace(trace_id: str) -> dict | None:
    """One trace with its stages ordered by seq, or None if there is no such trace."""
    conn = _conn()
    try:
        row = conn.execute(
            f"SELECT {_TRACE_COLUMNS} FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        stages = conn.execute(
            "SELECT id, stage, seq, parent_seq, ts_start, ts_end, ms, book, tokens_in, "
            "tokens_out, meta, error FROM trace_stages WHERE trace_id = ? ORDER BY seq ASC",
            (trace_id,),
        ).fetchall()
    finally:
        conn.close()

    out, depth_of = [], {}
    for st in stages:
        stage = dict(st)
        if stage.get("meta"):
            try:
                stage["meta"] = json.loads(stage["meta"])
            except Exception:
                pass  # keep the raw string rather than lose the row
        # Rendering convenience: how deep this stage sits under its parent chain.
        # Derived, not stored — parent_seq is the source of truth.
        parent = stage.get("parent_seq")
        stage["depth"] = depth_of[parent] + 1 if parent in depth_of else 0
        depth_of[stage["seq"]] = stage["depth"]
        out.append(stage)
    return {"trace": dict(row), "stages": out}


def list_traces(session_id: str | None = None, limit: int = 50,
                since: str | None = None) -> list:
    """Summary rows, newest first. `since` filters on ts_start and takes the same
    'YYYY-MM-DD HH:MM:SS.mmm' shape the rows carry (a bare date works too — string
    comparison is a time comparison in this format)."""
    try:
        limit = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        limit = 50

    where, params = [], []
    if session_id:
        where.append("session_id = ?")
        params.append(session_id)
    if since:
        where.append("ts_start >= ?")
        params.append(since)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    params.append(limit)

    conn = _conn()
    try:
        rows = conn.execute(
            f"SELECT {_TRACE_COLUMNS} FROM traces{clause} "
            "ORDER BY ts_start DESC LIMIT ?", tuple(params)
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
