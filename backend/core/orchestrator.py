import asyncio
import os
import time
import uuid

from core.brain import plan
from core.outcomes import log_outcome
from core.triage import triage
from tools.registry import fetch_from, format_pool_block
from core.secret_mode import detect_toggle, is_secret_mode, set_secret_mode
from core.memory import get_history
from core.cognition import cognition_pass
from core.looper import run_looper
from core.action_dispatch import detect_action, run_action, action_material, fn_name
from agents.leadgen import LeadgenTeacher, get_cached_leadgen, is_export_request
from agents.study import StudyTeacher
from agents.work import WorkTeacher
from agents.fitness import FitnessTeacher
from agents.spirit import SpiritTeacher
from agents.life import LifeTeacher
from agents.brainstorm import BrainstormTeacher
from agents.jobs import JobsTeacher

# Domain → Teacher. Unknown domains fall through to the short-circuit path
# (cognition handles them herself, no teacher).
TEACHERS = {
    "leadgen": LeadgenTeacher,
    "study": StudyTeacher,
    "work": WorkTeacher,
    "fitness": FitnessTeacher,
    "spirit": SpiritTeacher,
    "life": LifeTeacher,
    "brainstorm": BrainstormTeacher,
    "jobs": JobsTeacher,
}


def _log(trace_id: str, stage: str, msg: str):
    print(f"[orch:{trace_id[:8]}] {stage}: {msg}")


# ── General-path (domain=none) tiering ──
# The none short-circuit used to ALWAYS hit the 8B fast lane, so real knowledge
# questions ("explain CAP theorem") got the weakest book and truncated. These two
# helpers size a book tier from complexity for the general path only — teachers keep
# their own (already-correct) tiers, and privacy/secret is handled downstream.

# Reasoning verbs that mark a genuine knowledge/analysis question (vs casual chat).
# The local 3B triage routinely under-rates these as 'trivial'; this lifts them off
# the fast lane without trusting that guess.
_REASON_VERBS = ("explain", "compare", "design", "why", "how does",
                 "walk through", "analyze", "tradeoff")
_REASON_STARTS = ("explain", "compare", "design", "analyze", "walk through")


def _general_complexity(message: str, complexity: str) -> str:
    """Heuristic UPGRADE for the general path only: a reasoning-verb question is at
    LEAST 'simple', so it leaves the trivial fast lane. Only ever raises trivial→simple
    — never downgrades, never touches an already simple/complex rating."""
    if complexity != "trivial":
        return complexity
    low = (message or "").lower().strip()
    starts = low.startswith(_REASON_STARTS)
    contains = any(v in low for v in _REASON_VERBS)
    if starts or (contains and len(low.split()) > 6):
        return "simple"
    return complexity


def _brain_gen_tier(brain: dict, complexity: str, loop_worthy: bool) -> str:
    """Map the brain's per-task plan → a cognition book tier (consumed via
    ctx['gen_tier']):
       strong/frontier plan, loop_worthy, or complex → 'strong' (nemotron chain)
       trivial small-talk with NO tools drawn → 'fast_lane' (groq_fast, sub-second)
       everything else (real lookups, tool-backed asks)  → 'fast' (groq 70b chain)
    The brain replaces the old pure-complexity mapping, but the complexity heuristics
    stay as an upgrade-only floor so a brain under-rating can't strand a real question
    on the small-talk lane."""
    if brain.get("book_tier") in ("strong", "frontier") or loop_worthy or complexity == "complex":
        return "strong"
    if complexity == "trivial" and not brain.get("tools"):
        return "fast_lane"
    return "fast"


# ── Slice 8: multi-domain async fan-out ──
# Hard backstop on top of brain.py's own MAX_FANOUT_DOMAINS cap — quota/latency bound.
_MAX_FANOUT = 3


async def _run_fanout_teacher(domain: str, base_ctx: dict, trace_id: str) -> dict:
    """Run ONE fanned-out teacher's FULL Slice X flow independently: its OWN per-domain
    brain call (tools/tier/ensemble scoped to just this domain, not the router brain's
    plan) → teacher.run() → the same ensemble/looper/plain material resolution the
    single-domain path uses. Runs concurrently with the other domains via the caller's
    asyncio.gather — this is what makes each teacher's Slice X flow independent instead
    of sharing one plan across domains. Never raises: a failed domain contributes no
    material (an honest gap) rather than crashing the whole multi-domain turn."""
    domain_ctx = {**base_ctx, "domain": domain}
    teacher = TEACHERS[domain]()

    sub_brain = await plan(
        base_ctx.get("message") or "", domain=domain,
        sensitivity=base_ctx.get("sensitivity", "public"),
        complexity=base_ctx.get("complexity"), loop_worthy=base_ctx.get("loop_worthy"),
        default_tier=teacher.reasoning_tier(domain_ctx),
    )
    domain_ctx["brain"] = sub_brain
    _log(trace_id, "fanout_brain",
         f"domain={domain} source={sub_brain['source']} "
         f"prompt={'composed' if sub_brain.get('composed_prompt') else 'static-template'} "
         f"tools={sub_brain['tools']} tier={sub_brain['book_tier']} "
         f"ensemble={sub_brain['ensemble']} — {sub_brain['reason']}")

    t0 = time.monotonic()
    try:
        teach = await teacher.run(domain_ctx)
    except Exception as e:
        _log(trace_id, "fanout_teacher", f"domain={domain} FAILED: {e}")
        return {"domain": domain, "material": None, "deliverable": False}

    # Same three-way resolution the single-domain path applies (ensemble → skip looper;
    # loop_worthy → looper; else the plain teach material) — just scoped per domain here.
    ensemble = teach.get("ensemble")
    if ensemble:
        material = "\n\n".join(
            (r.get("raw_text") or "").strip() for r in ensemble if r.get("raw_text"))
        out_book = "+".join(r.get("book_used", "?") for r in ensemble)
        out_passed = any(r.get("self_check", {}).get("passes") for r in ensemble)
        out_attempts = 1
    elif base_ctx.get("loop_worthy"):
        looped = await run_looper(domain_ctx, teacher, teach["raw_text"])
        material = looped["refined_material"]
        out_book = looped.get("book_used") or teach.get("book_used", "none")
        out_passed = looped.get("confidence") == "high"
        out_attempts = looped.get("attempts", 1)
    else:
        material = teach.get("raw_text", "")
        out_book = teach.get("book_used", "none")
        out_passed = bool(teach.get("self_check", {}).get("passes"))
        out_attempts = 1

    latency_ms = int((time.monotonic() - t0) * 1000)
    _log(trace_id, "fanout_teacher",
         f"domain={domain} book={out_book} tokens={teach.get('tokens')} "
         f"ensemble={bool(ensemble)} deliverable={teach.get('deliverable')} "
         f"latency_ms={latency_ms}")

    # Same passive outcome row Slice A writes for a single-domain teacher task — one
    # row per teacher that actually ran, domain-tagged.
    log_outcome(
        domain=domain, book_used=out_book, ensemble=bool(ensemble),
        self_check_passed=out_passed, attempts=out_attempts,
        tokens=teach.get("tokens", 0), latency_ms=latency_ms,
        complexity=base_ctx.get("complexity"), session_id=base_ctx.get("session_id", "default"),
    )

    return {"domain": domain, "material": material, "deliverable": bool(teach.get("deliverable"))}


async def handle_message(message: str, session_id: str = "default", voice_flag: bool = False) -> dict:
    """The spine. Wires triage → (teacher → looper) → cognition end to end.
    A trace_id flows through every stage for debuggability."""
    trace_id = str(uuid.uuid4())

    # ── [1] ENTRY — light: load history, build context. No reasoning here. ──
    ctx = {
        "message": message,
        "session_id": session_id,
        "history": get_history(session_id),
        "trace_id": trace_id,
        "voice": voice_flag,
    }
    _log(trace_id, "entry", f"session={session_id} len={len(message)}")

    # ── [1.5] SECRET-MODE TOGGLE — explicit, user-driven, checked BEFORE triage. ──
    # Brainstorm/secret mode is OFF by default and only flips on an explicit command.
    # A toggle is a CONTROL command, not a query: set the session flag, confirm, and
    # return — nothing else runs. This is the only thing that turns the local-only
    # clamp on; triage no longer auto-forces it for brainstorm.
    toggle = detect_toggle(message)
    if toggle is not None:
        set_secret_mode(session_id, toggle == "on")
        confirm = ("Secret mode on, Sir. Local only." if toggle == "on"
                   else "Secret mode off, Sir.")
        _log(trace_id, "secret_mode", f"toggle -> {toggle} (session={session_id})")
        if voice_flag:
            try:
                from voice.speaker import speak
                speak(confirm, "neutral")
            except Exception as e:
                _log(trace_id, "voice", f"failed: {e}")
        return {
            "response": confirm,
            "mood": "neutral",
            "provider_used": "none",
            "domain": "control",
            "sensitivity": "secret" if toggle == "on" else "public",
            "trace_id": trace_id,
            "file_path": None,
        }

    # ── [2] TRIAGE — sensitivity / complexity / domain / loop_worthy ──
    verdict = triage(message)  # fail-safe → private+complex defaults live in triage
    ctx.update(verdict)
    _log(trace_id, "triage", str(verdict))

    # ── SECRET-MODE OVERRIDE — explicit session posture beats triage routing. ──
    # When this session toggled secret mode ON, force EVERY turn local-only (secret)
    # regardless of domain. This only ever RAISES to secret, never lowers, so triage's
    # Layer-1 hard privacy rules (PAN/password/account → secret) and the retrieval
    # guard are untouched — the structural privacy layer can never be weakened by this.
    secret_mode = is_secret_mode(session_id)
    if secret_mode:
        ctx["sensitivity"] = "secret"
    _log(trace_id, "secret_mode",
         f"state={'on' if secret_mode else 'off'} -> sensitivity={ctx['sensitivity']}")

    sensitivity = ctx["sensitivity"]
    complexity = ctx["complexity"]
    domain = ctx["domain"]
    loop_worthy = ctx["loop_worthy"]

    dossier_text = None  # the detailed structured material, if a teacher produced one
    file_path = None     # a real file a teacher/export wrote, surfaced in the response

    # ── [2.5] ACTION DISPATCH — the ONLY stage that EXECUTES a tool. ──
    # Cognition only generates text, so without this an action request like
    # "remind me…" is spoken back but never written. Detect a reminder command,
    # RUN the real tool, and hand cognition the REAL result so her confirmation
    # reflects an actual DB write — never a fabricated "Done, Sir". Extraction is
    # local (ollama), so it runs regardless of sensitivity and leaks nothing.
    action = detect_action(message, session_id)
    if action is not None:
        res = run_action(action)
        ctx["action"] = action["action"]
        ctx["action_result"] = res
        # An email draft / revision / re-show is a structured block (to/subject/body)
        # she must present in full — give cognition deliverable room, not the 60-token
        # confirmation cap.
        if action.get("tool") == "email" and action["action"] in ("draft", "revise", "reconfirm"):
            ctx["deliverable"] = True
        # Hand the retrieved material's sensitivity tier to the cloud guard in
        # cognition. The guard there (not this call site) decides cloud vs local for
        # ALL tiers and tools — private AND secret force local. This stays generic:
        # any tool that returns a `tier` is covered without its own routing patch.
        if res.get("tier"):
            ctx["material_tier"] = res["tier"]
            if res["tier"] in ("private", "secret"):
                sensitivity = res["tier"]  # reflect the guarded tier in the report
        _log(trace_id, "action", f"tool={action.get('tool')} action={fn_name(action)} result={res}")
        # The [reminder]/[jobs] WRITE line prints on every successful write.
        result = await cognition_pass(ctx, action_material(action, res))
        _log(trace_id, "cognition", f"provider={result.get('provider_used')} mood={result.get('mood')}")
        if voice_flag:
            try:
                from voice.speaker import speak
                speak(result["response"], result.get("mood", "neutral"))
            except Exception as e:
                _log(trace_id, "voice", f"failed: {e}")
        return {
            "response": result["response"],
            "mood": result["mood"],
            "provider_used": result["provider_used"],
            "domain": domain,
            "sensitivity": sensitivity,
            "trace_id": trace_id,
            "file_path": None,
        }

    # ── EXPORT FOLLOW-UP OVERRIDE ──
    # "Give me a call sheet for those" triages to domain=none and would short-circuit,
    # losing the prior leadgen results. If this session has a cached leadgen set and the
    # line is an export request, route it BACK to leadgen so those cached rows become a
    # real file. Secret is left alone (handled local-only below).
    if (domain != "leadgen" and sensitivity != "secret"
            and is_export_request(message) and get_cached_leadgen(session_id)):
        _log(trace_id, "route",
             f"export follow-up — domain '{domain}' overridden to leadgen (cached set present)")
        domain = "leadgen"
        ctx["domain"] = "leadgen"

    # ── ROUTING ──
    # The domain tag decides the path, NOT complexity. If triage routed this to a real
    # teacher (leadgen, jobs, …), it needs that teacher's real data/tools — a "find
    # studios" line tagged trivial must still hit leadgen+Places, never short-circuit to
    # cognition (that path has no tools and hallucinates). So: real teacher → ALWAYS
    # route through it (any complexity); only domain 'none'/unknown short-circuits to
    # fast cognition. Secret is the one hard override — stays local-only, no teacher.
    teacher_cls = TEACHERS.get(domain)
    use_teacher = (
        sensitivity != "secret"        # secret stays local-only — no teacher, no cloud
        and teacher_cls is not None     # real domain → its teacher; 'none'/unknown → short-circuit
    )

    # ── [2.7] BRAIN — one cheap per-task plan: tools / book tier / ensemble. ──
    # Runs for BOTH paths (teacher AND general) so a casual "find me hidden gems" can
    # draw Places from the shared pool instead of hallucinating. The teacher's declared
    # tier is the brain's default + fallback. plan() is fail-soft (keyword rules) and
    # clamps itself for secret/private BEFORE any cloud call — privacy can't be routed
    # around by the selector.
    teacher = teacher_cls() if use_teacher else None
    default_tier = teacher.reasoning_tier(ctx) if teacher else None
    brain = await plan(message, domain=domain, sensitivity=sensitivity,
                       complexity=complexity, loop_worthy=loop_worthy,
                       default_tier=default_tier, known_domains=tuple(TEACHERS.keys()))
    ctx["brain"] = brain
    _log(trace_id, "brain",
         f"source={brain['source']} "
         f"prompt={'composed' if brain.get('composed_prompt') else 'static-template'} "
         f"tools={brain['tools']} tier={brain['book_tier']} "
         f"ensemble={brain['ensemble']} domains={brain['domains']} — {brain['reason']}")

    # ── [2.8] MULTI-DOMAIN FAN-OUT (Slice 8) ──
    # The brain flagged 2+ genuinely distinct teacher domains for this ONE task
    # ("plan my week and suggest a workout" → life + fitness). public-only, matching
    # every other cloud fan-out gate in this file (ensemble, tool draws) — brain.py
    # already returns domains=[] for secret/private, this is the structural backstop.
    # Single-domain routing below is completely untouched by this branch existing.
    fanout_domains = brain.get("domains", [])[:_MAX_FANOUT]
    multi_domain = sensitivity == "public" and len(fanout_domains) >= 2

    if multi_domain:
        _log(trace_id, "route",
             f"multi-domain fan-out — domains={fanout_domains} (parallel via asyncio.gather)")
        fanout_t0 = time.monotonic()
        outcomes_list = await asyncio.gather(
            *(_run_fanout_teacher(d, ctx, trace_id) for d in fanout_domains))
        _log(trace_id, "route",
             f"fan-out complete — {len(outcomes_list)} teacher(s) confirmed parallel, "
             f"wall_ms={int((time.monotonic() - fanout_t0) * 1000)}")

        # Any fanned-out domain may call for a structured block (e.g. study/work/jobs/
        # leadgen do, life/fitness/spirit/brainstorm don't) — render structured if ANY
        # source needs it.
        ctx["deliverable"] = any(o["deliverable"] for o in outcomes_list)

        # Reuse cognition's EXISTING ensemble-combine path — no new combiner. Each
        # domain contributes ONE labeled text block; _format_material wraps a list of
        # 2+ strings as "── Source A/B/C ──" and tells cognition to reconcile them —
        # here that reconciliation is ACROSS DOMAINS instead of across books.
        material_list = [
            f"[{o['domain'].upper()}]\n{o['material']}"
            for o in outcomes_list if o.get("material")
        ]
        dossier_text = "\n\n".join(material_list) if material_list else None

        domain = "+".join(fanout_domains)
        ctx["domain"] = domain
        result = await cognition_pass(ctx, material_list or None)

    elif not use_teacher:
        # Straight to cognition. For secret, select_book forces ollama via the tier in ctx.
        material = None
        if sensitivity == "secret":
            why = "secret → local-only cognition"
        else:
            # General path: the brain sized the book tier per task and may have drawn
            # pool tools; complexity heuristics stay as an upgrade-only floor. cognition
            # reads gen_tier to pick the book + token cap; privacy stays absolute
            # downstream.
            complexity = _general_complexity(message, complexity)
            ctx["complexity"] = complexity
            ctx["gen_tier"] = _brain_gen_tier(brain, complexity, loop_worthy)
            why = (f"no teacher for domain '{domain}' "
                   f"(complexity={complexity}, gen_tier={ctx['gen_tier']})")
            # ── GENERAL PATH TOOL ACCESS — the pool is not teacher-only anymore. ──
            # The brain's picks are fetched and handed to cognition as REAL material,
            # so "hidden gems in Delhi" presents Places rows instead of hallucinating.
            # brain already returns tools=[] for secret/private turns.
            if brain["tools"]:
                fetched = await fetch_from(brain["tools"], message)
                counts = ", ".join(f"{k}({len(v)})" for k, v in fetched.items())
                _log(trace_id, "pool", f"general-path tools fired: {counts}")
                block = format_pool_block(fetched)
                if block:
                    material = (
                        "REAL RESULTS from the shared tool pool (fetched live for this "
                        "turn — use ONLY these as the factual basis for any names, "
                        "places or facts you present; do not invent beyond them):\n"
                        + block)
        _log(trace_id, "route", f"short-circuit — {why}")
        result = await cognition_pass(ctx, material)

    else:
        _log(trace_id, "route", f"teacher — domain '{domain}' → {teacher_cls.__name__} (complexity={complexity})")

        # ── [3] TEACHER — raw material ──
        # Wall-clock timer spans the teacher's work + any looper refinement (the material-
        # production stage), for the passive outcomes log below. Cognition is timed apart.
        teach_t0 = time.monotonic()
        teach = await teacher.run(ctx)
        ctx["deliverable"] = teach.get("deliverable", False)  # let cognition see it
        file_path = teach.get("file_path")  # leadgen export wrote a real .xlsx, if any
        ensemble = teach.get("ensemble")  # list of raw book outputs when 2 books ran
        _log(trace_id, "teacher", f"books={teach.get('book_used')} tokens={teach.get('tokens')} "
                                  f"ensemble={len(ensemble) if ensemble else 0} "
                                  f"deliverable={ctx['deliverable']} file={file_path}")

        # ── [4/5] LOOPER — conditional quality gate ──
        # A teacher ensemble (two parallel books) IS the quality path: hand BOTH raw
        # outputs to cognition and let her reconcile them. The single-book looper is
        # bypassed here — it would collapse the two passes back to one. It still runs on
        # the non-ensemble paths (secret/private turns, or the ensemble fallback).
        if ensemble:
            material = ensemble
            # Log BOTH books comma-joined; passed if either pass cleared its self-check.
            out_book = ",".join(r.get("book_used", "?") for r in ensemble)
            out_passed = any(r.get("self_check", {}).get("passes") for r in ensemble)
            out_attempts = 1  # each book called once, in parallel — no retries
            _log(trace_id, "looper",
                 f"skipped (ensemble of {len(ensemble)} book(s) → cognition combines)")
        elif loop_worthy:
            looped = await run_looper(ctx, teacher, teach["raw_text"])
            material = looped["refined_material"]
            out_book = looped.get("book_used") or teach.get("book_used", "none")
            out_passed = looped.get("confidence") == "high"
            out_attempts = looped.get("attempts", 1)
            _log(trace_id, "looper", f"confidence={looped['confidence']} attempts={looped['attempts']}")
        else:
            material = teach
            out_book = teach.get("book_used", "none")
            out_passed = bool(teach.get("self_check", {}).get("passes"))
            out_attempts = 1
            _log(trace_id, "looper", "skipped (not loop_worthy)")

        # ── OUTCOME LOG (Slice A — passive, write-only, fail-soft) ──
        # One row per completed teacher task. No routing decision is taken from it yet;
        # it just accumulates data. log_outcome swallows its own errors, so this can
        # never break the response.
        log_outcome(
            domain=domain,
            book_used=out_book,
            ensemble=bool(ensemble),
            self_check_passed=out_passed,
            attempts=out_attempts,
            tokens=teach.get("tokens", 0),
            latency_ms=int((time.monotonic() - teach_t0) * 1000),
            complexity=complexity,
            session_id=session_id,
        )

        # Keep the detailed dossier text for a possible file export (NOT cognition's
        # trimmed chat framing).
        dossier_text = material if isinstance(material, str) else teach["raw_text"]

        # ── [6] COGNITION — her answer ──
        result = await cognition_pass(ctx, material)

    _log(trace_id, "cognition", f"provider={result.get('provider_used')} mood={result.get('mood')}")

    # ── EXPORT — on-demand file, only when Sir explicitly asks AND it's a deliverable ──
    # Skipped when a teacher already wrote a file (leadgen .xlsx) — no double export.
    low = message.lower()
    if not file_path and dossier_text and ctx.get("deliverable") and any(k in low for k in ("pdf", "csv", "export", "download")):
        fmt = "csv" if "csv" in low else "pdf"  # default pdf
        export_title = " ".join(message.split()[:6]) or domain
        try:
            if fmt == "csv":
                from tools.csv_gen import make_csv
                file_path = make_csv(export_title, dossier_text)
            else:
                from tools.pdf_gen import make_pdf
                file_path = make_pdf(export_title, dossier_text)
            # File replaces the chat dump — short confirmation in her voice, no duplication.
            result["response"] = f"Done, Sir. Full dossier saved: {os.path.basename(file_path)}."
            result["mood"] = "neutral"
            _log(trace_id, "export", f"{fmt} → {file_path}")
        except Exception as e:
            _log(trace_id, "export", f"failed: {e}")

    # Speak if requested — same path /chat uses.
    if voice_flag:
        try:
            from voice.speaker import speak
            speak(result["response"], result.get("mood", "neutral"))
        except Exception as e:
            _log(trace_id, "voice", f"failed: {e}")

    return {
        "response": result["response"],
        "mood": result["mood"],
        "provider_used": result["provider_used"],
        "domain": domain,
        "sensitivity": sensitivity,
        "trace_id": trace_id,
        "file_path": file_path,
    }
