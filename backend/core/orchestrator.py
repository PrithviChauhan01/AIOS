import os
import uuid

from core.triage import triage
from core.memory import get_history
from core.cognition import cognition_pass
from core.looper import run_looper
from core.action_dispatch import detect_action, run_action, action_material, fn_name
from agents.leadgen import LeadgenTeacher
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

    # ── [2] TRIAGE — sensitivity / complexity / domain / loop_worthy ──
    verdict = triage(message)  # fail-safe → private+complex defaults live in triage
    ctx.update(verdict)
    _log(trace_id, "triage", str(verdict))

    sensitivity = ctx["sensitivity"]
    complexity = ctx["complexity"]
    domain = ctx["domain"]
    loop_worthy = ctx["loop_worthy"]

    dossier_text = None  # the detailed structured material, if a teacher produced one

    # ── [2.5] ACTION DISPATCH — the ONLY stage that EXECUTES a tool. ──
    # Cognition only generates text, so without this an action request like
    # "remind me…" is spoken back but never written. Detect a reminder command,
    # RUN the real tool, and hand cognition the REAL result so her confirmation
    # reflects an actual DB write — never a fabricated "Done, Sir". Extraction is
    # local (ollama), so it runs regardless of sensitivity and leaks nothing.
    action = detect_action(message)
    if action is not None:
        res = run_action(action)
        ctx["action"] = action["action"]
        ctx["action_result"] = res
        # PRIVACY GUARD: if the result is secret-tier (e.g. a vault document pulled
        # up), force this turn local — cognition then runs on ollama and the
        # decrypted content can never reach a cloud provider. Reuses the same tier
        # switch the rest of the system routes on.
        if res.get("tier") == "secret":
            ctx["sensitivity"] = "secret"
            sensitivity = "secret"
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

    # ── ROUTING ──
    # A teacher (plus its book + looper) costs 2-3 extra LLM calls (~5s). It only earns
    # that when the line genuinely needs domain knowledge or a deliverable. Trivial
    # small-talk never does — even when triage tags it domain:life/work/etc. — so it
    # short-circuits straight to her (one call, ~1.5s). The domain tag must NOT force a
    # teacher path on a trivial line.
    teacher_cls = TEACHERS.get(domain)
    use_teacher = (
        sensitivity != "secret"        # secret stays local-only — no teacher, no cloud
        and complexity != "trivial"     # trivial small-talk: one call, never a teacher
        and teacher_cls is not None     # 'none'/unknown domains have no teacher anyway
    )

    if not use_teacher:
        # Straight to cognition. For secret, select_book forces ollama via the tier in ctx.
        if sensitivity == "secret":
            why = "secret → local-only cognition"
        elif complexity == "trivial":
            why = f"trivial (domain '{domain}' ignored)"
        else:
            why = f"no teacher for domain '{domain}'"
        _log(trace_id, "route", f"short-circuit — {why}")
        result = await cognition_pass(ctx)

    else:
        teacher = teacher_cls()
        _log(trace_id, "route", f"teacher — domain '{domain}' → {teacher_cls.__name__} (complexity={complexity})")

        # ── [3] TEACHER — raw material ──
        teach = await teacher.run(ctx)
        ctx["deliverable"] = teach.get("deliverable", False)  # let cognition see it
        _log(trace_id, "teacher", f"book={teach.get('book_used')} tokens={teach.get('tokens')} deliverable={ctx['deliverable']}")

        # ── [4/5] LOOPER — conditional quality gate ──
        if loop_worthy:
            looped = await run_looper(ctx, teacher, teach["raw_text"])
            material = looped["refined_material"]
            _log(trace_id, "looper", f"confidence={looped['confidence']} attempts={looped['attempts']}")
        else:
            material = teach
            _log(trace_id, "looper", "skipped (not loop_worthy)")

        # Keep the detailed dossier text for a possible file export (NOT cognition's
        # trimmed chat framing).
        dossier_text = material if isinstance(material, str) else teach["raw_text"]

        # ── [6] COGNITION — her answer ──
        result = await cognition_pass(ctx, material)

    _log(trace_id, "cognition", f"provider={result.get('provider_used')} mood={result.get('mood')}")

    # ── EXPORT — on-demand file, only when Sir explicitly asks AND it's a deliverable ──
    file_path = None
    low = message.lower()
    if dossier_text and ctx.get("deliverable") and any(k in low for k in ("pdf", "csv", "export", "download")):
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
