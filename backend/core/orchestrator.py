import os
import uuid

from core.triage import triage
from core.memory import get_history
from core.cognition import cognition_pass
from core.looper import run_looper
from agents.leadgen import LeadgenTeacher
from agents.study import StudyTeacher
from agents.work import WorkTeacher
from agents.fitness import FitnessTeacher
from agents.spirit import SpiritTeacher
from agents.life import LifeTeacher
from agents.brainstorm import BrainstormTeacher

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

    # ── ROUTING ──
    if sensitivity == "secret":
        # Local-only: cognition runs on ollama (forced by tier in select_book).
        # No teacher, no cloud, nothing leaves the machine.
        _log(trace_id, "route", "secret → local-only cognition")
        result = await cognition_pass(ctx)

    elif complexity in ("trivial", "simple") and domain == "none":
        # Nothing to research — straight to her, on a fast book.
        _log(trace_id, "route", "trivial/simple + no domain → short-circuit cognition")
        result = await cognition_pass(ctx)

    else:
        teacher_cls = TEACHERS.get(domain)
        if teacher_cls is None:
            _log(trace_id, "route", f"domain '{domain}' has no teacher → short-circuit cognition")
            result = await cognition_pass(ctx)
        else:
            teacher = teacher_cls()
            _log(trace_id, "route", f"domain '{domain}' → {teacher_cls.__name__}")

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
