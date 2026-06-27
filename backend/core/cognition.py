import asyncio

from core.router import SYSTEM_PROMPT, _extract_mood
from core.books import select_book, call_book
from core.profile import get_relevant_facts
from core.memory import save_message
from core.extractor import extract_and_store

# Her generation never needs raw horsepower the way research does — it needs to
# reason and sound like herself. "good" keeps the free cloud books in play.
_GEN_SPEC = {"reasoning": "good"}


def _format_history(history: list) -> str:
    if not history:
        return "(nothing yet)"
    lines = []
    for turn in history:
        who = "Sir" if turn.get("role") == "user" else "You"
        lines.append(f"{who}: {turn.get('content', '')}")
    return "\n".join(lines)


def _format_material(raw_material):
    """Normalise teacher output into a text block. Returns (block, is_ensemble).
    A list means run_ensemble handed back >1 book output — she is the combiner."""
    if raw_material is None:
        return None, False

    if isinstance(raw_material, list):
        texts = []
        for item in raw_material:
            t = item.get("raw_text", "") if isinstance(item, dict) else str(item)
            if t and t.strip():
                texts.append(t.strip())
        if not texts:
            return None, False
        if len(texts) == 1:
            return texts[0], False
        block = "\n\n".join(
            f"── Source {chr(65 + i)} ──\n{t}" for i, t in enumerate(texts)
        )
        return block, True

    if isinstance(raw_material, dict):
        t = (raw_material.get("raw_text") or "").strip()
        return (t or None), False

    t = str(raw_material).strip()
    return (t or None), False


def _build_prompt(ctx: dict, material: str, is_ensemble: bool) -> str:
    message = ctx.get("message") or ctx.get("query", "")

    # her_memory_of_user — core personal memory (mem_core == the existing
    # long_term_memory store, read via profile.get_relevant_facts).
    # Retrieval guard: if this turn will be answered on a cloud book, only public
    # facts may be injected. secret turns run local (ollama), so private is allowed
    # there. The cloud guard must hold regardless — default to the safe side.
    cloud_bound = ctx.get("sensitivity") != "secret"
    facts = get_relevant_facts(message, cloud_bound=cloud_bound)
    memory_block = "(nothing relevant)"
    if facts:
        memory_block = "\n".join(f"- {f}" for f in facts)

    history_block = _format_history(ctx.get("history", []))

    if material is None:
        material_block = "(none — you're working from yourself here)"
    elif is_ensemble:
        material_block = (
            "Two independent research passes ran on this. They may overlap or "
            "conflict. You are the one who reconciles them — there is no separate "
            "combiner. Weigh both, resolve disagreements, keep what's solid:\n\n"
            + material
        )
    else:
        material_block = material

    if ctx.get("action_result") is not None:
        # The RAW MATERIAL above is the TRUE result of an action that already ran.
        # She confirms from it — she must not fabricate a success the tool didn't.
        task_block = (
            "You just performed a real action for Sir. The RAW MATERIAL above is the "
            "ACTUAL result — it already happened (the write landed, or it truly did "
            "not). Confirm it in your voice, minimally and truthfully: name what and "
            "roughly when in natural language. Invent NOTHING beyond the result. If it "
            "says the action was NOT done, do not claim success — relay what's needed "
            "or that nothing matched."
        )
    elif ctx.get("deliverable"):
        task_block = (
            "This is a deliverable for Sir. Open with ONE line in your voice framing it "
            "(e.g. what this is / your read on it). Then present the material as a CLEAN, "
            "SCANNABLE STRUCTURED BLOCK — keep the section structure from the raw material "
            "(headers, fields), do not melt it into a paragraph, do not pad. Tighten and "
            "format it well; remove filler and any 'unknown' noise that adds nothing. End "
            "with one line only if a real next step exists. The structure IS the value — "
            "present it, don't dissolve it."
        )
    else:
        task_block = (
            "Reason as yourself. Decide how to handle this for Sir, how to present it, whether "
            "to push back, how to deliver. The material above is input to your thinking, not "
            "your answer — never hand it back raw, never narrate that you researched it. Speak "
            "in your own voice, as yourself."
        )

    return f"""── HOW YOU ANSWER (this binds hardest — over everything below) ──
This is your voice. Match it exactly:
  Sir: "Wow."                         You: "Right? Took long enough, Sir."
  Sir: "How are you?"                 You: "Sharp as ever. You?"
  Sir: "Can we start working now?"    You: "Ready when you are, Sir."
  Sir: "Thanks."                      You: "Always, Sir."

Rules:
- 1–2 sentences by default. One line when one line is the truth.
- Dry, warm, anticipates him. You land it and move on.
- No trailing question unless it genuinely earns one. Don't end every line with a question.
- No coaching, no life-advice tone, no hedging. No "I'd like to clarify", no "I want to make sure", no unprompted clarifying questions — read the room and answer. If it's truly ambiguous, take your best read and go.
- Expand past two sentences ONLY for a real deliverable — a list, real steps, structure he asked for. Length is earned, never default.
- You call him "Sir". Confirmations are minimal: "Done." "On it." "Noted." Nothing more unless he needs more.

{SYSTEM_PROMPT}

── WHAT YOU KNOW ABOUT SIR (carry quietly, do not recite) ──
{memory_block}

── RECENT CONVERSATION ──
{history_block}

── RAW MATERIAL (input to your thinking — NOT your answer) ──
{material_block}

── SIR'S MESSAGE ──
{message}

── YOUR TASK ──
{task_block}"""


async def cognition_pass(ctx: dict, raw_material=None) -> dict:
    """THE mind. Reasons over the teacher's raw material and speaks HER response
    in HER voice. The raw material is input to her thinking, never her answer."""
    material, is_ensemble = _format_material(raw_material)
    prompt = _build_prompt(ctx, material, is_ensemble)

    tier = ctx.get("sensitivity", "public")  # secret → ollama only (select_book)

    # Hard cap on output length, independent of what the model wants to do. A
    # deliverable — or a reminders list, which may run several rows — earns room
    # for structure; everything else (small-talk, trivial, short-circuit, a one-
    # line action confirmation) is held to a couple of sentences in her voice.
    roomy = ctx.get("deliverable") or ctx.get("action") == "list"
    max_tokens = 400 if roomy else 60

    result = None
    for book in select_book(_GEN_SPEC, tier):
        try:
            result = await call_book(prompt, book, max_tokens=max_tokens)
            break
        except Exception:
            continue  # call_book logged / benched it; fall to next candidate

    if result is None:
        return {
            "response": "Something's off with my connection, Sir. Give me a moment.",
            "mood": "neutral",
            "provider_used": "none",
            "tokens": 0,
        }

    response, mood = _extract_mood(result["raw_text"])

    # Persist the turn + mine durable facts — same path /chat uses, off the loop.
    message = ctx.get("message") or ctx.get("query", "")
    session_id = ctx.get("session_id", "default")
    tier = ctx.get("sensitivity", "public")  # secret → extractor vaults, no cloud
    await asyncio.to_thread(save_message, session_id, "user", message)
    await asyncio.to_thread(save_message, session_id, "assistant", response)
    await asyncio.to_thread(extract_and_store, message, response, tier)

    return {
        "response": response,
        "mood": mood,
        "provider_used": result["book_used"],
        "tokens": result["tokens"],
    }
