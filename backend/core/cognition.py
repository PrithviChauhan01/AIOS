import asyncio

from core.router import SYSTEM_PROMPT, _extract_mood
from core.books import select_book, select_fast_book, call_book, model_for
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

    # ── CLOUD GUARD (covers ALL retrieval: documents, memory, future tools) ──
    # Any RETRIEVED material injected this turn carries its sensitivity in
    # ctx["material_tier"]. Per the LLD the cloud guard must filter BOTH private AND
    # secret: retrieved private/secret content must NEVER reach a cloud book. So we
    # force the WHOLE pass local (ollama) when it fires — book selection AND the
    # fact extractor (which is itself a cloud call on the response). This is the one
    # guard layer; tools only tag ctx["material_tier"], they don't re-route.
    base_tier = ctx.get("sensitivity", "public")
    material_tier = ctx.get("material_tier")
    local_only = base_tier == "secret" or material_tier in ("private", "secret")
    if material_tier in ("private", "secret"):
        print(f"[guard] retrieved tier={material_tier} -> local-only")
    # 'secret' is the system's local-only selector (ollama for books, vault for the
    # extractor, never cloud). Route the whole turn through it when the guard fires.
    route_tier = "secret" if local_only else base_tier

    # Hard cap on output length, independent of what the model wants to do. A
    # deliverable — or a reminders/jobs list, which may run several rows — earns room
    # for structure; everything else (small-talk, trivial, short-circuit, a one-
    # line action confirmation) is held to a couple of sentences in her voice.
    # 400 truncated multi-row shortlists mid-sentence; 900 lets a full 3–6 role
    # block (each with several fields) finish cleanly.
    roomy = ctx.get("deliverable") or ctx.get("action") == "list"
    max_tokens = 900 if roomy else 60

    # Fast lane: trivial / short-circuit small-talk (no deliverable) needs her voice,
    # not reasoning horsepower — route it to Groq's 8B (sub-second). The few-shot voice
    # examples in the prompt do the heavy lifting; 8B follows them. Teacher/deliverable
    # paths keep the full 70B where reasoning actually earns its latency.
    fast_lane = ctx.get("complexity") == "trivial" and not ctx.get("deliverable")
    books = select_fast_book(route_tier) if fast_lane else select_book(_GEN_SPEC, route_tier)

    result = None
    for book in books:
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

    # Surface fast vs full so latency wins are visible per turn.
    print(f"[cognition] lane={'fast' if fast_lane else 'full'} "
          f"book={result['book_used']} model={model_for(result['book_used'])} "
          f"tokens={result['tokens']}")

    response, mood = _extract_mood(result["raw_text"])

    # Persist the turn + mine durable facts — same path /chat uses, off the loop.
    message = ctx.get("message") or ctx.get("query", "")
    session_id = ctx.get("session_id", "default")
    # route_tier carries the guard: private/secret → 'secret' → extractor vaults the
    # turn locally instead of sending the (private) response to the cloud extractor.
    await asyncio.to_thread(save_message, session_id, "user", message)
    await asyncio.to_thread(save_message, session_id, "assistant", response)
    await asyncio.to_thread(extract_and_store, message, response, route_tier)

    return {
        "response": response,
        "mood": mood,
        "provider_used": result["book_used"],
        "tokens": result["tokens"],
    }
