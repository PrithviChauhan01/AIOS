from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real fitness brief touches at least some of these ──
_RESEARCH_SIGNALS = (
    "exercise", "rep", "set", "workout", "muscle", "training", "mobility",
    "recovery", "form", "warm", "cardio", "strength", "movement", "intensity",
)


class FitnessTeacher(Teacher):
    """Domain expert for fitness and training. Produces structured raw material on a
    workout, movement, or training question — exercises, programming, form cues —
    for cognition to deliver. Raw material only: never the pep talk, never a sign-off."""

    domain = "fitness"
    memory_ns = "mem_fitness"
    deliverable = False

    def required_capability(self, ctx):
        return {"reasoning": "good"}

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        topic = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown fitness context (Sir's goals, injuries, equipment, training history):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a fitness analyst. Produce factual, structured training material on the topic below. \
Output RAW structured material only — no greeting, no motivational framing, no opinion in your own voice, no sign-off.

TOPIC:
{topic}{memory_block}

Return the following sections. If something is unknown, write "unknown" — do not invent it.

OVERVIEW: what this addresses and who it suits.
MOVEMENTS: the key exercises or drills, with the muscles / systems they target.
PROGRAMMING: sets, reps, frequency, intensity, progression where relevant.
FORM CUES: technique points and the common mistakes to avoid.
NOTES: recovery, contraindications, or anything else materially useful.

Be specific and concise. Substance only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real training material"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of advising"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable fitness substance (movements/programming/form)"}

        return {"passes": True, "feedback": "on-topic training material with substance"}
