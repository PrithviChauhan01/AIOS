from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── Real reflective material touches at least some of these ──
_RESEARCH_SIGNALS = (
    "reflect", "journal", "meditat", "mood", "feeling", "intention",
    "practice", "breath", "mindful", "gratitude", "theme", "observ",
)


class SpiritTeacher(Teacher):
    """Domain expert for Sir's OWN inner work — journaling, meditation, mood logging,
    personal reflection. Produces structured raw material to support that reflection:
    themes, prompts, practices. Raw material only: never the homily, never a sign-off."""

    domain = "spirit"
    memory_ns = "mem_spirit"
    deliverable = False

    def required_capability(self, ctx):
        return {"reasoning": "good"}

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        topic = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown reflective context (Sir's recurring themes, prior entries, what he's working through):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a reflection analyst supporting Sir's own inner work. Produce structured raw material \
to ground his journaling, meditation, or reflection on what's below. \
Output RAW structured material only — no greeting, no preaching, no opinion in your own voice, no sign-off.

ENTRY:
{topic}{memory_block}

Return the following sections. Stay grounded in what Sir actually wrote — do not invent feelings or events.

THEMES: the underlying patterns or concerns present in this entry.
REFLECTION PROMPTS: open questions that would help Sir go deeper.
PRACTICES: concrete meditation, journaling, or grounding practices that fit this moment.
OBSERVATIONS: anything notable to surface gently, without judgment.
NOTES: anything else materially useful for this reflection.

Be specific and concise. Substance only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real reflective material"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of reflecting"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable reflective substance (themes/prompts/practices)"}

        return {"passes": True, "feedback": "on-topic reflective material with substance"}
