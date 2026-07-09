from agents.base import Teacher
from tools.registry import fetch_from

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real study dossier touches at least some of these ──
_RESEARCH_SIGNALS = (
    "overview", "concept", "definition", "key", "example", "context",
    "history", "principle", "summary", "fact", "background", "term",
)


class StudyTeacher(Teacher):
    """Domain expert for study research. Draws the shared 'wikipedia' tool for real
    facts BEFORE asking a book to assemble them into a structured study dossier.
    First consumer of the shared tool pool. Never adds personality, never the answer."""

    domain = "study"
    memory_ns = "mem_study"
    deliverable = True  # study research is a structured dossier too

    def reasoning_tier(self, ctx):
        return "strong"  # real research/reasoning — Nemotron Super, not Groq

    async def run(self, ctx: dict) -> dict:
        # Pull real, verified facts from the shared pool before the book runs.
        topic = ctx.get("message") or ctx.get("query", "")
        fetched = await fetch_from(["wikipedia"], topic)
        ctx["_verified_facts"] = "\n".join(
            f"- {r}" for results in fetched.values() for r in results
        )
        return await super().run(ctx)

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        topic = ctx.get("message") or ctx.get("query", "")

        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown study context (what Sir is working on, prior notes):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        verified_block = ""
        facts = ctx.get("_verified_facts", "")
        if facts:
            verified_block = (
                "\n\nUse ONLY these verified facts where available; mark anything not "
                "covered as 'unknown' — do NOT invent.\nVERIFIED FACTS:\n" + facts
            )

        return f"""You are a study-research analyst. Produce a factual study dossier on the topic below. \
Output RAW structured research only — no greeting, no opinion of your own voice, no chat framing, no sign-off.

TOPIC:
{topic}{memory_block}{verified_block}

Return the following sections. If a fact is unknown, write "unknown" — do not invent it.

OVERVIEW: what this is, in plain terms.
KEY CONCEPTS: the core ideas / terms worth knowing.
CONTEXT: history, background, or where this fits.
EXAMPLES: concrete illustrations if any.
NOTES: anything else materially useful to study this.

Be specific and concise. Facts only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real research"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of researching"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable study substance (overview/concepts/context)"}

        return {"passes": True, "feedback": "on-topic research with substance"}
