from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real work dossier touches at least some of these ──
_RESEARCH_SIGNALS = (
    "objective", "task", "approach", "step", "deliverable", "stakeholder",
    "deadline", "priority", "risk", "plan", "scope", "outcome", "milestone",
)


class WorkTeacher(Teacher):
    """Domain expert for professional work. Produces a structured working brief on a
    task or project — objective, approach, steps, risks — for cognition to act on.
    Raw material only: never the final memo, never personality, never a sign-off."""

    domain = "work"
    memory_ns = "mem_work"
    deliverable = True  # a working brief is inherently a structured block

    def required_capability(self, ctx):
        return {"reasoning": "good"}

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        task = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown work context (Sir's role, ongoing projects, what he cares about):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a work analyst. Produce a factual working brief on the task below. \
Output RAW structured material only — no greeting, no opinion in your own voice, no memo prose, no sign-off.

TASK:
{task}{memory_block}

Return the following sections. If something is unknown, write "unknown" — do not invent it.

OBJECTIVE: what success on this task actually looks like.
KEY POINTS: the facts, constraints, and considerations that matter.
APPROACH: the concrete steps or sequence to get it done.
RISKS: what could go wrong, blockers, dependencies.
NOTES: anything else materially useful to execute this.

Be specific and concise. Substance only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be a real brief"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of working"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable work substance (objective/approach/steps/risks)"}

        return {"passes": True, "feedback": "on-topic working brief with substance"}
