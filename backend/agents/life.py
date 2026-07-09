from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real life-admin brief touches at least some of these ──
_RESEARCH_SIGNALS = (
    "schedule", "task", "habit", "reminder", "routine", "appointment",
    "deadline", "plan", "priority", "calendar", "daily", "weekly",
)


class LifeTeacher(Teacher):
    """Domain expert for life admin — schedule, habits, reminders, routines. Produces
    structured raw material on organizing Sir's day or week, for cognition to deliver.
    Raw material only: never the nagging, never personality, never a sign-off."""

    domain = "life"
    memory_ns = "mem_life"
    deliverable = False

    def reasoning_tier(self, ctx):
        return "fast"  # schedule/habits/reminders planning — Groq is enough

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        request = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown life context (Sir's routines, standing commitments, habits he's building):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a life-admin analyst. Produce a factual, structured plan for the request below. \
Output RAW structured material only — no greeting, no opinion in your own voice, no chit-chat, no sign-off.

REQUEST:
{request}{memory_block}

Return the following sections. If something is unknown, write "unknown" — do not invent it.

OVERVIEW: what Sir is trying to organize or get done.
SCHEDULE: the tasks, appointments, or blocks and where they sit in time.
HABITS: any recurring routines or behaviors relevant here.
REMINDERS: what needs a nudge and when.
NOTES: priorities, conflicts, or anything else materially useful.

Be specific and concise. Substance only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be a real plan"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of planning"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable life-admin substance (schedule/habits/reminders)"}

        return {"passes": True, "feedback": "on-topic life-admin plan with substance"}
