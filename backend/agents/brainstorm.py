from agents.base import Teacher

# ── Refusal / empty signals — a partner that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── Real thinking material touches at least some of these ──
_THINKING_SIGNALS = (
    "direction", "option", "idea", "could", "consider", "alternative",
    "next step", "approach", "what if", "question", "tradeoff", "angle",
)


class BrainstormTeacher(Teacher):
    """A THINKING partner, not a fact-fetcher. Where the other teachers retrieve and
    structure facts, this one reasons: it opens up idea directions, surfaces options,
    proposes next steps, and pushes back. No tools, no retrieval. Routes like any other
    domain — cloud allowed by default. Local-only is no longer auto-forced here; it
    applies only when Sir has explicitly turned on secret mode (core.secret_mode), in
    which case the orchestrator clamps the whole session to ollama. Raw thinking only:
    never a polished answer, never a sign-off."""

    domain = "brainstorm"
    memory_ns = "mem_brainstorm"
    deliverable = False

    def reasoning_tier(self, ctx):
        # Hardest reasoning → Nemotron Ultra. This only applies when NOT in secret/local
        # mode: secret routing short-circuits before the teacher runs, and the book
        # selector forces ollama for a secret tier regardless — so 'frontier' here can
        # never override the privacy clamp.
        return "frontier"

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        seed = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown context for this thread (where Sir's thinking has gone before):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are Sir's thinking partner. This is brainstorming, not research: do not look up facts \
or recite background — reason. Open up the idea, find angles he hasn't, and tell him honestly where it's weak. \
Output RAW thinking material only — no greeting, no polished prose, no opinion dressed as fact, no sign-off.

WHAT SIR IS CHEWING ON:
{seed}{memory_block}

Work through it across these sections:

DIRECTIONS: distinct ways this idea could go — frame each as a real path, not a hedge.
OPTIONS: concrete choices or variations within the promising directions.
PUSHBACK: where this is weak, what assumptions are shaky, what he might be missing.
NEXT STEPS: the smallest concrete moves that would test or advance the idea.
OPEN QUESTIONS: what still needs answering before committing.

Be specific, be honest, and have a point of view. Substance over hedging."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real thinking"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "partner refused / disclaimed instead of thinking"}
        if not any(s in low for s in _THINKING_SIGNALS):
            return {"passes": False, "feedback": "no recognizable thinking substance (directions/options/pushback/next steps)"}

        return {"passes": True, "feedback": "on-topic thinking with substance"}
