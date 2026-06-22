from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real research dossier touches at least some of these ──
_RESEARCH_SIGNALS = (
    "service", "contact", "email", "phone", "website", "instagram",
    "social", "founder", "owner", "studio", "company", "address",
    "fit", "pricing", "portfolio", "client",
)


class LeadgenTeacher(Teacher):
    """Domain expert for lead research. Pulls structured RAW intel on a studio /
    company — services, contact, social, fit signals — for cognition to qualify
    and act on. Never writes the outreach, never adds personality."""

    domain = "leadgen"
    memory_ns = "mem_leadgen"

    def required_capability(self, ctx):
        return {"reasoning": "good"}

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        target = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown leadgen context (ICP, past leads, what Sir cares about):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a lead-research analyst. Produce a factual research dossier on the target below. \
Output RAW structured research only — no greeting, no opinion of your own voice, no outreach copy, no sign-off.

TARGET:
{target}{memory_block}

Return the following sections. If a fact is unknown, write "unknown" — do not invent it.

SERVICES: what they offer / sell.
CONTACT: email, phone, website, physical location.
SOCIAL: instagram / linkedin / other handles and follower scale if known.
FIT SIGNALS: concrete evidence for or against this being a good lead — size, activity, recent posts, gaps a service could fill, budget cues.
NOTES: anything else materially useful for qualifying this lead.

Be specific and concise. Facts only."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be real research"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of researching"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable research substance (services/contact/social/fit)"}

        return {"passes": True, "feedback": "on-topic research with substance"}
