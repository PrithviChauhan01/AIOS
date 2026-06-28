from agents.base import Teacher

# ── Refusal / empty signals — a book that punts is a self-check failure ──
_REFUSALS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "as an ai",
    "i don't have access", "i do not have access", "i'm sorry, but",
    "unable to provide", "i'm not able",
)

# ── A real job shortlist touches at least some of these ──
_RESEARCH_SIGNALS = (
    "company", "role", "fit", "resume", "source", "engineering", "design",
    "position", "title", "apply", "hiring", "url",
)


class JobsTeacher(Teacher):
    """Domain expert for the job hunt (income track). Given a role query + any
    filters (company type, location, stage), it produces a structured RAW shortlist
    of roles — company, role, why-fit, which resume to use, source/url — for
    cognition to present. Raw material only: never the cover letter, never
    personality, never a sign-off. Resume routing (engineering vs design) is decided
    per role from the role/JD and recorded in the output."""

    domain = "jobs"
    memory_ns = "mem_jobs"
    deliverable = True  # a role shortlist is inherently a structured block

    def required_capability(self, ctx):
        return {"reasoning": "good"}

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        query = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown job-hunt context (Sir's background, target roles, "
                "preferences, both resumes):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        return f"""You are a job-search specialist. Produce a focused, factual SHORTLIST of roles \
matching Sir's query below. Output RAW structured material only — no greeting, no opinion in your \
own voice, no cover-letter prose, no sign-off.

QUERY (role + any filters such as company type, location, or stage/seniority):
{query}{memory_block}

Honor every filter Sir gave (company_type, location, stage). Return 3–6 candidate roles. For EACH role, \
use exactly these fields:

ROLE N
COMPANY: the employer.
ROLE: the job title.
WHY-FIT: 1–2 concrete sentences on why this fits Sir — match to his background/target, what makes it a good shot.
RESUME: which resume to send — exactly "engineering" or "design". Decide from the role/JD:
  - engineering → software / SWE / backend / frontend / full-stack / data / ML / infra / devops / platform / embedded.
  - design → product design / UX / UI / visual / brand / creative / design-engineer-leaning.
  When a role straddles both (e.g. "design engineer"), pick the side the day-to-day work leans toward and say so in WHY-FIT.
SOURCE/URL: where to find/apply (careers page, board, or "unknown" — never invent a URL).

After the roles, add:
LIVE DATA NOTE: state plainly that these are reasoned candidates from role knowledge, NOT live postings — \
real-time listings (job-board/API search) would plug in here to confirm openings and fill exact URLs.

Be specific and concise. Substance only. If a fact is unknown, write "unknown" — do not invent it."""

    def self_check(self, book_output: str, ctx: dict) -> dict:
        text = (book_output or "").strip()
        low = text.lower()

        if len(text) < 60:
            return {"passes": False, "feedback": "output too thin to be a real shortlist"}
        if any(r in low for r in _REFUSALS):
            return {"passes": False, "feedback": "book refused / disclaimed instead of shortlisting"}
        if not any(s in low for s in _RESEARCH_SIGNALS):
            return {"passes": False, "feedback": "no recognizable job-shortlist substance (company/role/fit/resume)"}
        if "engineering" not in low and "design" not in low:
            return {"passes": False, "feedback": "no resume routing (engineering/design) on any role"}

        return {"passes": True, "feedback": "on-topic role shortlist with resume routing"}
