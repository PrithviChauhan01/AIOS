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

    def _resume_block(self) -> str:
        """Build the resume-routing context from Sir's stored resumes. Only the
        variant + derived skills go into the prompt (no raw resume content → no PII
        to the cloud book). Fail-soft: a store error just yields no block."""
        try:
            from tools.jobs import resume_profiles
            profiles = resume_profiles()
        except Exception as e:
            print(f"[jobs] resume profile load failed: {e}")
            return ""

        if not profiles:
            print("[jobs] resume context: none on file")
            return ("\n\nRESUMES ON FILE: NONE. Sir has not saved a resume yet — do NOT "
                    "guess a resume per role; set RESUME to \"none on file\" and add one "
                    "line telling him to save an engineering/design resume so you can match.")

        print("[jobs] resume context: "
              + "; ".join(f"{p['variant']}({len(p['skills'])} skills)" for p in profiles))
        lines = "\n".join(
            f"- {p['variant'].upper()} resume ('{p['name']}') — skills: "
            f"{', '.join(p['skills']) or 'unknown'}"
            for p in profiles
        )
        return ("\n\nSir's RESUMES ON FILE (route each role to the one whose REAL skills "
                "fit best — grounded in his actual resumes, do not invent):\n" + lines)

    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        query = ctx.get("message") or ctx.get("query", "")
        memory_block = ""
        if domain_memory:
            memory_block = (
                "\n\nKnown job-hunt context (Sir's background, target roles, "
                "preferences, both resumes):\n"
                + "\n".join(f"- {m}" for m in domain_memory)
            )

        # Resume routing is grounded in Sir's ACTUAL stored resumes (Slice E). We
        # inject only the variant + derived skills — never raw resume content, so no
        # PII reaches the cloud book. If none are on file we say so and forbid a guess.
        resume_block = self._resume_block()

        return f"""You are a job-search specialist. Produce a focused, factual SHORTLIST of roles \
matching Sir's query below. Output RAW structured material only — no greeting, no opinion in your \
own voice, no cover-letter prose, no sign-off.

QUERY (role + any filters such as company type, location, or stage/seniority):
{query}{memory_block}{resume_block}

Honor every filter Sir gave (company_type, location, stage). Return 3–6 candidate roles. For EACH role, \
use exactly these fields:

ROLE N
COMPANY: the employer.
ROLE: the job title.
WHY-FIT: 1–2 concrete sentences on why this fits Sir — match to his background/target, what makes it a good shot.
RESUME: which of Sir's resumes ON FILE (above) to send — match the role to the resume whose REAL skills fit best, \
and name a matching skill or two in WHY-FIT. If a resume is on file, the value must be exactly "engineering" or \
"design". If NO resume is on file, set this to "none on file" and do NOT guess.
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
