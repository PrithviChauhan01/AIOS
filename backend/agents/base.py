import asyncio
from abc import ABC, abstractmethod
from db.chroma_init import get_chroma_client
from core.books import (
    select_book_for_tier, select_ensemble_books, upgrade_tier, call_book,
)
from tools.registry import fetch_from, format_pool_block

_BRAIN_TIERS = ("fast", "strong", "frontier")


class Teacher(ABC):
    """Domain sub-agent (LLD §3). A Teacher is a domain expert that retrieves its
    own long-term memory, frames a domain-specific prompt, picks the right book
    (model) for the job and pulls RAW material out of it. It returns raw material
    only — never a user-facing answer. Personality, synthesis and final delivery
    are cognition's job, not the Teacher's."""

    domain: str = "general"
    memory_ns: str = "long_term_memory"
    deliverable: bool = False  # True → cognition presents as a structured block, not prose

    # Pool tools this teacher fetches ITSELF in a specialized way (structured rows,
    # augmented queries — e.g. leadgen's Places dicts, jobs' "<query> job openings"
    # search). The generic brain-driven fetch skips these names so no tool fires twice
    # for one task. Everything is still drawn from the ONE shared pool either way.
    owns_tools: tuple = ()

    # ── Subclass contract ──
    @abstractmethod
    def reasoning_tier(self, ctx: dict) -> str:
        """The reasoning tier this teacher wants from the book pool — one of
        'fast' | 'strong' | 'frontier' (see core.books). This is the teacher's DECLARED
        FLOOR; book selection maps it to a model and triage complexity may only raise it,
        never lower it below this. Replaces the old capability_spec — teachers no longer
        depend on triage's complexity guess to reach Nemotron."""

    @abstractmethod
    def build_book_prompt(self, ctx: dict, domain_memory: list) -> str:
        """Domain-expert prompt that asks the book for structured RAW material."""

    @abstractmethod
    def self_check(self, book_output: str, ctx: dict) -> dict:
        """Cheap sanity gate on raw material. Returns {passes: bool, feedback: str}."""

    # ── Shared machinery ──
    def retrieve_memory(self, query: str, n_results: int = 5) -> list:
        """Semantic recall against this Teacher's own Chroma namespace."""
        client = get_chroma_client()
        collection = client.get_or_create_collection(self.memory_ns)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        return results["documents"][0] if results["documents"] else []

    async def run(self, ctx: dict) -> dict:
        query = ctx.get("message") or ctx.get("query", "")
        domain_memory = self.retrieve_memory(query)

        declared_tier = self.reasoning_tier(ctx)
        sensitivity = ctx.get("sensitivity", "public")
        complexity = ctx.get("complexity")
        loop_worthy = ctx.get("loop_worthy")

        # ── BRAIN PLAN (Slice X + v2) — decided upstream, one Gemini call per task. ──
        # The brain's per-task book_tier replaces the teacher's fixed tier (the teacher's
        # declared tier was the brain's default and remains the fallback). No brain in
        # ctx (direct invocation, tests) → teacher default, no tools, single book.
        brain = ctx.get("brain") or {}
        tier = brain.get("book_tier") if brain.get("book_tier") in _BRAIN_TIERS \
            else declared_tier

        # ── BOOK PROMPT — the brain's COMPOSED prompt when it wrote one, else the ──
        # static template. Public turns only (a composed prompt was personalized with
        # mem_core and written by a cloud brain; the local/sensitive path always runs
        # the static template — brain.py already returns None there, this is the
        # structural backstop). The teacher's own domain memory (mem_study etc.) is
        # appended to the composed prompt so switching prompts never loses it — the
        # static template folds it in itself.
        composed = (brain.get("composed_prompt") or "").strip()
        if composed and sensitivity == "public":
            prompt = composed
            if domain_memory:
                prompt += ("\n\nKNOWN DOMAIN CONTEXT (from memory — fold in where "
                           "relevant):\n" + "\n".join(f"- {m}" for m in domain_memory))
            prompt_src = "composed(gemini)"
        else:
            prompt = self.build_book_prompt(ctx, domain_memory)
            prompt_src = "static-template"
        print(f"[{self.domain}] book_prompt={prompt_src}")

        # ── SHARED-POOL TOOLS — the brain picked them; draw them generically. ──
        # Skip tools this teacher fetches itself (owns_tools) so nothing fires twice.
        # PUBLIC turns only: every selectable tool egresses the query to an external
        # API, and the brain already clamps tools=[] for secret/private — this gate is
        # the structural backstop, not the only line.
        wanted = [t for t in brain.get("tools", []) if t not in self.owns_tools]
        if wanted and sensitivity == "public":
            fetched = await fetch_from(wanted, query)
            counts = ", ".join(f"{k}({len(v)})" for k, v in fetched.items())
            print(f"[{self.domain}] pool tools fired: {counts}")
            block = format_pool_block(fetched)
            if block:
                prompt += (
                    "\n\nSHARED POOL RESULTS (REAL data fetched live for this task — use "
                    "as your factual basis; mark anything they don't cover as \"unknown\", "
                    "do not invent):\n" + block)

        # Effective tier after the upgrade signal — for the per-turn routing log only.
        eff_tier = "local" if sensitivity == "secret" else upgrade_tier(
            tier, complexity, loop_worthy)

        # ── ENSEMBLE — per-task, brain-decided (the quota guard). ──
        # Two parallel books ONLY when the brain judged this task genuinely benefits;
        # cognition reconciles both (her existing combiner). Public turns only — secret
        # is ollama-only and private never fans out to two cloud calls, and the brain
        # already returns ensemble=False for both.
        if bool(brain.get("ensemble")) and sensitivity == "public":
            books = select_ensemble_books(tier, sensitivity,
                                          complexity=complexity, loop_worthy=loop_worthy)
            if len(books) >= 2:
                ensembled = await self._ensemble_pass(prompt, books, ctx, eff_tier)
                if ensembled is not None:
                    return ensembled
            # only one book family available (rest benched), or both calls failed →
            # fall through to the single best-effort pass below.

        books = select_book_for_tier(tier, sensitivity,
                                     complexity=complexity, loop_worthy=loop_worthy)
        return await self._single_pass(prompt, books, ctx, eff_tier)

    async def _ensemble_pass(self, prompt: str, books: list, ctx: dict, eff_tier: str):
        """Fire the two chosen books at once (asyncio.gather) — two independent research
        passes on the SAME prompt. Returns a result dict carrying an `ensemble` list of the
        raw book outputs for cognition to reconcile, or None if BOTH calls failed (caller
        then falls back to a single pass). Never re-asks a book here — the two-pass spread
        IS the quality strategy; the single-book looper is bypassed for ensemble turns."""
        results = await asyncio.gather(
            *(call_book(prompt, b) for b in books), return_exceptions=True)
        passes = [r for r in results if isinstance(r, dict)]
        if not passes:
            return None  # both failed — let the caller try the single-pass fallback

        # Both books used this teacher turn, logged together.
        used = "+".join(r["book_used"] for r in passes)
        print(f"[{self.domain}] teacher_tier={eff_tier} ensemble_books={used}")
        for r in passes:
            r["self_check"] = self.self_check(r["raw_text"], ctx)
            if not r["self_check"]["passes"]:
                print(f"[{self.domain}] {r['book_used']} weak self-check: "
                      f"{r['self_check']['feedback']}")

        primary = passes[0]
        return {
            # raw_text/self_check track the primary book so downstream fields (export,
            # dossier text) stay a plain string; `ensemble` carries BOTH for cognition.
            "raw_text": primary["raw_text"],
            "book_used": used,
            "tokens": sum(r.get("tokens", 0) for r in passes),
            "domain": self.domain,
            "deliverable": self.deliverable,
            "self_check": primary["self_check"],
            "ensemble": passes,  # ≥1 raw book output — cognition reconciles them
        }

    async def _single_pass(self, prompt: str, books: list, ctx: dict, eff_tier: str) -> dict:
        """The original single-book path with a self-check retry down the fallback list.
        Serves secret/private turns (no ensemble) and the ensemble fallback when only one
        book family is available."""
        last = None
        for book in books:
            try:
                result = await call_book(prompt, book)
            except Exception:
                continue  # call_book already logged / benched; fall to next candidate
            # Per-turn routing visibility: tier the teacher asked for, book that served it.
            print(f"[{self.domain}] teacher_tier={eff_tier} chosen_book={result['book_used']}")
            check = self.self_check(result["raw_text"], ctx)
            last = {
                "raw_text": result["raw_text"],
                "book_used": result["book_used"],
                "tokens": result["tokens"],
                "domain": self.domain,
                "deliverable": self.deliverable,
                "self_check": check,
            }
            if check["passes"]:
                return last
            print(f"[{self.domain}] {book} failed self-check: {check['feedback']}")

        if last is not None:
            return last  # best effort — nothing passed, hand back the last attempt

        return {
            "raw_text": "",
            "book_used": "none",
            "tokens": 0,
            "domain": self.domain,
            "deliverable": self.deliverable,
            "self_check": {"passes": False, "feedback": "no book produced output"},
        }
