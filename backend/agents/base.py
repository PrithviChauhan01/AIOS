from abc import ABC, abstractmethod
from db.chroma_init import get_chroma_client
from core.books import select_book, call_book


class Teacher(ABC):
    """Domain sub-agent (LLD §3). A Teacher is a domain expert that retrieves its
    own long-term memory, frames a domain-specific prompt, picks the right book
    (model) for the job and pulls RAW material out of it. It returns raw material
    only — never a user-facing answer. Personality, synthesis and final delivery
    are cognition's job, not the Teacher's."""

    domain: str = "general"
    memory_ns: str = "long_term_memory"
    deliverable: bool = False  # True → cognition presents as a structured block, not prose

    # ── Subclass contract ──
    @abstractmethod
    def required_capability(self, ctx: dict) -> dict:
        """capability_spec consumed by core.books.select_book."""

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
        prompt = self.build_book_prompt(ctx, domain_memory)

        spec = self.required_capability(ctx)
        tier = ctx.get("sensitivity", "public")

        last = None
        for book in select_book(spec, tier):
            try:
                result = await call_book(prompt, book)
            except Exception:
                continue  # call_book already logged / benched; fall to next candidate
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
