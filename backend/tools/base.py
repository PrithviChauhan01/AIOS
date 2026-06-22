from abc import ABC, abstractmethod


class Tool(ABC):
    """Generic shared-pool tool. A Tool fetches real external facts for a query.
    Tools are shared infrastructure (like the book pool) — not owned by any teacher.
    A Tool NEVER raises: on error or no result it returns ok=False with empty results."""

    name: str = "tool"

    @abstractmethod
    async def fetch(self, query: str) -> dict:
        """Returns {source, query, results: list[str], ok: bool}. Must be fail-soft."""

    # ── Fail-soft result helpers ──
    def _ok(self, query: str, results: list) -> dict:
        return {"source": self.name, "query": query, "results": results, "ok": True}

    def _empty(self, query: str) -> dict:
        return {"source": self.name, "query": query, "results": [], "ok": False}
