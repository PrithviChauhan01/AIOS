"""Web search — real live results via the Tavily Search API.

Two layers, mirroring the rest of the pool:

  * web_search(query, max_results) — the raw async function. Returns a list of
    {title, url, content} dicts and RAISES a clean error if TAVILY_API_KEY is
    missing or the call fails. Teachers call this directly when they want the
    structured rows (leadgen dossier, jobs shortlist) and want to know whether
    live data was actually available.

  * SearchTool — the shared-pool wrapper (registry handle), same shape as
    WikipediaTool. fetch() is read-only and fail-soft: on a missing key or any
    error it returns ok=False with no results, never raising. This is what the
    generic pool plumbing (registry.fetch_from) calls.
"""

from core.trace import record_tool
from tools import tavily
from tools.base import Tool
from tools.tavily import TavilyKeyMissing  # re-exported: callers catch it from here


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a live web search through Tavily and return the top results.

    Returns a list of {"title", "url", "content"} dicts (content is Tavily's
    extracted snippet for the page). Raises TavilyKeyMissing if no API key is
    configured, and propagates httpx errors on a failed request — callers that
    must stay fail-soft should wrap this (see SearchTool.fetch)."""
    # Recorded HERE, not at the registry: leadgen/jobs declare search in owns_tools
    # and call this structured layer directly. Name only — never the query or hits.
    record_tool("search")
    return await tavily.search(query, max_results)


class SearchTool(Tool):
    """Keyed Tavily web search for the shared pool. Read-only and fail-soft: a
    missing key or any request error returns ok=False with no results."""

    name = "search"

    async def fetch(self, query: str) -> dict:
        try:
            hits = await web_search(query)
            results = [
                f"{h['title']}: {h['content']} ({h['url']})".strip()
                for h in hits if h.get("title") or h.get("content")
            ]
            if results:
                return self._ok(query, results)
        except TavilyKeyMissing as e:
            print(f"[tools:search] {e}")
        except Exception as e:
            print(f"[tools:search] failed: {e}")
        return self._empty(query)
