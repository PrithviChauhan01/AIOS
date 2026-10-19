"""Tavily — live web search and page extraction, the pool's one Tavily client.

Two layers, mirroring the rest of the pool:

  * search(query, max_results) / extract(url) — the raw async functions. They
    return structured dicts and RAISE (TavilyKeyMissing, httpx errors) so a caller
    can tell "searched and found nothing" from "could not search at all" — leadgen
    depends on that distinction to never cry not-found when the search never ran.

  * TavilyTool — the shared-pool wrapper (registry name "tavily"). Read-only and
    fail-soft: a missing key or any error returns ok=False, never raises.

tools/search.py's web_search delegates here, so there is exactly one place that
talks to api.tavily.com.
"""

import httpx

from config import Config
from core.trace import record_tool
from tools.base import Tool

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_TIMEOUT = 20


class TavilyKeyMissing(RuntimeError):
    """Raised when TAVILY_API_KEY is not configured."""


def _key() -> str:
    key = (Config.TAVILY_API_KEY or "").strip()
    if not key:
        raise TavilyKeyMissing(
            "TAVILY_API_KEY is not set — add it to backend/.env to enable web search.")
    return key


async def search(query: str, max_results: int = 5, search_depth: str = "basic") -> list[dict]:
    """Live web search. Returns [{"title", "url", "content"}] (content is Tavily's
    snippet for the page). "advanced" depth returns longer, more relevant snippets
    at twice the credit cost — leadgen prospecting uses it, chat search doesn't."""
    # Name only — never the query or hits.
    record_tool("tavily")
    payload = {"api_key": _key(), "query": query, "max_results": max_results,
               "search_depth": search_depth}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(_SEARCH_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return [{"title": h.get("title", ""), "url": h.get("url", ""),
             "content": h.get("content", "")}
            for h in data.get("results", [])[:max_results]]


async def extract(url: str) -> dict:
    """Full readable text of one page. Returns {"url", "content"}; content is ""
    when Tavily could not extract the page (it reports those as failed_results)."""
    record_tool("tavily")
    payload = {"api_key": _key(), "urls": [url]}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(_EXTRACT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    hits = data.get("results") or []
    return {"url": url, "content": (hits[0].get("raw_content") or "") if hits else ""}


class TavilyTool(Tool):
    """Shared-pool handle. Read-only, fail-soft: a missing key or any request error
    returns ok=False with no results."""

    name = "tavily"

    async def fetch(self, query: str) -> dict:
        try:
            hits = await search(query)
            results = [f"{h['title']}: {h['content']} ({h['url']})".strip()
                       for h in hits if h.get("title") or h.get("content")]
            if results:
                return self._ok(query, results)
        except TavilyKeyMissing as e:
            print(f"[tools:tavily] {e}")
        except Exception as e:
            print(f"[tools:tavily] failed: {e}")
        return self._empty(query)
