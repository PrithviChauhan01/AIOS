from urllib.parse import quote

import httpx

from tools.base import Tool

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Wikipedia rejects requests with no descriptive User-Agent (403).
_HEADERS = {"User-Agent": "AIOS/1.0 (personal assistant; contact: psc15042003@gmail.com)"}


class WikipediaTool(Tool):
    """Keyless Wikipedia lookup — search, then pull the summary extract of the
    top 1-2 hits. Fail-soft: any error returns ok=False with no results."""

    name = "wikipedia"

    async def fetch(self, query: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5, headers=_HEADERS) as client:
                sr = await client.get(_SEARCH_URL, params={
                    "action": "query", "list": "search",
                    "srsearch": query, "format": "json",
                })
                sr.raise_for_status()
                hits = sr.json().get("query", {}).get("search", [])[:2]

                results = []
                for hit in hits:
                    title = hit.get("title")
                    if not title:
                        continue
                    summ = await client.get(_SUMMARY_URL.format(quote(title, safe="")))
                    if summ.status_code != 200:
                        continue
                    extract = summ.json().get("extract")
                    if extract:
                        results.append(f"{title}: {extract}")

                if results:
                    return self._ok(query, results)
        except Exception as e:
            print(f"[tools:wikipedia] failed: {e}")
        return self._empty(query)
