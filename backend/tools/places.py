"""Google Places (New) — real business directory data for leadgen prospecting.

This is the REAL-DATA engine for finding businesses by category + location. Unlike
a web-search snippet, Places returns structured fields straight from Google's
business directory — names, phones, websites, addresses, ratings, review counts —
so leadgen can hand cognition VERIFIED rows and never has to invent one.

Two layers, same shape as the rest of the pool:

  * search_places(query, region, min_reviews) — the raw async function. Returns a
    list of {name, phone, website, address, rating, reviews} dicts. NEVER raises
    into cognition: a missing key, zero results, or any request error all return
    [] (the caller treats [] as an honest "found nothing", not an excuse to make
    something up).

  * PlacesTool — the shared-pool wrapper (registry handle), read-only, fail-soft.
"""

import httpx

from config import Config
from core.trace import record_tool
from tools.base import Tool

_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Only the fields leadgen actually uses — a tight field mask keeps the call cheap.
_FIELD_MASK = ",".join((
    "places.displayName",
    "places.nationalPhoneNumber",
    "places.websiteUri",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
))


async def search_places(query: str, region: str | None = None,
                        min_reviews: int = 0) -> list[dict]:
    """Search Google Places by free-text query and return structured business rows.

    query       — anything Places understands, e.g. "wedding photographers in Austin".
    region      — ISO 3166-1 alpha-2 region code (US, GB, IN…) passed as regionCode
                  to bias results. Omitted entirely when None so Places infers from
                  the query text.
    min_reviews — drop any business with fewer than this many ratings (0 = keep all).

    Returns [{name, phone, website, address, rating, reviews}], deduped by
    name+phone, in the order Places ranked them. NO category/chain filtering — what
    Places returns is what comes back.

    Fail-soft by contract: a missing GOOGLE_PLACES_API_KEY, zero results, or any
    HTTP/parse error all return [] — this function never raises into cognition."""
    # Recorded HERE, not at the registry: leadgen declares places in owns_tools and
    # calls this structured layer directly, so the registry never sees its draw. Name
    # only — the query and the returned rows never touch a trace.
    record_tool("places")
    key = (Config.GOOGLE_PLACES_API_KEY or "").strip()
    if not key or not (query or "").strip():
        if not key:
            print("[tools:places] GOOGLE_PLACES_API_KEY not set — returning [].")
        return []

    payload: dict = {"textQuery": query}
    if region:
        payload["regionCode"] = region

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": _FIELD_MASK,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(_SEARCH_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        # Surface Google's actual reason (e.g. SERVICE_DISABLED / key restriction /
        # billing) — without it a 403 looks identical to a genuine "no results".
        body = (e.response.text or "")[:500]
        print(f"[tools:places] search rejected ({e.response.status_code}): {body}")
        return []
    except Exception as e:
        print(f"[tools:places] search failed: {e}")
        return []

    rows: list[dict] = []
    seen: set = set()
    for p in data.get("places", []):
        name = (p.get("displayName") or {}).get("text", "").strip()
        phone = (p.get("nationalPhoneNumber") or "").strip()
        reviews = int(p.get("userRatingCount") or 0)
        if min_reviews and reviews < min_reviews:
            continue
        dedup_key = (name.lower(), phone)
        if not name or dedup_key in seen:
            continue
        seen.add(dedup_key)
        rows.append({
            "name": name,
            "phone": phone,
            "website": (p.get("websiteUri") or "").strip(),
            "address": (p.get("formattedAddress") or "").strip(),
            "rating": p.get("rating"),
            "reviews": reviews,
        })
    return rows


class PlacesTool(Tool):
    """Shared-pool handle for Google Places business search. Read-only and
    fail-soft: missing key / zero results / any error → ok=False, no results."""

    name = "places"

    async def fetch(self, query: str) -> dict:
        try:
            rows = await search_places(query)
            results = [
                f"{r['name']} — {r['phone'] or 'no phone'} — {r['website'] or 'no site'} "
                f"({r['reviews']} reviews, rating {r['rating'] if r['rating'] is not None else 'n/a'})"
                for r in rows
            ]
            if results:
                return self._ok(query, results)
        except Exception as e:
            print(f"[tools:places] fetch failed: {e}")
        return self._empty(query)
