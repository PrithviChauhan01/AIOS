from db.chroma_init import get_chroma_client
import uuid

# Only these two tiers may ever reach Chroma. `secret` is handled upstream by the
# vault and must never be embedded — see core/vault.py / core/extractor.py.
_CHROMA_TIERS = ("public", "private")


def store_fact(fact: str, category: str = "general", tier: str = "public"):
    # Fail-safe: anything that isn't an explicit Chroma-safe tier is treated as
    # the most sensitive case and refused here (it belongs in the vault, not the
    # embedded store). Never embed something we're unsure about.
    if tier not in _CHROMA_TIERS:
        print(f"[profile] refusing to embed fact at tier '{tier}' — not stored in Chroma")
        return
    client = get_chroma_client()
    collection = client.get_collection("long_term_memory")
    collection.add(
        documents=[fact],
        metadatas=[{"category": category, "tier": tier}],
        ids=[str(uuid.uuid4())]
    )


# Relevance ceiling (L2 distance). Chroma always returns the N *nearest* docs,
# even when "nearest" is unrelated — that was the memory bleed: on small-talk the
# densest cluster in the store (job/AI-software/wake-time) came back at d≈1.74–1.9
# and got injected into every prompt. Calibrated against this store: genuinely
# on-topic hits land at ~1.2–1.5; small-talk's nearest facts at ~1.74+. 1.65 sits
# in the dead band between them — keep real matches, drop the bleed. Err tight:
# the prompt's own rule is "carry quietly, do not recite", so silence > noise.
_MAX_DISTANCE = 1.65


def get_relevant_facts(query: str, n_results: int = 4, cloud_bound: bool = True) -> list:
    client = get_chroma_client()
    collection = client.get_collection("long_term_memory")
    count = collection.count()
    if count == 0:
        return []

    kwargs = {
        "query_texts": [query],
        "n_results": min(n_results, count),
        "include": ["documents", "distances"],
    }
    if cloud_bound:
        # HARD filter at the query, not post-filtering: only public facts may be
        # injected into a prompt that will be sent to a cloud provider. Facts with
        # no `tier` metadata (legacy, pre-tagging) are excluded by this — fail-safe.
        kwargs["where"] = {"tier": "public"}
    # cloud_bound=False is the local/ollama path: public + private are allowed.
    # secret never reaches Chroma, so no extra guard is needed for it here.

    results = collection.query(**kwargs)
    docs = results["documents"][0] if results["documents"] else []
    dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

    # Relevance-filter + dedup. The store is full of near-identical facts, so an
    # unfiltered top-N is mostly noise and repeats. Keep only genuinely-close,
    # distinct facts — nothing relevant returns [] → "(nothing relevant)".
    out, seen = [], set()
    for doc, dist in zip(docs, dists):
        if dist > _MAX_DISTANCE:
            continue
        key = doc.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out
