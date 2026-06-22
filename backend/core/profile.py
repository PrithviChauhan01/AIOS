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


def get_relevant_facts(query: str, n_results: int = 5, cloud_bound: bool = True) -> list:
    client = get_chroma_client()
    collection = client.get_collection("long_term_memory")
    count = collection.count()
    if count == 0:
        return []

    kwargs = {"query_texts": [query], "n_results": min(n_results, count)}
    if cloud_bound:
        # HARD filter at the query, not post-filtering: only public facts may be
        # injected into a prompt that will be sent to a cloud provider. Facts with
        # no `tier` metadata (legacy, pre-tagging) are excluded by this — fail-safe.
        kwargs["where"] = {"tier": "public"}
    # cloud_bound=False is the local/ollama path: public + private are allowed.
    # secret never reaches Chroma, so no extra guard is needed for it here.

    results = collection.query(**kwargs)
    return results["documents"][0] if results["documents"] else []
