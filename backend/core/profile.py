from db.chroma_init import get_chroma_client
import uuid

def store_fact(fact: str, category: str = "general"):
    client = get_chroma_client()
    collection = client.get_collection("long_term_memory")
    collection.add(
        documents=[fact],
        metadatas=[{"category": category}],
        ids=[str(uuid.uuid4())]
    )

def get_relevant_facts(query: str, n_results: int = 5) -> list:
    client = get_chroma_client()
    collection = client.get_collection("long_term_memory")
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )
    return results["documents"][0] if results["documents"] else []