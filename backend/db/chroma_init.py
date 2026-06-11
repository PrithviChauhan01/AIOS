import chromadb
from config import Config

_client = None

def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
    return _client

def init_chroma():
    client = get_chroma_client()

    client.get_or_create_collection(name="long_term_memory")
    client.get_or_create_collection(name="study_documents")
    client.get_or_create_collection(name="lead_profiles")

    print("[db] ChromaDB ready")