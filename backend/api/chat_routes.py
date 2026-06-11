from flask import Blueprint, request, jsonify
from core.router import chat
from core.memory import save_message, get_history
from core.profile import store_fact
from core.extractor import extract_and_store

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    fact = data.get("fact", None)
    category = data.get("category", "general")

    if fact:
        store_fact(fact, category)

    history = get_history(session_id)
    result = chat(message, history)

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["response"])

    # Auto-extract facts in background
    extract_and_store(message, result["response"])

    return jsonify(result)