from flask import Blueprint, request, jsonify
from core.router import chat
from core.memory import save_message, get_history
from core.profile import store_fact
from core.extractor import extract_and_store
from core.onboarding import get_next_question, get_pending_question, record_question

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    fact = data.get("fact", None)
    category = data.get("category", "general")
    onboarding_answer = data.get("onboarding_answer", False)

    if fact:
        store_fact(fact, category)

    # If this message is answering an onboarding question — record it
    pending = get_pending_question()
    if onboarding_answer and pending:
        record_question(pending, message)
        store_fact(f"{pending} — {message}", "preference")

    history = get_history(session_id)
    result = chat(message, history)

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["response"])

    extract_and_store(message, result["response"])

    # Append onboarding question if active
    next_question = get_next_question()
    if next_question:
        record_question(next_question, "")
        result["onboarding_question"] = next_question

    return jsonify(result)