from flask import Blueprint, request, jsonify
from core.router import chat
from core.memory import save_message, get_history

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")

    history = get_history(session_id)
    result = chat(message, history)

    save_message(session_id, "user", message)
    save_message(session_id, "assistant", result["response"])

    return jsonify(result)