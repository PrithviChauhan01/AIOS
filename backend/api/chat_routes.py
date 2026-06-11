from flask import Blueprint, request, jsonify
from core.router import chat

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])

    result = chat(message, history)
    return jsonify(result)