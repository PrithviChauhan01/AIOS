from flask import Blueprint, jsonify
import time

health_bp = Blueprint("health", __name__)
START_TIME = time.time()

@health_bp.route("/health", methods=["GET"])
def health():
    uptime_seconds = int(time.time() - START_TIME)
    h = uptime_seconds // 3600
    m = (uptime_seconds % 3600) // 60
    s = uptime_seconds % 60
    return jsonify({
        "status": "ok",
        "vision": False,
        "memory": "connected",
        "uptime": f"{h}:{m:02d}:{s:02d}"
    })