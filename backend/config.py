import os
from dotenv import load_dotenv

load_dotenv()

JWT_MIN_BYTES = 32  # HS256 secret must be at least this long, or we refuse to run

# Single source of truth for path resolution. config.py lives in backend/, so
# this is always the backend dir regardless of which CWD a process was launched
# from (uvicorn, chat client, scheduler, ad-hoc scripts).
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def _anchor(path: str) -> str:
    """Resolve a configured path to an absolute one. Absolute paths pass through;
    relative paths anchor to the backend dir — NOT the CWD. This guarantees every
    component opens the identical DB file no matter where it was started."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_BACKEND_DIR, path))


class Config:
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_ENV") != "production"
    JWT_SECRET = os.getenv("JWT_SECRET", "change_this")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
    GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
    PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Fernet key for the encrypted local vault (secret-tier data). Empty → vault
    # generates an ephemeral key at runtime and prints instructions to persist it.
    VAULT_KEY = os.getenv("VAULT_KEY", "")

    # Gmail send (SMTP). Use a Google App Password, NOT the account password.
    # Empty → the email tool drafts but refuses to send, with a clear instruction.
    GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

    # Anchored to backend/ so writer and scheduler (and every other consumer)
    # always open the same file, independent of CWD.
    SQLITE_PATH = _anchor(os.getenv("SQLITE_PATH", "./db/aios.db"))
    CHROMA_PATH = _anchor(os.getenv("CHROMA_PATH", "./db/chroma"))
    VISION_ENABLED = os.getenv("VISION_ENABLED", "false").lower() == "true"

    @classmethod
    def validate(cls):
        """Fail closed at startup on an insecure auth secret. Call before serving."""
        if len(cls.JWT_SECRET.encode("utf-8")) < JWT_MIN_BYTES:
            raise RuntimeError(
                f"JWT_SECRET is too short (need >= {JWT_MIN_BYTES} bytes). "
                "Refusing to run with an insecure secret. Set a strong JWT_SECRET "
                "in backend/.env, e.g.:\n"
                '    python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )