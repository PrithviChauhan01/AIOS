import os
from dotenv import load_dotenv

load_dotenv()

JWT_MIN_BYTES = 32  # HS256 secret must be at least this long, or we refuse to run


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

    SQLITE_PATH = os.getenv("SQLITE_PATH", "./db/aios.db")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./db/chroma")
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