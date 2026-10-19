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


# The personal stores' default locations. Public mode refuses to boot pointed at any
# of them (see Config.validate_public) — a public visitor must never share a file
# with Sir's memory, conversations or vault, even read-only.
_PERSONAL_SQLITE = _anchor("./db/aios.db")
_PERSONAL_CHROMA = _anchor("./db/chroma")
_PERSONAL_DEFAULTS = (_PERSONAL_SQLITE, _PERSONAL_CHROMA)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class Config:
    HOST = os.getenv("HOST") or "127.0.0.1"
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_ENV") != "production"
    JWT_SECRET = os.getenv("JWT_SECRET", "change_this")
    JWT_TTL_HOURS = 12

    # Owner login (POST /token, personal mode). Generate both with
    # scripts/setup_owner.py. Unset → every login attempt fails.
    OWNER_PASSWORD_HASH = os.getenv("OWNER_PASSWORD_HASH", "")
    OWNER_TOTP_SECRET = os.getenv("OWNER_TOTP_SECRET", "")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
    # NVIDIA NIM (OpenAI-compatible) — Nemotron high-reasoning books.
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
    PICOVOICE_API_KEY = os.getenv("PICOVOICE_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    # Google Places (New) — real business directory data for leadgen prospecting.
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

    # Fernet key for the encrypted local vault (secret-tier data). Empty → vault
    # generates an ephemeral key at runtime and prints instructions to persist it.
    VAULT_KEY = os.getenv("VAULT_KEY", "")

    # Gmail send (SMTP). Use a Google App Password, NOT the account password.
    # Empty → the email tool drafts but refuses to send, with a clear instruction.
    GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
    # The one line outreach emails may say about Studio OS. The composer is told to
    # describe the product ONLY with this text, so set it to what Studio OS really is.
    STUDIO_OS_PITCH = os.getenv(
        "STUDIO_OS_PITCH", "Studio OS, software I'm building for creative studios")

    # Anchored to backend/ so writer and scheduler (and every other consumer)
    # always open the same file, independent of CWD.
    SQLITE_PATH = _anchor(os.getenv("SQLITE_PATH", "./db/aios.db"))
    CHROMA_PATH = _anchor(os.getenv("CHROMA_PATH", "./db/chroma"))
    # Encrypted vault table lives in the main SQLite file unless pointed elsewhere.
    VAULT_PATH = _anchor(os.getenv("VAULT_PATH") or SQLITE_PATH)
    OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://localhost:11434"
    # Reminder delivery backend: toast (Windows) | email (to self) | log.
    NOTIFIER = os.getenv("NOTIFIER", "toast")
    VISION_ENABLED = os.getenv("VISION_ENABLED", "false").lower() == "true"

    # ── Public mode ──
    # true → the same app serves anonymous visitors: no JWT, in-memory session
    # history only, every writing tool refused, neutral persona, per-IP rate limit.
    # Default false — personal mode is untouched unless this is set explicitly.
    PUBLIC_MODE = os.getenv("PUBLIC_MODE", "false").lower() == "true"
    PUBLIC_SESSION_IDLE_S = 30 * 60      # a visitor's history expires after 30 min idle
    PUBLIC_MAX_INPUT_CHARS = 2000
    PUBLIC_RATE_DAILY = (5, 24 * 60 * 60)  # 5 messages / day, per IP

    @classmethod
    def validate(cls):
        """Fail closed at startup on an insecure auth secret (personal mode) or on a
        public deployment pointed at the personal stores. Call before serving."""
        if cls.PUBLIC_MODE:
            cls.validate_public()
            return  # public mode issues no JWTs, so the secret is never used
        if len(cls.JWT_SECRET.encode("utf-8")) < JWT_MIN_BYTES:
            raise RuntimeError(
                f"JWT_SECRET is too short (need >= {JWT_MIN_BYTES} bytes). "
                "Refusing to run with an insecure secret. Set a strong JWT_SECRET "
                "in backend/.env, e.g.:\n"
                '    python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if not (cls.OWNER_PASSWORD_HASH and cls.OWNER_TOTP_SECRET):
            print("[config] WARNING: OWNER_PASSWORD_HASH / OWNER_TOTP_SECRET not set — "
                  "owner login will refuse every attempt. Run scripts/setup_owner.py.")

    @classmethod
    def validate_public(cls):
        """Refuse to boot in public mode if any store resolves to a personal default.
        An unset SQLITE_PATH/CHROMA_PATH falls back to exactly those defaults, so a
        public deployment must point all three somewhere of its own explicitly."""
        for name in ("SQLITE_PATH", "CHROMA_PATH", "VAULT_PATH"):
            path = getattr(cls, name)
            if any(_same_path(path, d) for d in _PERSONAL_DEFAULTS):
                raise RuntimeError(
                    f"PUBLIC_MODE=true but {name} resolves to a personal default "
                    f"({path}). Refusing to run a public server against personal data. "
                    f"Set SQLITE_PATH, CHROMA_PATH and VAULT_PATH to public-only "
                    f"locations in the public deployment's environment."
                )
