import os
from dotenv import load_dotenv

load_dotenv()

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

    SQLITE_PATH = os.getenv("SQLITE_PATH", "./db/aios.db")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./db/chroma")
    VISION_ENABLED = os.getenv("VISION_ENABLED", "false").lower() == "true"