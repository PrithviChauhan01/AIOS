from typing import Optional
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from core.router import chat
from core.memory import save_message, get_history
from core.profile import store_fact
from core.extractor import extract_and_store
from core.onboarding import get_next_question, get_pending_question, record_question

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = ""
    session_id: str = "default"
    fact: Optional[str] = None
    category: str = "general"
    onboarding_answer: bool = False
    voice: bool = False


class ChatResponse(BaseModel):
    response: str
    mood: str
    provider_used: str
    tokens_used: int
    onboarding_question: Optional[str] = None


def _handle_chat(data: ChatRequest) -> dict:
    """Synchronous chat turn — identical logic to the Flask endpoint."""
    if data.fact:
        store_fact(data.fact, data.category)

    pending = get_pending_question()
    if data.onboarding_answer and pending:
        record_question(pending, data.message)
        store_fact(f"{pending} — {data.message}", "preference")

    history = get_history(data.session_id)
    result = chat(data.message, history)

    save_message(data.session_id, "user", data.message)
    save_message(data.session_id, "assistant", result["response"])

    extract_and_store(data.message, result["response"])

    # Speak response if voice requested
    if data.voice:
        from voice.speaker import speak
        speak(result["response"], result.get("mood", "neutral"))

    # Append onboarding question if active
    next_question = get_next_question()
    if next_question:
        record_question(next_question, "")
        result["onboarding_question"] = next_question

    return result


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_endpoint(data: ChatRequest):
    # Core logic is blocking (network + DB), so run it off the event loop.
    return await run_in_threadpool(_handle_chat, data)
