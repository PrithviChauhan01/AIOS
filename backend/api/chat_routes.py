from typing import Optional
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from core.orchestrator import handle_message
from core.profile import store_fact
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


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_endpoint(data: ChatRequest):
    if data.fact:
        store_fact(data.fact, data.category)

    pending = get_pending_question()
    if data.onboarding_answer and pending:
        record_question(pending, data.message)
        store_fact(f"{pending} — {data.message}", "preference")

    # Orchestrator owns history load, the LLM chain, save_message and fact extraction.
    # voice_flag=False so it does NOT speak — we speak here, where mood + onboarding live.
    result = await handle_message(data.message, data.session_id, voice_flag=False)

    response = {
        "response": result["response"],
        "mood": result["mood"],
        "provider_used": result["provider_used"],
        "tokens_used": result.get("tokens_used", result.get("tokens", 0)),
    }

    # Speak response if voice requested (blocking — keep off the event loop).
    if data.voice:
        from voice.speaker import speak
        await run_in_threadpool(speak, response["response"], response["mood"])

    # Append onboarding question if active
    next_question = get_next_question()
    if next_question:
        record_question(next_question, "")
        response["onboarding_question"] = next_question

    return response
