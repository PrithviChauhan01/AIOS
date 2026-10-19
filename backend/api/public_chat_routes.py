"""POST /chat for public mode — anonymous, rate-limited, nothing persisted.

Mounted by main.create_app INSTEAD of api.chat_routes when Config.PUBLIC_MODE is on.
No JWT: the server issues the session_id (see core.public_session) and the client
echoes it back to continue a conversation.
"""

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config
from core import public_session, rate_limit
from core.public_orchestrator import handle_public_message

router = APIRouter()


class PublicChatRequest(BaseModel):
    message: str = ""
    session_id: Optional[str] = None
    trace: bool = False  # true → include per-stage routing/latency in the response


class PublicChatResponse(BaseModel):
    response: str
    mood: str
    provider_used: str
    tokens_used: int
    session_id: str
    remaining: int
    trace: Optional[dict] = None


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/quota")
async def public_quota(request: Request):
    """Free messages left today for this IP — read-only, consumes nothing."""
    return {"remaining": rate_limit.remaining(_client_ip(request)),
            "limit": Config.PUBLIC_RATE_DAILY[0]}


@router.post("/chat", response_model=PublicChatResponse, response_model_exclude_none=True)
async def public_chat_endpoint(data: PublicChatRequest, request: Request):
    # Validate before counting, so an empty or oversized message doesn't spend one
    # of the visitor's few daily messages.
    message = (data.message or "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"detail": "Say something first!"})
    if len(message) > Config.PUBLIC_MAX_INPUT_CHARS:
        return JSONResponse(status_code=400, content={
            "detail": f"That message is a bit long — please keep it under "
                      f"{Config.PUBLIC_MAX_INPUT_CHARS} characters."})

    allowed, remaining, retry_after, msg = rate_limit.check(_client_ip(request))
    if not allowed:
        return JSONResponse(status_code=429, content={"detail": msg, "remaining": 0},
                            headers={"Retry-After": str(retry_after)})

    session_id = public_session.resolve(data.session_id)
    result = await handle_public_message(message, session_id)

    response = {
        "response": result["response"],
        "mood": result["mood"],
        "provider_used": result["provider_used"],
        "tokens_used": result.get("tokens", 0),
        "session_id": session_id,
        "remaining": remaining,
    }
    if data.trace:
        response["trace"] = result["trace"]
    return response
