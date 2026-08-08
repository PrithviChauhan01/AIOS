from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.auth import require_auth
from core.memory import archive_session, get_conversation, get_sessions

router = APIRouter(prefix="/conversations", dependencies=[Depends(require_auth)])


class ArchiveRequest(BaseModel):
    session_id: str


@router.get("/sessions")
async def sessions():
    return get_sessions()


@router.get("")
async def conversation(session_id: str = "main", limit: Optional[int] = None):
    return get_conversation(session_id, limit)


@router.post("/archive")
async def archive(data: ArchiveRequest):
    archive_session(data.session_id)
    return {"status": "ok"}
