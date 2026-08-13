from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from core.auth import require_auth
from core.trace import get_trace, list_traces

# Read-only surface over the trace spine. JWT-required like every other non-health
# router here. No SQL lives in this file — core.trace owns the queries, matching the
# dashboard_routes/core.dashboard split.
router = APIRouter(dependencies=[Depends(require_auth)])


@router.get("/trace/{trace_id}")
async def trace_detail(trace_id: str):
    """One trace plus its stages, ordered by seq."""
    result = get_trace(trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return result


@router.get("/traces")
async def trace_summaries(session_id: Optional[str] = None, limit: int = 50,
                          since: Optional[str] = None):
    """Summary rows, newest first. `since` filters on ts_start
    ('YYYY-MM-DD HH:MM:SS.mmm', or a bare date). limit is clamped to 1..500."""
    return list_traces(session_id=session_id, limit=limit, since=since)
