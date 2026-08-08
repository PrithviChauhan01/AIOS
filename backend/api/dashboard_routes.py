from fastapi import APIRouter, Depends

from core.auth import require_auth
from core import dashboard
from tools.registry import get_tool

router = APIRouter(prefix="/dashboard", dependencies=[Depends(require_auth)])


@router.get("/jobs")
async def jobs():
    return dashboard.get_jobs()


@router.get("/leads")
async def leads():
    return dashboard.get_leads()


@router.get("/fitness")
async def fitness():
    return dashboard.get_fitness_logs()


@router.get("/study")
async def study():
    return dashboard.get_study_sessions()


@router.get("/habits")
async def habits():
    return dashboard.get_habits()


@router.get("/reminders")
async def reminders():
    return dashboard.get_reminders()


# ── The only two WRITE routes on this surface; everything else here is read-only. ──
# Neither contains SQL: both delegate to the SAME RemindersTool the chat action path
# executes (tools/reminders.py, via the shared registry), so "complete" and "delete"
# have exactly one implementation in the system rather than a second copy behind the
# UI. `ok` is the tool's real result — False means no row matched (already gone), and
# the panel refetches either way so it shows the actual table state, never an assumed one.
@router.post("/reminders/{reminder_id}/complete")
async def complete_reminder(reminder_id: int):
    return {"ok": bool(get_tool("reminders").complete(reminder_id)), "id": reminder_id}


@router.post("/reminders/{reminder_id}/delete")
async def delete_reminder(reminder_id: int):
    return {"ok": bool(get_tool("reminders").delete(reminder_id)), "id": reminder_id}


@router.get("/overview")
async def overview():
    return dashboard.get_overview()
