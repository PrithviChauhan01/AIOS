from fastapi import APIRouter, Depends

from core.auth import require_auth
from core import dashboard

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


@router.get("/overview")
async def overview():
    return dashboard.get_overview()
