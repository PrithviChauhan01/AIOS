from fastapi import APIRouter
import time

router = APIRouter()
START_TIME = time.time()

@router.get("/health")
async def health():
    uptime_seconds = int(time.time() - START_TIME)
    h = uptime_seconds // 3600
    m = (uptime_seconds % 3600) // 60
    s = uptime_seconds % 60
    return {
        "status": "ok",
        "vision": False,
        "memory": "connected",
        "uptime": f"{h}:{m:02d}:{s:02d}"
    }
