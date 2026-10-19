from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from api.health_routes import router as health_router, health_check
from api.chat_routes import router as chat_router
from api.dashboard_routes import router as dashboard_router
from api.conversations_routes import router as conversations_router
from api.trace_routes import router as trace_router
from api.public_chat_routes import router as public_chat_router
from db.sqlite_init import init_sqlite
from db.chroma_init import init_chroma
from core.auth import generate_token
from core import owner_auth
from core.scheduler import start_scheduler, stop_scheduler


class TokenRequest(BaseModel):
    password: str = Field("", max_length=1024)  # bounded: it is hashed per attempt
    code: str = Field("", max_length=16)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Config.PUBLIC_MODE:
        # Nothing to open: public turns persist nothing, and no reminders fire.
        print("[aios] PUBLIC_MODE — anonymous /chat only; no stores, no scheduler")
        yield
        return
    init_sqlite()
    init_chroma()
    start_scheduler()  # reminder firing — runs whenever the backend is up
    try:
        yield
    finally:
        stop_scheduler()


def create_app() -> FastAPI:
    Config.validate()  # fail closed on a weak JWT_SECRET before anything serves
    app = FastAPI(title="AIOS", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",   # dashboard/ — Vite dev server
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if Config.PUBLIC_MODE:
        # Public surface is /health and /chat only. No /token, and none of the routers
        # that read Sir's data (dashboard, conversations, traces, /health/deep).
        app.add_api_route("/health", health_check, methods=["GET"])
        app.include_router(public_chat_router)
        return app

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(dashboard_router)
    app.include_router(conversations_router)
    app.include_router(trace_router)

    @app.post("/token")
    async def token(data: TokenRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        # bcrypt is deliberately slow — keep it off the event loop.
        outcome = await run_in_threadpool(owner_auth.attempt, ip, data.password, data.code)
        if outcome == "ok":
            return {"token": generate_token()}
        if outcome == "locked":
            return JSONResponse(status_code=429,
                                content={"detail": "Too many attempts. Try again later."})
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials."})

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
    )
