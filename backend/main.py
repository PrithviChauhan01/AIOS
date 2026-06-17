from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from api.health_routes import router as health_router
from api.chat_routes import router as chat_router
from db.sqlite_init import init_sqlite
from db.chroma_init import init_chroma
from core.auth import generate_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sqlite()
    init_chroma()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="AIOS", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)

    @app.get("/token")
    async def token():
        return {"token": generate_token()}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        reload=Config.DEBUG,
    )
