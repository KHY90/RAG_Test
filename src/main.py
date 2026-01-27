"""FastAPI 애플리케이션 진입점."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db.connection import get_db_pool, close_db_pool
from src.api.health import router as health_router
from src.api.documents import router as documents_router
from src.api.chat import router as chat_router


# 전역 서비스 인스턴스 (시작 시 초기화)
embedding_service = None
generation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """시작 및 종료를 위한 애플리케이션 수명 주기 관리자."""
    global embedding_service, generation_service

    # 시작
    print("Starting up...")

    # 데이터베이스 풀 초기화
    print("Connecting to database...")
    pool = await get_db_pool()
    app.state.db_pool = pool
    print("Database connected.")

    # 임베딩 서비스 초기화
    print("Loading embedding model...")
    from src.services.embedding import EmbeddingService

    embedding_service = EmbeddingService()
    await embedding_service.load_model()
    app.state.embedding_service = embedding_service
    print("Embedding model loaded.")

    # 생성 서비스 초기화 (메모리 절약을 위한 지연 로드)
    print("Generation service will be loaded on first use.")
    app.state.generation_service = None

    yield

    # 종료
    print("Shutting down...")
    await close_db_pool()
    print("Database connection closed.")


app = FastAPI(
    title="Hybrid RAG Search API",
    description="Local RAG system with hybrid search (dense + sparse + trigram)",
    version="1.0.0",
    lifespan=lifespan,
)

# 로컬 개발을 위한 CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(health_router)
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
