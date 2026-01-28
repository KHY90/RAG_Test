"""FastAPI 애플리케이션 진입점."""

import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings
from src.db.connection import get_db_pool, close_db_pool
from src.api.health import router as health_router
from src.api.documents import router as documents_router
from src.api.chat import router as chat_router
from src.models.schemas import ErrorResponse

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅을 위한 미들웨어."""

    async def dispatch(self, request: Request, call_next):
        """요청을 처리하고 로깅합니다."""
        start_time = time.perf_counter()

        # 요청 로깅
        logger.info(
            f"Request: {request.method} {request.url.path}"
        )

        # 요청 처리
        response = await call_next(request)

        # 응답 시간 계산
        process_time_ms = (time.perf_counter() - start_time) * 1000

        # 응답 로깅
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} time={process_time_ms:.2f}ms"
        )

        # 응답 헤더에 처리 시간 추가
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """예외 처리 및 ErrorResponse 형식 반환을 위한 미들웨어."""

    async def dispatch(self, request: Request, call_next):
        """요청을 처리하고 예외를 잡아 ErrorResponse로 변환합니다."""
        try:
            return await call_next(request)
        except Exception as exc:
            # 오류 로깅
            logger.error(
                f"Unhandled exception: {type(exc).__name__}: {str(exc)}\n"
                f"{traceback.format_exc()}"
            )

            # ErrorResponse 형식으로 반환
            error_response = ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred.",
                details={"exception": type(exc).__name__} if settings.debug else None,
            )

            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
            )


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

# 미들웨어 추가 (역순으로 실행됨 - 마지막에 추가된 것이 먼저 실행)
# 1. CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 오류 처리 미들웨어
app.add_middleware(ErrorHandlingMiddleware)

# 3. 요청 로깅 미들웨어
app.add_middleware(RequestLoggingMiddleware)

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
