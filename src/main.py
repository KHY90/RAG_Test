"""FastAPI 애플리케이션 진입점."""

import logging
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings
from src.db.connection import get_db_pool, close_db_pool
from src.api.health import router as health_router
from src.api.documents import router as documents_router
from src.api.chat import router as chat_router
from src.api.pages import router as pages_router
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


async def auto_load_documents(app: FastAPI):
    """data 폴더의 파일들을 자동으로 임베딩합니다."""
    from src.db.repositories import DocumentRepository, ChunkRepository
    from src.services.ingestion import IngestionService

    data_dir = Path("data")
    if not data_dir.exists():
        print("Data folder not found, skipping auto-load.")
        return

    supported_formats = {"txt", "md", "json"}
    pool = app.state.db_pool
    embedding_service = app.state.embedding_service

    document_repo = DocumentRepository(pool)
    chunk_repo = ChunkRepository(pool, settings.chunk_table)
    ingestion_service = IngestionService(document_repo, chunk_repo, embedding_service)

    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lstrip(".").lower()
        if ext not in supported_formats:
            continue

        # 이미 DB에 있는지 확인
        existing = await document_repo.get_by_filename(file_path.name, settings.chunk_table)
        if existing:
            print(f"  Skipping (already exists): {file_path.name}")
            continue

        # 파일 읽기 및 처리
        content = file_path.read_bytes()
        result = await ingestion_service.process_document(
            filename=file_path.name,
            content=content,
            format=ext,
        )
        print(f"  Loaded: {file_path.name} ({result['chunk_count']} chunks)")


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

    # 생성 서비스 초기화 (시작 시 로드)
    print("Loading LLM model...")
    from src.services.generation import GenerationService

    generation_service = GenerationService()
    try:
        generation_service._load_model()
        app.state.generation_service = generation_service
        print("LLM model loaded.")
    except ImportError as e:
        print(f"WARNING: {e}")
        print("Chat API will use fallback responses.")
        app.state.generation_service = None
    except Exception as e:
        print(f"WARNING: Failed to load LLM model: {e}")
        print("Chat API will use fallback responses until model is available.")
        app.state.generation_service = None

    # data 폴더 자동 임베딩
    print("Loading documents from data folder...")
    await auto_load_documents(app)
    print("Documents loaded.")

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

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# 라우터 포함
app.include_router(health_router)
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(pages_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
