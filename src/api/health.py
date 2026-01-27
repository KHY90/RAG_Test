"""헬스 체크 엔드포인트."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """API가 실행 중이고 서비스를 사용할 수 있는지 확인합니다."""
    status = "healthy"
    database = "disconnected"
    embedding_model = "not_loaded"
    llm_model = "not_loaded"

    # 데이터베이스 연결 확인
    try:
        pool = request.app.state.db_pool
        if pool:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                database = "connected"
    except Exception:
        status = "unhealthy"
        database = "disconnected"

    # 임베딩 서비스 확인
    try:
        embedding_service = request.app.state.embedding_service
        if embedding_service and embedding_service.is_loaded:
            embedding_model = "loaded"
    except Exception:
        pass

    # 생성 서비스 확인
    try:
        generation_service = request.app.state.generation_service
        if generation_service and generation_service.is_loaded:
            llm_model = "loaded"
    except Exception:
        pass

    response = HealthResponse(
        status=status,
        database=database,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )

    if status == "unhealthy":
        return JSONResponse(content=response.model_dump(), status_code=503)

    return response
