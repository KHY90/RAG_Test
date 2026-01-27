"""채팅 및 검색 API 엔드포인트."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from src.config import settings
from src.db.repositories import ChunkRepository
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    SourceReference,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ErrorResponse,
)
from src.services.search import SearchService
from src.services.generation import GenerationService

router = APIRouter(tags=["Chat"])


def _get_generation_service(request: Request) -> GenerationService:
    """생성 서비스를 가져오거나 생성합니다 (지연 로드)."""
    if request.app.state.generation_service is None:
        request.app.state.generation_service = GenerationService()
    return request.app.state.generation_service


@router.post("/chat", response_model=ChatResponse)
async def ask_question(request: Request, body: ChatRequest) -> ChatResponse:
    """업로드된 문서에서 생성된 답변을 받기 위해 질문합니다.

    하이브리드 검색을 사용하여 관련 컨텍스트를 찾은 다음 답변을 생성합니다.
    """
    # 서비스 가져오기
    pool = request.app.state.db_pool
    embedding_service = request.app.state.embedding_service

    chunk_repo = ChunkRepository(pool)
    search_service = SearchService(chunk_repo, embedding_service)

    # 검색 수행
    search_start = time.perf_counter()
    search_results = await search_service.hybrid_search(
        query=body.question,
        limit=body.top_k,
        search_type="hybrid",
    )
    search_time_ms = (time.perf_counter() - search_start) * 1000

    # 결과 없음 처리
    if not search_results:
        # 관련 내용이 없음을 나타내는 응답 반환
        generation_service = _get_generation_service(request)
        no_context_answer = generation_service.generate_no_context_response(body.question)

        return ChatResponse(
            answer=no_context_answer,
            sources=[],
            search_time_ms=search_time_ms,
            generation_time_ms=0.0,
        )

    # 생성을 위한 컨텍스트 준비
    context_texts = [result.content for result in search_results]

    # 답변 생성
    generation_start = time.perf_counter()
    try:
        generation_service = _get_generation_service(request)
        answer = generation_service.generate_answer(
            question=body.question,
            context=context_texts,
        )
    except FileNotFoundError as e:
        # 모델 파일을 찾을 수 없음 - 플레이스홀더 답변과 함께 검색 결과 반환
        answer = _generate_fallback_answer(body.question, context_texts)
    except Exception as e:
        # 생성 실패 - 오류 메시지와 함께 검색 결과 반환
        answer = f"생성 오류: {str(e)}. 관련 소스는 같습니다."
    generation_time_ms = (time.perf_counter() - generation_start) * 1000

    # 소스 참조 생성
    sources = [
        SourceReference(
            document_id=result.document_id,
            filename=result.filename,
            chunk_index=result.chunk_index,
            content_preview=result.content[:200],
            relevance_score=min(result.score, 1.0),  # RRF 점수에 대해 1.0으로 제한
        )
        for result in search_results
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        search_time_ms=search_time_ms,
        generation_time_ms=generation_time_ms,
    )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: Request, body: SearchRequest) -> SearchResponse:
    """답변을 생성하지 않고 관련 문서 청크를 검색합니다.

    검색 품질 디버깅 및 테스트에 유용합니다.
    """
    # search_type 유효성 검사
    valid_types = {"hybrid", "dense", "sparse", "trigram"}
    if body.search_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="INVALID_SEARCH_TYPE",
                message=f"Invalid search_type. Must be one of: {', '.join(valid_types)}",
            ).model_dump(),
        )

    # 서비스 가져오기
    pool = request.app.state.db_pool
    embedding_service = request.app.state.embedding_service

    chunk_repo = ChunkRepository(pool)
    search_service = SearchService(chunk_repo, embedding_service)

    # 검색 수행
    search_start = time.perf_counter()
    search_results = await search_service.hybrid_search(
        query=body.query,
        limit=body.top_k,
        search_type=body.search_type,
    )
    search_time_ms = (time.perf_counter() - search_start) * 1000

    # 응답 생성
    results = [
        SearchResultItem(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            chunk_index=result.chunk_index,
            content=result.content,
            score=result.score,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            trigram_score=result.trigram_score,
        )
        for result in search_results
    ]

    return SearchResponse(
        results=results,
        search_time_ms=search_time_ms,
    )


def _generate_fallback_answer(question: str, context: list[str]) -> str:
    """LLM을 사용할 수 없을 때 간단한 대체 답변을 생성합니다.

    인수:
        question: 사용자의 질문
        context: 관련 텍스트 청크 목록

    반환값:
        컨텍스트에 기반한 간단한 답변
    """
    # 질문이 한국어인지 감지
    is_korean = any('\uac00' <= char <= '\ud7a3' for char in question)

    if not context:
        if is_korean:
            return "관련 문서를 찾을 수 없습니다."
        return "관련 문서를 찾을 수 없습니다."

    # 가장 관련성 높은 컨텍스트와 함께 메모 반환
    top_context = context[0][:500]

    if is_korean:
        return f"LLM 모델이 로드되지 않았습니다. 관련 내용:\n\n{top_context}"
    return f"LLM 모델이 로드되지 않았습니다. 관련 내용:\n\n{top_context}"
