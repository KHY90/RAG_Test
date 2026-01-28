"""API 요청/응답 모델을 위한 Pydantic 스키마."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# 문서 스키마
class DocumentUploadResponse(BaseModel):
    """문서 업로드 성공 후 응답."""

    id: UUID
    filename: str
    format: str
    file_size: int
    chunk_count: int
    created_at: datetime


class DocumentListItem(BaseModel):
    """목록 응답의 문서 항목."""

    id: UUID
    filename: str
    format: str
    file_size: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """문서 목록 조회 응답."""

    documents: list[DocumentListItem]
    total: int


class DocumentDetail(DocumentListItem):
    """상세 문서 정보."""

    content_preview: str = Field(max_length=500)


# 채팅 스키마
class ChatRequest(BaseModel):
    """질문 요청."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceReference(BaseModel):
    """소스 문서 청크에 대한 참조."""

    document_id: UUID
    filename: str
    chunk_index: int
    content_preview: str = Field(max_length=500)
    relevance_score: float


class ChatResponse(BaseModel):
    """생성된 답변이 포함된 응답."""

    answer: str
    sources: list[SourceReference]
    search_time_ms: float
    generation_time_ms: float


# 검색 스키마
class SearchRequest(BaseModel):
    """문서 검색 요청."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    search_type: str = Field(default="hybrid")  # hybrid, dense, sparse, trigram


class SearchResultItem(BaseModel):
    """개별 검색 결과 항목."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    content: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    trigram_score: Optional[float] = None


class SearchResponse(BaseModel):
    """검색 결과가 포함된 응답."""

    results: list[SearchResultItem]
    search_time_ms: float


# 헬스 체크 스키마
class HealthResponse(BaseModel):
    """헬스 체크 응답."""

    status: str  # healthy, unhealthy
    database: str = "disconnected"  # connected, disconnected
    embedding_model: str = "not_loaded"  # loaded, not_loaded
    llm_model: str = "not_loaded"  # loaded, not_loaded


# 오류 스키마
class ErrorResponse(BaseModel):
    """오류 응답 형식."""

    error: str
    message: str
    details: Optional[dict] = None
