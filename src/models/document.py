"""문서 및 청크 데이터 모델."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class Document:
    """업로드된 문서를 나타냅니다."""

    id: UUID
    filename: str
    content: str
    format: str  # 'txt', 'md', 'json'
    file_size: int
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0

    @classmethod
    def from_db_row(cls, row: dict) -> "Document":
        """데이터베이스 행에서 문서를 생성합니다."""
        return cls(
            id=row["id"],
            filename=row["filename"],
            content=row["content"],
            format=row["format"],
            file_size=row["file_size"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            chunk_count=row.get("chunk_count", 0),
        )


@dataclass
class Chunk:
    """검색을 위한 문서 콘텐츠 청크를 나타냅니다."""

    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    token_count: int
    embedding: list[float] = field(default_factory=list)
    created_at: Optional[datetime] = None

    # 검색 중 채워지는 선택적 필드
    filename: Optional[str] = None
    similarity_score: Optional[float] = None
    bm25_score: Optional[float] = None
    trigram_score: Optional[float] = None
    rrf_score: Optional[float] = None

    @classmethod
    def from_db_row(cls, row: dict) -> "Chunk":
        """데이터베이스 행에서 청크를 생성합니다."""
        return cls(
            id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            token_count=row["token_count"],
            embedding=list(row.get("embedding", [])) if row.get("embedding") else [],
            created_at=row.get("created_at"),
            filename=row.get("filename"),
            similarity_score=row.get("similarity"),
            bm25_score=row.get("bm25_rank"),
            trigram_score=row.get("trigram_sim"),
        )
