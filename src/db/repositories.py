"""문서 및 청크를 위한 데이터 액세스 계층."""

from typing import Optional
from uuid import UUID

import asyncpg

from src.models.document import Document, Chunk


class DocumentRepository:
    """문서 CRUD 작업을 위한 리포지토리."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create(
        self,
        filename: str,
        content: str,
        format: str,
        file_size: int,
    ) -> Document:
        """새 문서를 생성합니다."""
        query = """
            INSERT INTO documents (filename, content, format, file_size)
            VALUES ($1, $2, $3, $4)
            RETURNING id, filename, content, format, file_size, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, filename, content, format, file_size)
            return Document.from_db_row(dict(row))

    async def get_by_id(self, document_id: UUID) -> Optional[Document]:
        """청크 수와 함께 ID로 문서를 조회합니다."""
        query = """
            SELECT d.*, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.id = $1
            GROUP BY d.id
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, document_id)
            if row:
                return Document.from_db_row(dict(row))
            return None

    async def get_by_filename(self, filename: str) -> Optional[Document]:
        """파일 이름으로 문서를 조회합니다."""
        query = """
            SELECT d.*, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.filename = $1
            GROUP BY d.id
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, filename)
            if row:
                return Document.from_db_row(dict(row))
            return None

    async def list_all(self) -> list[Document]:
        """청크 수와 함께 모든 문서 목록을 조회합니다."""
        query = """
            SELECT d.*, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [Document.from_db_row(dict(row)) for row in rows]

    async def update(
        self,
        document_id: UUID,
        content: str,
        file_size: int,
    ) -> Optional[Document]:
        """문서 콘텐츠를 업데이트합니다."""
        query = """
            UPDATE documents
            SET content = $2, file_size = $3, updated_at = NOW()
            WHERE id = $1
            RETURNING id, filename, content, format, file_size, created_at, updated_at
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, document_id, content, file_size)
            if row:
                return Document.from_db_row(dict(row))
            return None

    async def delete(self, document_id: UUID) -> bool:
        """문서를 삭제합니다 (청크까지 연쇄 삭제)."""
        query = "DELETE FROM documents WHERE id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, document_id)
            return result == "DELETE 1"

    async def count(self) -> int:
        """총 문서 수를 계산합니다."""
        query = "SELECT COUNT(*) FROM documents"
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query)


class ChunkRepository:
    """청크 CRUD 작업을 위한 리포지토리."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_many(self, chunks: list[dict]) -> list[Chunk]:
        """문서에 대한 여러 청크를 생성합니다."""
        query = """
            INSERT INTO chunks (document_id, content, chunk_index, token_count, embedding, search_vector)
            VALUES ($1, $2, $3, $4, $5::vector, to_tsvector('simple', $2))
            RETURNING id, document_id, content, chunk_index, token_count, created_at
        """
        results = []
        async with self.pool.acquire() as conn:
            for chunk in chunks:
                row = await conn.fetchrow(
                    query,
                    chunk["document_id"],
                    chunk["content"],
                    chunk["chunk_index"],
                    chunk["token_count"],
                    chunk["embedding"],
                )
                results.append(Chunk.from_db_row(dict(row)))
        return results

    async def get_by_document_id(self, document_id: UUID) -> list[Chunk]:
        """문서의 모든 청크를 조회합니다."""
        query = """
            SELECT * FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, document_id)
            return [Chunk.from_db_row(dict(row)) for row in rows]

    async def delete_by_document_id(self, document_id: UUID) -> int:
        """문서의 모든 청크를 삭제합니다."""
        query = "DELETE FROM chunks WHERE document_id = $1"
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, document_id)
            # "DELETE N"에서 개수 추출
            return int(result.split()[1]) if result else 0

    async def dense_search(
        self, embedding: list[float], limit: int = 10
    ) -> list[Chunk]:
        """벡터 유사도로 청크를 검색합니다."""
        query = """
            SELECT c.*, d.filename,
                   1 - (c.embedding <=> $1::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> $1::vector
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, embedding, limit)
            return [Chunk.from_db_row(dict(row)) for row in rows]

    async def bm25_search(self, query_text: str, limit: int = 10) -> list[Chunk]:
        """BM25 (ts_rank)를 사용하여 청크를 검색합니다."""
        query = """
            SELECT c.*, d.filename,
                   ts_rank(c.search_vector, plainto_tsquery('simple', $1)) AS bm25_rank
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.search_vector @@ plainto_tsquery('simple', $1)
            ORDER BY bm25_rank DESC
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, query_text, limit)
            return [Chunk.from_db_row(dict(row)) for row in rows]

    async def trigram_search(self, query_text: str, limit: int = 10) -> list[Chunk]:
        """트라이그램 유사도를 사용하여 청크를 검색합니다."""
        query = """
            SELECT c.*, d.filename,
                   similarity(c.content, $1) AS trigram_sim
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.content % $1
            ORDER BY trigram_sim DESC
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, query_text, limit)
            return [Chunk.from_db_row(dict(row)) for row in rows]

    async def count(self) -> int:
        """총 청크 수를 계산합니다."""
        query = "SELECT COUNT(*) FROM chunks"
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query)
