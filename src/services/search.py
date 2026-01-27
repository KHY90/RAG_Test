"""하이브리드 검색 및 RRF 융합을 사용한 검색 서비스."""

from typing import Any
from uuid import UUID

from src.config import settings
from src.db.repositories import ChunkRepository
from src.services.embedding import EmbeddingService


def reciprocal_rank_fusion(
    rankings: list[list[tuple[Any, float]]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion을 적용하여 여러 순위 목록을 결합합니다.

    RRF 공식: 각 순위 i에 대해 score(d) = sum(1 / (k + rank_i(d)))

    인수:
        rankings: 순위 목록의 리스트, 각 순위는 관련성 순으로 정렬된
                  (document_id, original_score) 튜플의 리스트입니다.
        k: RRF 상수 (기본값 60). k가 작을수록 상위 순위에 더 많은 가중치를 둡니다.

    반환값:
        RRF 점수 내림차순으로 정렬된 (document_id, rrf_score) 튜플 목록으로서의 결합된 순위
    """
    if not rankings:
        return []

    # 각 문서에 대한 RRF 점수 계산
    rrf_scores: dict[Any, float] = {}

    for ranking in rankings:
        if not ranking:
            continue
        for rank, (doc_id, _original_score) in enumerate(ranking):
            # RRF 공식: 1 / (k + rank + 1)
            # 순위는 0부터 시작하므로 순위 0 -> 1/(k+1)
            rrf_contribution = 1.0 / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

    # RRF 점수 내림차순으로 정렬
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results


class SearchResult:
    """모든 메타데이터가 포함된 검색 결과."""

    def __init__(
        self,
        chunk_id: UUID,
        document_id: UUID,
        filename: str,
        chunk_index: int,
        content: str,
        score: float,
        dense_score: float | None = None,
        sparse_score: float | None = None,
        trigram_score: float | None = None,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.filename = filename
        self.chunk_index = chunk_index
        self.content = content
        self.score = score
        self.dense_score = dense_score
        self.sparse_score = sparse_score
        self.trigram_score = trigram_score


class SearchService:
    """밀집, 희소 및 트라이그램 방식을 결합한 하이브리드 검색을 위한 서비스."""

    def __init__(
        self,
        chunk_repo: ChunkRepository,
        embedding_service: EmbeddingService,
    ):
        self.chunk_repo = chunk_repo
        self.embedding_service = embedding_service
        self.rrf_k = settings.rrf_k

    async def dense_search(
        self, query: str, limit: int = 10
    ) -> list[tuple[UUID, float, dict]]:
        """밀집 (벡터 유사도) 검색을 수행합니다.

        인수:
            query: 검색 쿼리 텍스트
            limit: 최대 결과 수

        반환값:
            (chunk_id, score, chunk_data) 튜플 목록
        """
        # 쿼리 임베딩 가져오기
        query_embedding = self.embedding_service.encode_query(query)

        # 데이터베이스에서 검색
        chunks = await self.chunk_repo.dense_search(query_embedding, limit)

        results = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": getattr(chunk, "filename", ""),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            # 유사도는 쿼리에서 청크에 저장됩니다
            score = getattr(chunk, "similarity", 0.0)
            results.append((chunk.id, score, chunk_data))

        return results

    async def bm25_search(
        self, query: str, limit: int = 10
    ) -> list[tuple[UUID, float, dict]]:
        """BM25 (희소) 검색을 수행합니다.

        인수:
            query: 검색 쿼리 텍스트
            limit: 최대 결과 수

        반환값:
            (chunk_id, score, chunk_data) 튜플 목록
        """
        chunks = await self.chunk_repo.bm25_search(query, limit)

        results = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": getattr(chunk, "filename", ""),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            score = getattr(chunk, "bm25_rank", 0.0)
            results.append((chunk.id, score, chunk_data))

        return results

    async def trigram_search(
        self, query: str, limit: int = 10
    ) -> list[tuple[UUID, float, dict]]:
        """트라이그램 유사도 검색을 수행합니다.

        인수:
            query: 검색 쿼리 텍스트
            limit: 최대 결과 수

        반환값:
            (chunk_id, score, chunk_data) 튜플 목록
        """
        chunks = await self.chunk_repo.trigram_search(query, limit)

        results = []
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "filename": getattr(chunk, "filename", ""),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            score = getattr(chunk, "trigram_sim", 0.0)
            results.append((chunk.id, score, chunk_data))

        return results

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid",
    ) -> list[SearchResult]:
        """여러 방법을 결합하여 하이브리드 검색을 수행합니다.

        인수:
            query: 검색 쿼리 텍스트
            limit: 최대 결과 수
            search_type: 검색 유형 - "hybrid", "dense", "sparse", "trigram"

        반환값:
            관련성 순으로 정렬된 SearchResult 객체 목록
        """
        # search_type에 따라 각 검색 방법에서 결과 가져오기
        dense_results = []
        sparse_results = []
        trigram_results = []

        # RRF 융합을 위해 더 많은 결과 가져오기
        fetch_limit = limit * 3

        if search_type in ("hybrid", "dense"):
            dense_results = await self.dense_search(query, fetch_limit)

        if search_type in ("hybrid", "sparse"):
            sparse_results = await self.bm25_search(query, fetch_limit)

        if search_type in ("hybrid", "trigram"):
            trigram_results = await self.trigram_search(query, fetch_limit)

        # 청크 데이터 조회 생성
        chunk_data_map: dict[UUID, dict] = {}

        # 밀집 결과 처리
        dense_ranking = []
        for chunk_id, score, chunk_data in dense_results:
            dense_ranking.append((chunk_id, score))
            if chunk_id not in chunk_data_map:
                chunk_data_map[chunk_id] = chunk_data
            chunk_data_map[chunk_id]["dense_score"] = score

        # 희소 결과 처리
        sparse_ranking = []
        for chunk_id, score, chunk_data in sparse_results:
            sparse_ranking.append((chunk_id, score))
            if chunk_id not in chunk_data_map:
                chunk_data_map[chunk_id] = chunk_data
            chunk_data_map[chunk_id]["sparse_score"] = score

        # 트라이그램 결과 처리
        trigram_ranking = []
        for chunk_id, score, chunk_data in trigram_results:
            trigram_ranking.append((chunk_id, score))
            if chunk_id not in chunk_data_map:
                chunk_data_map[chunk_id] = chunk_data
            chunk_data_map[chunk_id]["trigram_score"] = score

        # 하이브리드인 경우 RRF 적용, 그렇지 않으면 단일 순위 사용
        if search_type == "hybrid":
            rankings = [r for r in [dense_ranking, sparse_ranking, trigram_ranking] if r]
            fused_results = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        elif search_type == "dense":
            fused_results = [(cid, score) for cid, score, _ in dense_results]
        elif search_type == "sparse":
            fused_results = [(cid, score) for cid, score, _ in sparse_results]
        elif search_type == "trigram":
            fused_results = [(cid, score) for cid, score, _ in trigram_results]
        else:
            fused_results = []

        # 최종 결과 생성
        results = []
        for chunk_id, score in fused_results[:limit]:
            if chunk_id not in chunk_data_map:
                continue

            data = chunk_data_map[chunk_id]
            results.append(SearchResult(
                chunk_id=data["chunk_id"],
                document_id=data["document_id"],
                filename=data.get("filename", ""),
                chunk_index=data["chunk_index"],
                content=data["content"],
                score=score,
                dense_score=data.get("dense_score"),
                sparse_score=data.get("sparse_score"),
                trigram_score=data.get("trigram_score"),
            ))

        return results
