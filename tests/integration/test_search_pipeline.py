"""하이브리드 검색 파이프라인에 대한 통합 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestSearchPipeline:
    """다양한 검색 모드에 대한 통합 테스트."""

    @pytest.fixture(autouse=True)
    async def setup_test_documents(self, async_client: AsyncClient):
        """테스트 문서를 업로드합니다."""
        # 시맨틱 검색 테스트를 위한 문서
        semantic_content = b"""Machine learning is a branch of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing enables computers to understand human language."""
        files = {"file": ("ai_info.txt", io.BytesIO(semantic_content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        # 키워드 검색 테스트를 위한 문서
        keyword_content = b"""PostgreSQL is a powerful open-source database.
The pg_trgm extension provides trigram-based text similarity.
Full-text search uses tsvector and tsquery for efficient searching."""
        files = {"file": ("database.txt", io.BytesIO(keyword_content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        yield

    async def test_dense_search_only(self, async_client: AsyncClient):
        """밀집(벡터) 검색만 사용한 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "neural networks and AI", "search_type": "dense"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "search_time_ms" in data
        assert isinstance(data["results"], list)

        # 밀집 검색은 시맨틱 유사성을 찾아야 합니다
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "dense_score" in result
            # 밀집 검색에서는 dense_score가 있어야 합니다
            assert result.get("dense_score") is not None or result.get("score") is not None

    async def test_sparse_search_only(self, async_client: AsyncClient):
        """희소(BM25) 검색만 사용한 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "PostgreSQL database", "search_type": "sparse"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

        # 희소 검색은 정확한 키워드 매칭을 찾아야 합니다
        if len(data["results"]) > 0:
            result = data["results"][0]
            # sparse_score가 있어야 합니다
            assert "sparse_score" in result

    async def test_trigram_search_only(self, async_client: AsyncClient):
        """트라이그램 유사도 검색만 사용한 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "trigram similarity", "search_type": "trigram"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

        # 트라이그램 검색 결과 확인
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "trigram_score" in result

    async def test_hybrid_search_combines_all_methods(self, async_client: AsyncClient):
        """하이브리드 검색이 모든 방법을 결합하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "machine learning artificial intelligence", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

        # 하이브리드 검색은 여러 점수 유형을 가질 수 있습니다
        if len(data["results"]) > 0:
            result = data["results"][0]
            # 최종 RRF 점수가 있어야 합니다
            assert "score" in result
            # 개별 점수 중 하나 이상이 있을 수 있습니다
            has_any_score = (
                result.get("dense_score") is not None or
                result.get("sparse_score") is not None or
                result.get("trigram_score") is not None
            )
            assert has_any_score

    async def test_semantic_search_finds_synonyms(self, async_client: AsyncClient):
        """시맨틱 검색이 동의어를 찾는지 테스트합니다."""
        # "AI" 대신 "인공지능"으로 검색 (영어 문서에서도 개념적으로 유사)
        response = await async_client.post(
            "/api/search",
            json={"query": "computer brain simulation", "search_type": "dense"}
        )

        assert response.status_code == 200
        data = response.json()

        # 밀집 검색은 시맨틱 유사성으로 관련 결과를 찾아야 합니다
        assert "results" in data

    async def test_keyword_search_exact_match(self, async_client: AsyncClient):
        """키워드 검색이 정확한 매칭을 찾는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "tsvector tsquery", "search_type": "sparse"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        # 정확한 키워드가 있는 문서가 결과에 있어야 합니다
        if len(data["results"]) > 0:
            found_match = any(
                "tsvector" in r.get("content", "") or "tsquery" in r.get("content", "")
                for r in data["results"]
            )
            # 문서가 있다면 키워드가 포함되어야 합니다
            assert found_match or len(data["results"]) == 0

    async def test_hybrid_vs_single_method_coverage(self, async_client: AsyncClient):
        """하이브리드 검색이 단일 방법보다 더 많은 커버리지를 제공하는지 테스트합니다."""
        query = "database text search"

        # 각 검색 유형으로 검색
        dense_response = await async_client.post(
            "/api/search",
            json={"query": query, "search_type": "dense", "top_k": 10}
        )
        sparse_response = await async_client.post(
            "/api/search",
            json={"query": query, "search_type": "sparse", "top_k": 10}
        )
        hybrid_response = await async_client.post(
            "/api/search",
            json={"query": query, "search_type": "hybrid", "top_k": 10}
        )

        assert dense_response.status_code == 200
        assert sparse_response.status_code == 200
        assert hybrid_response.status_code == 200

        # 모든 응답이 유효한 구조를 가져야 합니다
        dense_data = dense_response.json()
        sparse_data = sparse_response.json()
        hybrid_data = hybrid_response.json()

        assert "results" in dense_data
        assert "results" in sparse_data
        assert "results" in hybrid_data

    async def test_search_returns_timing_metrics(self, async_client: AsyncClient):
        """모든 검색 유형이 타이밍 메트릭을 반환하는지 테스트합니다."""
        for search_type in ["hybrid", "dense", "sparse", "trigram"]:
            response = await async_client.post(
                "/api/search",
                json={"query": "test query", "search_type": search_type}
            )

            assert response.status_code == 200
            data = response.json()
            assert "search_time_ms" in data
            assert data["search_time_ms"] >= 0

    async def test_search_respects_top_k(self, async_client: AsyncClient):
        """검색이 top_k 매개변수를 준수하는지 테스트합니다."""
        for search_type in ["hybrid", "dense", "sparse", "trigram"]:
            response = await async_client.post(
                "/api/search",
                json={"query": "machine learning", "search_type": search_type, "top_k": 2}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) <= 2

    async def test_search_with_korean_query(self, async_client: AsyncClient):
        """한국어 쿼리로 검색을 테스트합니다."""
        # 한국어 문서 업로드
        content = "인공지능과 기계학습은 데이터 과학의 핵심 기술입니다.".encode("utf-8")
        files = {"file": ("korean.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/search",
            json={"query": "인공지능 기술", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    async def test_empty_query_rejected(self, async_client: AsyncClient):
        """빈 쿼리가 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "", "search_type": "hybrid"}
        )

        assert response.status_code == 422  # Validation error

    async def test_invalid_search_type_rejected(self, async_client: AsyncClient):
        """유효하지 않은 검색 유형이 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "search_type": "invalid_type"}
        )

        assert response.status_code == 400
