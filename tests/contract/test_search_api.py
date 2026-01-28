"""검색 API 엔드포인트에 대한 계약 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestSearchAPIContract:
    """POST /api/search 엔드포인트에 대한 계약 테스트."""

    @pytest.fixture(autouse=True)
    async def setup_documents(self, async_client: AsyncClient):
        """테스트 문서를 업로드합니다."""
        content = b"""Hybrid search combines dense and sparse retrieval methods.
Dense retrieval uses vector embeddings for semantic similarity.
Sparse retrieval uses keyword matching like BM25.
Reciprocal Rank Fusion merges the results from different methods."""
        files = {"file": ("search_test.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)
        yield

    async def test_search_endpoint_exists(self, async_client: AsyncClient):
        """POST /api/search 엔드포인트가 존재하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test query"}
        )
        # 404가 아니어야 합니다
        assert response.status_code != 404

    async def test_search_response_schema(self, async_client: AsyncClient):
        """검색 응답이 올바른 스키마를 따르는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "hybrid search", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        # SearchResponse 스키마 확인
        assert "results" in data
        assert "search_time_ms" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["search_time_ms"], (int, float))

    async def test_search_result_item_schema(self, async_client: AsyncClient):
        """각 검색 결과 항목이 올바른 스키마를 따르는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "dense retrieval", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 0:
            result = data["results"][0]

            # SearchResultItem 필수 필드
            assert "chunk_id" in result
            assert "document_id" in result
            assert "filename" in result
            assert "chunk_index" in result
            assert "content" in result
            assert "score" in result

            # SearchResultItem 선택 필드 (개별 점수)
            assert "dense_score" in result
            assert "sparse_score" in result
            assert "trigram_score" in result

    async def test_search_type_hybrid(self, async_client: AsyncClient):
        """search_type=hybrid로 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "semantic similarity", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    async def test_search_type_dense(self, async_client: AsyncClient):
        """search_type=dense로 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "vector embeddings", "search_type": "dense"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # 밀집 검색 결과는 dense_score를 가져야 합니다
        if len(data["results"]) > 0:
            assert data["results"][0].get("dense_score") is not None

    async def test_search_type_sparse(self, async_client: AsyncClient):
        """search_type=sparse로 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "keyword matching BM25", "search_type": "sparse"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # 희소 검색 결과는 sparse_score를 가져야 합니다
        if len(data["results"]) > 0:
            assert data["results"][0].get("sparse_score") is not None

    async def test_search_type_trigram(self, async_client: AsyncClient):
        """search_type=trigram으로 검색을 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "Reciprocal Rank", "search_type": "trigram"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

        # 트라이그램 검색 결과는 trigram_score를 가져야 합니다
        if len(data["results"]) > 0:
            assert data["results"][0].get("trigram_score") is not None

    async def test_search_type_invalid(self, async_client: AsyncClient):
        """유효하지 않은 search_type이 오류를 반환하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "search_type": "invalid"}
        )

        assert response.status_code == 400

    async def test_search_default_search_type(self, async_client: AsyncClient):
        """search_type이 지정되지 않으면 기본값이 적용되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "hybrid search"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    async def test_search_top_k_parameter(self, async_client: AsyncClient):
        """top_k 매개변수가 올바르게 작동하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "retrieval", "top_k": 2}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    async def test_search_top_k_min_value(self, async_client: AsyncClient):
        """top_k 최소값(1)이 작동하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "top_k": 1}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 1

    async def test_search_top_k_max_value(self, async_client: AsyncClient):
        """top_k 최대값(50)이 작동하는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "top_k": 50}
        )

        assert response.status_code == 200

    async def test_search_top_k_invalid_zero(self, async_client: AsyncClient):
        """top_k=0이 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "top_k": 0}
        )

        assert response.status_code == 422

    async def test_search_top_k_invalid_negative(self, async_client: AsyncClient):
        """음수 top_k가 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "top_k": -1}
        )

        assert response.status_code == 422

    async def test_search_top_k_invalid_too_large(self, async_client: AsyncClient):
        """top_k가 최대값을 초과하면 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test", "top_k": 100}
        )

        assert response.status_code == 422

    async def test_search_empty_query_rejected(self, async_client: AsyncClient):
        """빈 쿼리가 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": ""}
        )

        assert response.status_code == 422

    async def test_search_query_too_long_rejected(self, async_client: AsyncClient):
        """너무 긴 쿼리가 거부되는지 테스트합니다."""
        long_query = "x" * 1001
        response = await async_client.post(
            "/api/search",
            json={"query": long_query}
        )

        assert response.status_code == 422

    async def test_search_missing_query_rejected(self, async_client: AsyncClient):
        """쿼리 필드가 없으면 거부되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={}
        )

        assert response.status_code == 422

    async def test_search_scores_are_numeric(self, async_client: AsyncClient):
        """모든 점수가 숫자인지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "hybrid", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            assert isinstance(result["score"], (int, float))
            if result.get("dense_score") is not None:
                assert isinstance(result["dense_score"], (int, float))
            if result.get("sparse_score") is not None:
                assert isinstance(result["sparse_score"], (int, float))
            if result.get("trigram_score") is not None:
                assert isinstance(result["trigram_score"], (int, float))

    async def test_search_time_ms_is_positive(self, async_client: AsyncClient):
        """검색 시간이 양수인지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_time_ms"] >= 0

    async def test_search_with_unicode_query(self, async_client: AsyncClient):
        """유니코드 쿼리가 올바르게 처리되는지 테스트합니다."""
        # 한국어
        response = await async_client.post(
            "/api/search",
            json={"query": "검색 테스트"}
        )
        assert response.status_code == 200

        # 일본어
        response = await async_client.post(
            "/api/search",
            json={"query": "検索テスト"}
        )
        assert response.status_code == 200

        # 이모지
        response = await async_client.post(
            "/api/search",
            json={"query": "test 🔍"}
        )
        assert response.status_code == 200

    async def test_search_results_ordered_by_score(self, async_client: AsyncClient):
        """검색 결과가 점수 내림차순으로 정렬되는지 테스트합니다."""
        response = await async_client.post(
            "/api/search",
            json={"query": "hybrid dense sparse", "search_type": "hybrid", "top_k": 10}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 1:
            scores = [r["score"] for r in data["results"]]
            assert scores == sorted(scores, reverse=True)

    async def test_search_chunk_id_is_uuid(self, async_client: AsyncClient):
        """chunk_id가 유효한 UUID 형식인지 테스트합니다."""
        import uuid

        response = await async_client.post(
            "/api/search",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            # UUID 형식 검증
            try:
                uuid.UUID(str(result["chunk_id"]))
            except ValueError:
                pytest.fail(f"Invalid UUID format: {result['chunk_id']}")

    async def test_search_document_id_is_uuid(self, async_client: AsyncClient):
        """document_id가 유효한 UUID 형식인지 테스트합니다."""
        import uuid

        response = await async_client.post(
            "/api/search",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            try:
                uuid.UUID(str(result["document_id"]))
            except ValueError:
                pytest.fail(f"Invalid UUID format: {result['document_id']}")
