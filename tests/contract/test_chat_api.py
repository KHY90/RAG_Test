"""채팅 API 엔드포인트에 대한 계약 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestChatAPI:
    """/api/chat 엔드포인트에 대한 계약 테스트."""

    async def test_chat_returns_answer_with_sources(self, async_client: AsyncClient):
        """POST /api/chat이 소스 참조와 함께 답변을 반환하는지 테스트합니다."""
        # 먼저 문서를 업로드합니다
        content = b"Python is a programming language created by Guido van Rossum."
        files = {"file": ("python_info.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        # 질문을 합니다
        response = await async_client.post(
            "/api/chat",
            json={"question": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()

        # openapi.yaml에 따라 응답 구조를 확인합니다
        assert "answer" in data
        assert "sources" in data
        assert "search_time_ms" in data
        assert "generation_time_ms" in data

        # 답변은 비어 있지 않아야 합니다
        assert len(data["answer"]) > 0

        # 소스는 리스트여야 합니다
        assert isinstance(data["sources"], list)

    async def test_chat_source_structure(self, async_client: AsyncClient):
        """소스 참조가 올바른 구조를 가지고 있는지 테스트합니다."""
        # 문서를 업로드합니다
        content = b"The capital of France is Paris. Paris is known for the Eiffel Tower."
        files = {"file": ("france.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "What is the capital of France?"}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["sources"]) > 0:
            source = data["sources"][0]
            # SourceReference 스키마에 따라 소스 구조를 확인합니다
            assert "document_id" in source
            assert "filename" in source
            assert "chunk_index" in source
            assert "content_preview" in source
            assert "relevance_score" in source

            # 관련성 점수는 0과 1 사이여야 합니다
            assert 0 <= source["relevance_score"] <= 1

    async def test_chat_with_top_k_parameter(self, async_client: AsyncClient):
        """POST /api/chat이 top_k 매개변수를 준수하는지 테스트합니다."""
        # Upload document
        content = b"Sample document content for testing top_k parameter."
        files = {"file": ("topk_test.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "Test query", "top_k": 3}
        )

        assert response.status_code == 200
        data = response.json()

        # 최대 top_k개의 소스를 반환해야 합니다
        assert len(data["sources"]) <= 3

    async def test_chat_empty_question_rejected(self, async_client: AsyncClient):
        """POST /api/chat이 빈 질문을 거부하는지 테스트합니다."""
        response = await async_client.post(
            "/api/chat",
            json={"question": ""}
        )

        # 빈 질문을 거부해야 합니다 (유효성 검사 오류)
        assert response.status_code == 422  # Validation error

    async def test_chat_question_too_long_rejected(self, async_client: AsyncClient):
        """POST /api/chat이 최대 길이를 초과하는 질문을 거부하는지 테스트합니다."""
        # 1000자를 초과하는 질문을 생성합니다
        long_question = "x" * 1001

        response = await async_client.post(
            "/api/chat",
            json={"question": long_question}
        )

        # 너무 긴 질문을 거부해야 합니다
        assert response.status_code == 422

    async def test_chat_with_korean_question(self, async_client: AsyncClient):
        """POST /api/chat이 한국어 텍스트를 올바르게 처리하는지 테스트합니다."""
        # 한국어 문서를 업로드합니다
        content = "인공지능은 컴퓨터 과학의 한 분야입니다. 머신러닝과 딥러닝이 포함됩니다.".encode("utf-8")
        files = {"file": ("korean.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "인공지능이란 무엇인가요?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    async def test_chat_timing_metrics_present(self, async_client: AsyncClient):
        """타이밍 메트릭이 반환되는지 테스트합니다."""
        content = b"Simple test content."
        files = {"file": ("timing_test.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "Test question"}
        )

        assert response.status_code == 200
        data = response.json()

        # 타이밍 메트릭은 양수여야 합니다
        assert data["search_time_ms"] >= 0
        assert data["generation_time_ms"] >= 0

    async def test_chat_no_documents_returns_appropriate_response(self, async_client: AsyncClient):
        """문서가 없을 때 POST /api/chat을 테스트합니다."""
        # 문서를 업로드하지 않고 질문을 합니다
        response = await async_client.post(
            "/api/chat",
            json={"question": "What is this about?"}
        )

        # "관련 내용 없음" 답변과 함께 200을 반환하거나
        # 구현 선택에 따라 404를 반환해야 합니다
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            # 답변은 관련 내용이 없음을 나타내야 합니다
            assert "answer" in data


@pytest.mark.asyncio
class TestSearchAPI:
    """/api/search 엔드포인트에 대한 계약 테스트 (5단계 미리보기)."""

    async def test_search_returns_results(self, async_client: AsyncClient):
        """POST /api/search가 검색 결과를 반환하는지 테스트합니다."""
        # Upload document
        content = b"Machine learning is a subset of artificial intelligence."
        files = {"file": ("ml_info.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/search",
            json={"query": "machine learning", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        # 응답 구조를 확인합니다
        assert "results" in data
        assert "search_time_ms" in data
        assert isinstance(data["results"], list)

    async def test_search_result_structure(self, async_client: AsyncClient):
        """검색 결과 항목이 올바른 구조를 가지고 있는지 테스트합니다."""
        content = b"Test content for search result structure verification."
        files = {"file": ("structure_test.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/search",
            json={"query": "test content", "search_type": "hybrid"}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 0:
            result = data["results"][0]
            # SearchResultItem 구조를 확인합니다
            assert "chunk_id" in result
            assert "document_id" in result
            assert "filename" in result
            assert "chunk_index" in result
            assert "content" in result
            assert "score" in result
