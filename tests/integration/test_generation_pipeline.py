"""질문 답변 생성 파이프라인에 대한 통합 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestGenerationPipeline:
    """질문 답변 파이프라인에 대한 엔드투엔드 테스트."""

    async def test_full_qa_pipeline_english(self, async_client: AsyncClient):
        """영어 콘텐츠로 전체 Q&A 파이프라인 테스트."""
        # 특정 콘텐츠가 있는 문서 업로드
        content = b"""
        Albert Einstein was a theoretical physicist born in Germany.
        He developed the theory of relativity, one of the two pillars of modern physics.
        Einstein received the Nobel Prize in Physics in 1921.
        His famous equation E=mc^2 relates energy and mass.
        """
        files = {"file": ("einstein.txt", io.BytesIO(content), "text/plain")}
        upload_response = await async_client.post("/api/documents", files=files)
        assert upload_response.status_code == 201

        # 콘텐츠에 대해 질문하기
        chat_response = await async_client.post(
            "/api/chat",
            json={"question": "When did Einstein receive the Nobel Prize?"}
        )

        assert chat_response.status_code == 200
        data = chat_response.json()

        # 답변에 관련 정보가 포함되어야 합니다
        assert "answer" in data
        # 답변에 1921년 또는 노벨상이 언급되어야 합니다
        answer_lower = data["answer"].lower()
        assert "1921" in answer_lower or "nobel" in answer_lower

        # 소스에 문서가 포함되어야 합니다
        assert len(data["sources"]) > 0
        assert any("einstein" in s["filename"].lower() for s in data["sources"])

    async def test_full_qa_pipeline_korean(self, async_client: AsyncClient):
        """한국어 콘텐츠로 전체 Q&A 파이프라인 테스트."""
        content = """
        대한민국의 수도는 서울입니다.
        서울은 한강을 중심으로 남북으로 나뉘어 있습니다.
        서울의 인구는 약 천만 명입니다.
        서울에는 경복궁, 남산타워 등 유명한 관광지가 있습니다.
        """.encode("utf-8")
        files = {"file": ("seoul.txt", io.BytesIO(content), "text/plain")}
        upload_response = await async_client.post("/api/documents", files=files)
        assert upload_response.status_code == 201

        # 한국어로 질문하기
        chat_response = await async_client.post(
            "/api/chat",
            json={"question": "대한민국의 수도는 어디인가요?"}
        )

        assert chat_response.status_code == 200
        data = chat_response.json()

        # 답변이 존재해야 합니다
        assert "answer" in data
        assert len(data["answer"]) > 0

    async def test_qa_with_multiple_documents(self, async_client: AsyncClient):
        """여러 문서를 인덱싱했을 때의 Q&A 테스트."""
        # 여러 문서 업로드
        doc1 = b"Python is a high-level programming language known for its simplicity."
        doc2 = b"JavaScript is the language of the web, running in browsers."
        doc3 = b"Rust is a systems programming language focused on safety."

        for name, content in [("python.txt", doc1), ("javascript.txt", doc2), ("rust.txt", doc3)]:
            files = {"file": (name, io.BytesIO(content), "text/plain")}
            response = await async_client.post("/api/documents", files=files)
            assert response.status_code == 201

        # 파이썬에 대해 질문하기
        response = await async_client.post(
            "/api/chat",
            json={"question": "What is Python known for?"}
        )

        assert response.status_code == 200
        data = response.json()

        # 파이썬 문서에서 검색해야 합니다
        assert len(data["sources"]) > 0

    async def test_qa_returns_relevant_sources(self, async_client: AsyncClient):
        """Q&A가 질문과 관련된 소스를 반환하는지 테스트."""
        # 문서 업로드
        content = b"""
        Chapter 1: Introduction to Machine Learning
        Machine learning is a method of data analysis that automates analytical model building.

        Chapter 2: Neural Networks
        Neural networks are computing systems inspired by biological neural networks.

        Chapter 3: Deep Learning
        Deep learning is part of machine learning based on artificial neural networks.
        """
        files = {"file": ("ml_book.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "What is deep learning?"}
        )

        assert response.status_code == 200
        data = response.json()

        # 소스에 관련 콘텐츠가 포함되어야 합니다
        if len(data["sources"]) > 0:
            # 적어도 하나의 소스에 딥러닝 또는 신경망이 언급되어야 합니다
            source_contents = " ".join([s["content_preview"].lower() for s in data["sources"]])
            assert "deep" in source_contents or "neural" in source_contents or "learning" in source_contents

    async def test_qa_handles_no_relevant_results(self, async_client: AsyncClient):
        """Q&A가 관련 콘텐츠가 없는 질문을 적절하게 처리하는지 테스트."""
        # 관련 없는 문서 업로드
        content = b"This document is about cooking recipes and kitchen equipment."
        files = {"file": ("cooking.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        # 완전히 관련 없는 질문하기
        response = await async_client.post(
            "/api/chat",
            json={"question": "What is quantum computing?"}
        )

        # 적절한 답변과 함께 200을 반환해야 합니다 (관련 정보 없음을 나타낼 수 있음)
        assert response.status_code in [200, 404]

    async def test_qa_respects_top_k_sources(self, async_client: AsyncClient):
        """top_k 매개변수가 반환된 소스를 제한하는지 테스트."""
        # 여러 청크를 생성할 대용량 문서 업로드
        content = (b"This is test content for chunking. " * 200)
        files = {"file": ("large_doc.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        # 특정 top_k로 요청
        response = await async_client.post(
            "/api/chat",
            json={"question": "What is this about?", "top_k": 2}
        )

        assert response.status_code == 200
        data = response.json()

        # 최대 2개의 소스를 반환해야 합니다
        assert len(data["sources"]) <= 2

    async def test_qa_timing_is_reasonable(self, async_client: AsyncClient):
        """Q&A 타이밍 메트릭이 합리적인 범위 내에 있는지 테스트."""
        content = b"Short test content for timing verification."
        files = {"file": ("timing.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "What is this?"}
        )

        assert response.status_code == 200
        data = response.json()

        # 타이밍은 양수여야 합니다 (밀리초 단위)
        assert data["search_time_ms"] >= 0
        assert data["generation_time_ms"] >= 0

        # 검색은 합리적으로 빨라야 합니다 (5초 미만)
        assert data["search_time_ms"] < 5000

    async def test_qa_with_json_document(self, async_client: AsyncClient):
        """JSON 문서 콘텐츠로 Q&A 테스트."""
        content = b'''
        {
            "product": {
                "name": "SuperWidget",
                "description": "An amazing widget that does everything",
                "features": ["fast", "reliable", "affordable"],
                "price": 99.99
            }
        }
        '''
        files = {"file": ("product.json", io.BytesIO(content), "application/json")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "What is the SuperWidget?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    async def test_qa_with_markdown_document(self, async_client: AsyncClient):
        """마크다운 문서 콘텐츠로 Q&A 테스트."""
        content = b"""
        # Project Documentation

        ## Overview
        This project implements a hybrid search system.

        ## Features
        - Dense search using embeddings
        - Sparse search using BM25
        - Trigram similarity search

        ## Installation
        Run `pip install -r requirements.txt`
        """
        files = {"file": ("readme.md", io.BytesIO(content), "text/markdown")}
        await async_client.post("/api/documents", files=files)

        response = await async_client.post(
            "/api/chat",
            json={"question": "What search methods does the project use?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
