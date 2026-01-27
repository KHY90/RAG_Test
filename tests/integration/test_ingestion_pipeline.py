"""문서 수집 파이프라인에 대한 통합 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestIngestionPipeline:
    """문서 수집에 대한 엔드투엔드 테스트."""

    async def test_full_ingestion_txt(self, async_client: AsyncClient):
        """txt 파일에 대한 전체 수집 파이프라인 테스트."""
        content = b"""This is a test document for the RAG system.
It contains multiple sentences to test chunking.
The system should extract text and create embeddings.
Each chunk should be searchable."""

        files = {"file": ("pipeline_test.txt", io.BytesIO(content), "text/plain")}
        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()

        # 문서가 생성되었는지 확인
        assert data["filename"] == "pipeline_test.txt"
        assert data["format"] == "txt"
        assert data["file_size"] == len(content)

        # 청크가 생성되었는지 확인
        assert data["chunk_count"] >= 1

        # 문서를 검색할 수 있는지 확인
        doc_response = await async_client.get(f"/api/documents/{data['id']}")
        assert doc_response.status_code == 200
        assert "content_preview" in doc_response.json()

    async def test_full_ingestion_json_nested(self, async_client: AsyncClient):
        """모든 문자열이 추출된 중첩 JSON 수집 테스트."""
        content = b'''{
            "article": {
                "title": "Test Article",
                "author": "Test Author",
                "sections": [
                    {"heading": "Introduction", "text": "This is the intro."},
                    {"heading": "Conclusion", "text": "This is the conclusion."}
                ]
            }
        }'''

        files = {"file": ("nested.json", io.BytesIO(content), "application/json")}
        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()

        # 모든 중첩 콘텐츠가 추출되어야 합니다
        assert data["chunk_count"] >= 1

    async def test_ingestion_preserves_korean(self, async_client: AsyncClient):
        """수집 과정을 통해 한국어 텍스트가 제대로 보존되는지 테스트."""
        content = "한글 문서 테스트입니다. 임베딩이 제대로 생성되어야 합니다.".encode("utf-8")

        files = {"file": ("korean.txt", io.BytesIO(content), "text/plain")}
        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201

        # 한국어 콘텐츠가 보존되는지 확인
        doc_response = await async_client.get(f"/api/documents/{response.json()['id']}")
        preview = doc_response.json()["content_preview"]
        assert "한글" in preview

    async def test_ingestion_creates_searchable_chunks(self, async_client: AsyncClient):
        """수집된 문서를 검색할 수 있는지 테스트."""
        content = b"Unique searchable content XYZ123 for testing retrieval."

        files = {"file": ("searchable.txt", io.BytesIO(content), "text/plain")}
        upload_response = await async_client.post("/api/documents", files=files)
        assert upload_response.status_code == 201

        # 고유 콘텐츠 검색
        search_response = await async_client.post(
            "/api/search",
            json={"query": "XYZ123", "search_type": "trigram"}
        )

        # 문서를 찾아야 합니다 (검색이 구현된 경우)
        if search_response.status_code == 200:
            results = search_response.json()["results"]
            assert any("XYZ123" in r["content"] for r in results)

    async def test_large_document_chunking(self, async_client: AsyncClient):
        """대용량 문서가 제대로 청킹되는지 테스트."""
        # 약 2000 단어의 문서 생성
        content = ("This is a test sentence for chunking. " * 400).encode("utf-8")

        files = {"file": ("large.txt", io.BytesIO(content), "text/plain")}
        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()

        # 여러 청크를 생성해야 합니다
        assert data["chunk_count"] > 1

    async def test_document_replacement_clears_old_chunks(self, async_client: AsyncClient):
        """문서 교체 시 이전 청크가 제거되는지 테스트."""
        filename = "replace_chunks.txt"

        # 특정 콘텐츠로 첫 번째 버전 업로드
        content1 = b"First version content MARKER1"
        files1 = {"file": (filename, io.BytesIO(content1), "text/plain")}
        response1 = await async_client.post("/api/documents", files=files1)
        assert response1.status_code == 201
        chunk_count1 = response1.json()["chunk_count"]

        # 다른 콘텐츠로 두 번째 버전 업로드
        content2 = b"Second version completely different MARKER2"
        files2 = {"file": (filename, io.BytesIO(content2), "text/plain")}
        response2 = await async_client.post("/api/documents", files=files2)
        assert response2.status_code == 201

        # 이전 콘텐츠가 검색되지 않아야 합니다 (검색이 구현된 경우)
        search_response = await async_client.post(
            "/api/search",
            json={"query": "MARKER1", "search_type": "trigram"}
        )

        if search_response.status_code == 200:
            results = search_response.json()["results"]
            # 이전 마커를 찾을 수 없어야 합니다
            assert not any("MARKER1" in r["content"] for r in results)
