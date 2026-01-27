"""문서 API 엔드포인트에 대한 계약 테스트."""

import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestDocumentsAPI:
    """/api/documents 엔드포인트에 대한 계약 테스트."""

    async def test_upload_txt_document(self, async_client: AsyncClient):
        """txt 파일로 POST /api/documents 테스트."""
        content = b"This is test content for the document."
        files = {"file": ("test.txt", io.BytesIO(content), "text/plain")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["filename"] == "test.txt"
        assert data["format"] == "txt"
        assert data["file_size"] == len(content)
        assert data["chunk_count"] >= 1
        assert "created_at" in data

    async def test_upload_md_document(self, async_client: AsyncClient):
        """마크다운 파일로 POST /api/documents 테스트."""
        content = b"# Title\n\nSome **markdown** content."
        files = {"file": ("readme.md", io.BytesIO(content), "text/markdown")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "readme.md"
        assert data["format"] == "md"

    async def test_upload_json_document(self, async_client: AsyncClient):
        """JSON 파일로 POST /api/documents 테스트."""
        content = b'{"title": "Test", "content": "JSON document"}'
        files = {"file": ("data.json", io.BytesIO(content), "application/json")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "data.json"
        assert data["format"] == "json"

    async def test_upload_invalid_format(self, async_client: AsyncClient):
        """지원되지 않는 형식으로 POST /api/documents 테스트."""
        content = b"Some binary content"
        files = {"file": ("image.png", io.BytesIO(content), "image/png")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "INVALID_FILE_FORMAT" in data["error"]

    async def test_upload_empty_file(self, async_client: AsyncClient):
        """빈 파일로 POST /api/documents 테스트."""
        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    async def test_upload_file_too_large(self, async_client: AsyncClient):
        """용량 제한을 초과하는 파일로 POST /api/documents 테스트."""
        # 10MB보다 큰 콘텐츠 생성
        content = b"x" * (10 * 1024 * 1024 + 1)
        files = {"file": ("large.txt", io.BytesIO(content), "text/plain")}

        response = await async_client.post("/api/documents", files=files)

        assert response.status_code == 413
        data = response.json()
        assert "error" in data

    async def test_list_documents_empty(self, async_client: AsyncClient):
        """문서가 없을 때 GET /api/documents 테스트."""
        response = await async_client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    async def test_list_documents_with_data(self, async_client: AsyncClient):
        """GET /api/documents가 업로드된 문서를 반환하는지 테스트."""
        # 먼저 문서를 업로드합니다
        content = b"Test document content"
        files = {"file": ("list_test.txt", io.BytesIO(content), "text/plain")}
        await async_client.post("/api/documents", files=files)

        # 그리고 목록 조회
        response = await async_client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(d["filename"] == "list_test.txt" for d in data["documents"])

    async def test_get_document_by_id(self, async_client: AsyncClient):
        """GET /api/documents/{id}가 문서 세부 정보를 반환하는지 테스트."""
        # 먼저 업로드
        content = b"Document for get test"
        files = {"file": ("get_test.txt", io.BytesIO(content), "text/plain")}
        upload_response = await async_client.post("/api/documents", files=files)
        doc_id = upload_response.json()["id"]

        # 그리고 조회
        response = await async_client.get(f"/api/documents/{doc_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == doc_id
        assert data["filename"] == "get_test.txt"
        assert "content_preview" in data

    async def test_get_document_not_found(self, async_client: AsyncClient):
        """존재하지 않는 ID로 GET /api/documents/{id} 테스트."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.get(f"/api/documents/{fake_id}")

        assert response.status_code == 404

    async def test_delete_document(self, async_client: AsyncClient):
        """DELETE /api/documents/{id}가 문서를 제거하는지 테스트."""
        # 먼저 업로드
        content = b"Document to delete"
        files = {"file": ("delete_test.txt", io.BytesIO(content), "text/plain")}
        upload_response = await async_client.post("/api/documents", files=files)
        doc_id = upload_response.json()["id"]

        # 삭제
        response = await async_client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 204

        # 삭제 확인
        get_response = await async_client.get(f"/api/documents/{doc_id}")
        assert get_response.status_code == 404

    async def test_delete_document_not_found(self, async_client: AsyncClient):
        """존재하지 않는 ID로 DELETE /api/documents/{id} 테스트."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await async_client.delete(f"/api/documents/{fake_id}")

        assert response.status_code == 404

    async def test_upload_duplicate_replaces(self, async_client: AsyncClient):
        """중복된 파일 이름 업로드가 기존 문서를 대체하는지 테스트."""
        filename = "duplicate_test.txt"

        # 먼저 업로드
        content1 = b"Original content"
        files1 = {"file": (filename, io.BytesIO(content1), "text/plain")}
        response1 = await async_client.post("/api/documents", files=files1)
        assert response1.status_code == 201
        id1 = response1.json()["id"]

        # 같은 파일 이름으로 두 번째 업로드
        content2 = b"Updated content"
        files2 = {"file": (filename, io.BytesIO(content2), "text/plain")}
        response2 = await async_client.post("/api/documents", files=files2)
        assert response2.status_code == 201

        # 대체되어야 함 (같거나 새로운 ID, 하지만 해당 파일 이름을 가진 문서는 하나뿐)
        list_response = await async_client.get("/api/documents")
        docs = list_response.json()["documents"]
        matching = [d for d in docs if d["filename"] == filename]
        assert len(matching) == 1
