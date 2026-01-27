"""텍스트 추출, 청킹 및 임베딩을 위한 문서 수집 서비스."""

import json
from typing import Any
from uuid import UUID

from transformers import AutoTokenizer

from src.config import settings
from src.db.repositories import DocumentRepository, ChunkRepository
from src.services.embedding import EmbeddingService


# 청킹을 위한 전역 토크나이저
_tokenizer = None


def get_tokenizer():
    """청킹을 위한 토크나이저를 가져오거나 생성합니다."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    return _tokenizer


def extract_txt(content: str) -> str:
    """txt 파일 콘텐츠에서 텍스트를 추출합니다.

    인수:
        content: 원본 txt 파일 콘텐츠

    반환값:
        추출된 텍스트 (txt 파일의 경우 변경 없음)
    """
    return content


def extract_md(content: str) -> str:
    """마크다운 파일 콘텐츠에서 텍스트를 추출합니다.

    의미론적 검색이 마크다운 서식을 처리할 수 있기 때문입니다.
    인덱싱을 위해 마크다운 콘텐츠를 그대로 보존합니다.

    인수:
        content: 원본 마크다운 파일 콘텐츠

    반환값:
        추출된 텍스트 (마크다운 보존됨)
    """
    return content


def extract_json(content: str) -> str:
    """JSON 콘텐츠에서 모든 문자열 값을 재귀적으로 추출합니다.

    인수:
        content: 원본 JSON 파일 콘텐츠

    반환값:
        공백으로 연결된 모든 문자열 값

    예외:
        ValueError: 콘텐츠가 유효한 JSON이 아닌 경우
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    strings = []
    _extract_strings_recursive(data, strings)
    return " ".join(strings)


def _extract_strings_recursive(obj: Any, strings: list[str]) -> None:
    """JSON 객체에서 모든 문자열 값을 재귀적으로 추출합니다.

    인수:
        obj: JSON 객체 (dict, list 또는 기본형)
        strings: 문자열을 추가할 목록
    """
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _extract_strings_recursive(value, strings)
    elif isinstance(obj, list):
        for item in obj:
            _extract_strings_recursive(item, strings)
    # 숫자, 불리언, None 무시


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict]:
    """텍스트를 중복하여 청크로 분할합니다.

    인수:
        text: 청킹할 텍스트
        chunk_size: 청크당 최대 토큰 수 (설정의 기본값 사용)
        overlap: 청크 간 토큰 중복 (설정의 기본값 사용)

    반환값:
        content, chunk_index, token_count가 포함된 청크 사전 목록
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    if not text or not text.strip():
        return []

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) == 0:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_content = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({
            "content": chunk_content,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
        })

        # 중복을 고려하여 시작 지점을 앞으로 이동
        start += chunk_size - overlap
        chunk_index += 1

        # 엣지 케이스에 대한 무한 루프 방지
        if start >= len(tokens) or (end == len(tokens) and start < end):
            break

    return chunks


class IngestionService:
    """문서 처리 및 저장을 위한 서비스."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        embedding_service: EmbeddingService,
    ):
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.embedding_service = embedding_service

    async def process_document(
        self,
        filename: str,
        content: bytes,
        format: str,
    ) -> dict:
        """문서 처리: 텍스트 추출, 청킹, 임베딩 및 저장.

        인수:
            filename: 원본 파일 이름
            content: 원본 파일 콘텐츠 바이트
            format: 파일 형식 (txt, md, json)

        반환값:
            문서 정보와 청크 수가 포함된 사전
        """
        # 콘텐츠 디코딩
        text_content = content.decode("utf-8")

        # 형식에 따라 텍스트 추출
        if format == "txt":
            extracted_text = extract_txt(text_content)
        elif format == "md":
            extracted_text = extract_md(text_content)
        elif format == "json":
            extracted_text = extract_json(text_content)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # 같은 파일 이름을 가진 기존 문서 확인
        from src.config import settings
        existing = await self.document_repo.get_by_filename(filename, settings.chunk_table)
        if existing:
            # 기존 문서 및 해당 청크 삭제
            await self.document_repo.delete(existing.id)

        # 문서 생성
        document = await self.document_repo.create(
            filename=filename,
            content=extracted_text,
            format=format,
            file_size=len(content),
        )

        # 텍스트 청킹
        chunks_data = chunk_text(extracted_text)

        if chunks_data:
            # 모든 청크에 대한 임베딩 생성
            chunk_contents = [c["content"] for c in chunks_data]
            embeddings = self.embedding_service.encode_documents(chunk_contents)

            # 저장을 위한 청크 준비
            chunks_to_store = []
            for i, chunk_data in enumerate(chunks_data):
                chunks_to_store.append({
                    "document_id": document.id,
                    "content": chunk_data["content"],
                    "chunk_index": chunk_data["chunk_index"],
                    "token_count": chunk_data["token_count"],
                    "embedding": embeddings[i],
                })

            # 청크 저장
            await self.chunk_repo.create_many(chunks_to_store)

        return {
            "id": document.id,
            "filename": document.filename,
            "format": document.format,
            "file_size": document.file_size,
            "chunk_count": len(chunks_data),
            "created_at": document.created_at,
        }
