"""RAG 시스템 테스트를 위한 Pytest 구성 및 픽스처."""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# 테스트 환경 설정
os.environ["DATABASE_NAME"] = "ragtest_test"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """테스트 세션을 위한 이벤트 루프를 생성합니다."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """API 테스트를 위한 비동기 HTTP 클라이언트를 생성합니다."""
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_txt_content() -> str:
    """테스트용 샘플 텍스트 파일 콘텐츠."""
    return """하이브리드 검색은 밀집 검색과 희소 검색을 결합한 방법입니다.
밀집 검색은 의미적 유사성을 기반으로 하고,
희소 검색은 키워드 매칭을 기반으로 합니다.
Reciprocal Rank Fusion을 사용하여 두 결과를 통합합니다."""


@pytest.fixture
def sample_md_content() -> str:
    """테스트용 샘플 마크다운 파일 콘텐츠."""
    return """# RAG 시스템 개요

## 소개

검색 증강 생성(RAG)은 검색과 언어 모델을 결합합니다.

## 기능

- 문서 수집
- 하이브리드 검색
- 답변 생성
"""


@pytest.fixture
def sample_json_content() -> str:
    """테스트용 샘플 JSON 파일 콘텐츠."""
    return """{
    "title": "RAG 시스템 개요",
    "content": "검색 증강 생성은 검색과 언어 모델을 결합합니다.",
    "topics": ["AI", "NLP", "검색"],
    "metadata": {
        "author": "테스트 사용자",
        "version": "1.0"
    }
}"""


@pytest.fixture
def sample_documents() -> dict[str, str]:
    """테스트용 샘플 문서 모음."""
    return {
        "sample.txt": "이것은 RAG 시스템 테스트를 위한 샘플 텍스트 문서입니다.",
        "sample.md": "# 샘플 마크다운\n\n이것은 **마크다운** 문서입니다.",
        "sample.json": '{"title": "샘플", "content": "JSON 문서 콘텐츠"}',
    }
