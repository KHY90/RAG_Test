"""문서 청킹 기능에 대한 테스트."""

import pytest


class TestChunking:
    """고정 크기 및 중복을 사용한 텍스트 청킹 테스트."""

    def test_chunk_short_text(self):
        """청크 크기보다 짧은 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=512, overlap=50)

        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["token_count"] > 0

    def test_chunk_exact_size(self):
        """청크 크기에 딱 맞는 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        # 청크 크기 정도의 텍스트 생성
        text = "word " * 100  # ~100 tokens
        chunks = chunk_text(text.strip(), chunk_size=100, overlap=10)

        assert len(chunks) >= 1
        assert all(c["chunk_index"] >= 0 for c in chunks)

    def test_chunk_long_text_creates_multiple_chunks(self):
        """긴 텍스트가 여러 청크를 생성하는지 테스트."""
        from src.services.ingestion import chunk_text

        # 긴 텍스트 생성
        text = "This is a test sentence. " * 200  # Should exceed 512 tokens
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        # 순차적 인덱스 확인
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_overlap_preserves_context(self):
        """중복이 청크 간의 컨텍스트를 보존하는지 테스트."""
        from src.services.ingestion import chunk_text

        text = "First part of the document. " * 50 + "MARKER " + "Second part continues. " * 50
        chunks = chunk_text(text, chunk_size=100, overlap=30)

        # 중복이 있으면 일부 콘텐츠가 인접한 청크에 나타나야 합니다
        if len(chunks) > 1:
            # 청크가 완전히 분리되지 않았는지 확인
            all_content = " ".join(c["content"] for c in chunks)
            assert "MARKER" in all_content

    def test_chunk_token_count_accuracy(self):
        """토큰 수가 정확한지 테스트."""
        from src.services.ingestion import chunk_text

        text = "Hello world, this is a test."
        chunks = chunk_text(text, chunk_size=512, overlap=50)

        assert len(chunks) == 1
        # 토큰 수는 합리적이어야 합니다 (0이 아니고, 너무 크지 않음)
        assert 0 < chunks[0]["token_count"] < 100

    def test_chunk_korean_text(self):
        """한국어 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        text = "한글 텍스트 테스트입니다. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) >= 1
        assert all("한글" in c["content"] or "테스트" in c["content"] for c in chunks)

    def test_chunk_mixed_language(self):
        """혼합된 한국어/영어 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        text = "Hello 안녕하세요. This is 테스트입니다. " * 50
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) >= 1

    def test_chunk_empty_text(self):
        """빈 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        chunks = chunk_text("", chunk_size=512, overlap=50)

        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]["content"] == "")

    def test_chunk_whitespace_only(self):
        """공백만 있는 텍스트 청킹 테스트."""
        from src.services.ingestion import chunk_text

        chunks = chunk_text("   \n\t  ", chunk_size=512, overlap=50)

        # 우아하게 처리해야 합니다
        assert len(chunks) <= 1

    def test_chunk_preserves_all_content(self):
        """청킹 중에 콘텐츠가 손실되지 않는지 테스트."""
        from src.services.ingestion import chunk_text

        original_words = ["word1", "word2", "word3", "word4", "word5"] * 100
        text = " ".join(original_words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)

        # 모든 콘텐츠가 존재해야 합니다 (중복 고려)
        combined = " ".join(c["content"] for c in chunks)
        for word in ["word1", "word2", "word3", "word4", "word5"]:
            assert word in combined

    def test_chunk_default_parameters(self):
        """기본 매개변수 (512 토큰, 50 중복)로 청킹 테스트."""
        from src.services.ingestion import chunk_text

        text = "Test content. " * 200
        chunks = chunk_text(text)  # Use defaults

        assert len(chunks) >= 1
        assert all(c["token_count"] <= 600 for c in chunks)  # 약간의 버퍼 허용
