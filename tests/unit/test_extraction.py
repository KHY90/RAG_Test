"""다양한 파일 형식에서의 텍스트 추출 테스트."""

import json
import pytest


class TestTextExtraction:
    """extract_txt, extract_md, extract_json 함수에 대한 테스트."""

    def test_extract_txt_simple(self):
        """간단한 txt 파일에서 텍스트 추출 테스트."""
        from src.services.ingestion import extract_txt

        content = "Hello, this is a test document.\nWith multiple lines."
        result = extract_txt(content)

        assert result == content
        assert "Hello" in result
        assert "multiple lines" in result

    def test_extract_txt_unicode(self):
        """한국어 문자가 포함된 텍스트 추출 테스트."""
        from src.services.ingestion import extract_txt

        content = "한글 텍스트 테스트입니다.\n영어와 한글이 섞여 있습니다."
        result = extract_txt(content)

        assert result == content
        assert "한글" in result

    def test_extract_txt_empty(self):
        """빈 콘텐츠에서 텍스트 추출 테스트."""
        from src.services.ingestion import extract_txt

        result = extract_txt("")
        assert result == ""

    def test_extract_md_preserves_content(self):
        """마크다운 콘텐츠가 보존되는지 테스트."""
        from src.services.ingestion import extract_md

        content = "# Heading\n\nParagraph with **bold** text.\n\n- List item"
        result = extract_md(content)

        # 인덱싱을 위해 마크다운 콘텐츠를 그대로 보존해야 합니다
        assert "Heading" in result
        assert "bold" in result
        assert "List item" in result

    def test_extract_md_code_blocks(self):
        """코드 블록이 있는 마크다운 테스트."""
        from src.services.ingestion import extract_md

        content = "```python\nprint('hello')\n```\n\nSome text."
        result = extract_md(content)

        assert "print" in result
        assert "Some text" in result

    def test_extract_json_simple(self):
        """간단한 JSON에서 텍스트 추출 테스트."""
        from src.services.ingestion import extract_json

        content = json.dumps({"title": "Test", "content": "Document content"})
        result = extract_json(content)

        assert "Test" in result
        assert "Document content" in result

    def test_extract_json_nested(self):
        """중첩된 JSON에서 텍스트 추출 테스트."""
        from src.services.ingestion import extract_json

        content = json.dumps({
            "title": "Nested Test",
            "metadata": {
                "author": "Test User",
                "tags": ["tag1", "tag2"]
            },
            "sections": [
                {"heading": "Section 1", "text": "Content 1"},
                {"heading": "Section 2", "text": "Content 2"}
            ]
        })
        result = extract_json(content)

        assert "Nested Test" in result
        assert "Test User" in result
        assert "tag1" in result
        assert "Section 1" in result
        assert "Content 1" in result

    def test_extract_json_array(self):
        """JSON 배열에서 텍스트 추출 테스트."""
        from src.services.ingestion import extract_json

        content = json.dumps(["item1", "item2", {"nested": "value"}])
        result = extract_json(content)

        assert "item1" in result
        assert "item2" in result
        assert "value" in result

    def test_extract_json_with_numbers(self):
        """추출에서 숫자가 제외되는지 테스트."""
        from src.services.ingestion import extract_json

        content = json.dumps({"title": "Test", "count": 42, "price": 19.99})
        result = extract_json(content)

        assert "Test" in result
        # 숫자는 문자열로 추출된 텍스트에 없어야 합니다
        assert "42" not in result or result.count("42") == 0

    def test_extract_json_invalid(self):
        """유효하지 않은 JSON 처리 테스트."""
        from src.services.ingestion import extract_json

        with pytest.raises(ValueError):
            extract_json("not valid json {")

    def test_extract_json_korean(self):
        """한국어 콘텐츠가 포함된 JSON 테스트."""
        from src.services.ingestion import extract_json

        content = json.dumps({
            "제목": "한글 테스트",
            "내용": "한글 내용입니다."
        }, ensure_ascii=False)
        result = extract_json(content)

        assert "한글 테스트" in result
        assert "한글 내용입니다" in result
