"""pydantic-settings를 사용한 애플리케이션 설정."""

from typing import ClassVar
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """환경 변수에서 로드된 애플리케이션 설정."""

    # 데이터베이스
    database_host: str = Field(default="localhost")
    database_port: int = Field(default=5432)
    database_name: str = Field(default="ragtest")
    database_user: str = Field(default="postgres")
    database_password: str = Field(default="")

    # 서버
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=True)

    # 모델
    embedding_model_type: str = Field(
        default="multilingual",
        description="임베딩 모델 타입: 'multilingual' 또는 'minilm'"
    )
    llm_model_path: str = Field(
        default="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
        description="GGUF 모델 파일 경로"
    )
    llm_context_length: int = Field(default=4096, description="LLM 컨텍스트 길이")
    llm_gpu_layers: int = Field(default=0, description="GPU에 로드할 레이어 수 (0=CPU only)")
    
    # 사용 가능한 임베딩 모델들
    EMBEDDING_MODELS: ClassVar[dict[str, str]] = {
        "multilingual": "intfloat/multilingual-e5-base",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    @property
    def embedding_model(self) -> str:
        """선택된 임베딩 모델 반환."""
        return self.EMBEDDING_MODELS.get(
            self.embedding_model_type, 
            self.EMBEDDING_MODELS["multilingual"]
        )

    # 검색
    default_top_k: int = Field(default=5)
    rrf_k: int = Field(default=60)

    # 청킹(Chunking)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    chunk_table: str = Field(default="chunks_768")

    # 파일 업로드
    max_file_size: int = Field(default=10485760)  # 10MB

    # 시스템 프롬프트
    system_prompt: str = Field(
        default=(
            "저는 업로드된 문서를 기반으로 질문에 답변하는 RAG 챗봇입니다. "
            "문서에서 관련 정보를 검색하여 정확한 답변을 제공합니다."
        )
    )

    # 기본 질문 패턴 (역할, 기능 등)
    basic_question_patterns: list[str] = Field(
        default=[
            "너는 누구",
            "뭐야",
            "누구야",
            "무엇을 할 수 있",
            "뭘 할 수 있",
            "어떤 역할",
            "소개해",
            "기능이 뭐",
            "what can you do",
            "who are you",
            "what are you",
        ]
    )

    # 기본 질문에 대한 응답
    basic_response: str = Field(
        default=(
            "저는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.\n\n"
            "**주요 기능:**\n"
            "- 업로드된 문서에서 관련 정보를 검색합니다\n"
            "- 검색된 컨텍스트를 바탕으로 질문에 답변합니다\n"
            "- 참조한 소스 문서를 함께 제공합니다\n\n"
            "문서를 업로드하고 질문해 주세요!"
        )
    )

    @property
    def database_url(self) -> str:
        """asyncpg를 위한 데이터베이스 URL 생성."""
        if self.database_password:
            return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
        return f"postgresql://{self.database_user}@{self.database_host}:{self.database_port}/{self.database_name}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 전역 설정 인스턴스
settings = Settings()
