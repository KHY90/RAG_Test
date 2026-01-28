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
    llm_model_path: str = Field(default="./models/qwen2.5-3b-instruct-q4_k_m.gguf")
    
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

    # 파일 업로드
    max_file_size: int = Field(default=10485760)  # 10MB

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
