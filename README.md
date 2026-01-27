# Hybrid RAG Search System

로컬 환경에서 동작하는 하이브리드 검색 기반 RAG(Retrieval-Augmented Generation) 시스템입니다. GPU 없이 CPU만으로 문서를 검색하고 질문에 답변할 수 있습니다.

## 🎯 주요 기능

- **다양한 문서 형식 지원**: `.txt`, `.md`, `.json` 파일 업로드 및 처리
- **하이브리드 검색**: 의미 기반 검색(Dense)과 키워드 검색(Sparse)을 결합
  - **Dense Search**: 벡터 임베딩을 통한 의미적 유사도 검색
  - **Sparse Search**: BM25 알고리즘 기반 키워드 매칭
  - **Trigram Search**: 부분 단어 매칭을 위한 트라이그램 검색
  - **RRF (Reciprocal Rank Fusion)**: 세 가지 검색 결과를 통합
- **자연어 질의응답**: 업로드된 문서 기반 질문 답변 생성
- **출처 추적**: 답변에 사용된 문서 참조 정보 제공
- **CPU 전용**: GPU 없이도 동작 가능 (8GB RAM 권장)

## 🏗️ 기술 스택

### Backend
- **Framework**: FastAPI
- **Database**: PostgreSQL 15+ with pgvector, pg_trgm extensions
- **Embedding Models** (선택 가능):
  - `intfloat/multilingual-e5-base` (768차원, 다국어 지원, ~1GB) - 기본값
  - `sentence-transformers/all-MiniLM-L6-v2` (384차원, 영어 최적화, 빠름, ~90MB)
- **LLM**: Qwen 2.5-3B Instruct (GGUF Q4_K_M, ~2GB)
- **Vector Search**: pgvector with HNSW index
- **Full-text Search**: PostgreSQL tsvector with BM25 ranking
- **Trigram Search**: pg_trgm extension

### Key Libraries
- `sentence-transformers`: 임베딩 생성
- `llama-cpp-python`: CPU 기반 LLM 추론
- `asyncpg`: 비동기 PostgreSQL 드라이버
- `pydantic`: 데이터 검증 및 설정 관리

## 📋 시스템 요구사항

- **Python**: 3.11 이상
- **PostgreSQL**: 15 이상
- **RAM**: 최소 8GB
- **디스크 공간**: 약 5GB (모델 파일용)
- **OS**: Windows, Linux, macOS

## 🚀 빠른 시작

### 1. 데이터베이스 설정

```bash
# PostgreSQL 데이터베이스 생성
createdb ragtest

# 확장 기능 설치 (psql에서 실행)
psql -d ragtest -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d ragtest -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# 스키마 초기화
psql -d ragtest -f src/db/schema.sql
```

### 2. Python 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 의존성 설치
pip install -r requirements.txt
```

### 3. 모델 다운로드

```bash
# 모델 디렉토리 생성
mkdir models

# LLM 모델 다운로드 (Qwen 2.5-3B, ~2GB)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models
```

> **참고**: 임베딩 모델(`multilingual-e5-base`)은 첫 실행 시 자동으로 다운로드됩니다.

### 4. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```env
# Database
DATABASE_URL=postgresql://localhost/ragtest
DATABASE_USER=postgres
DATABASE_PASSWORD=
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=ragtest

# Models
# 임베딩 모델 타입 선택 (다음 중 하나를 선택):
#   - multilingual: intfloat/multilingual-e5-base (다국어 지원, 기본값)
#   - minilm: sentence-transformers/all-MiniLM-L6-v2 (영어 최적화, 빠름)
EMBEDDING_MODEL_TYPE=multilingual
LLM_MODEL_PATH=./models/qwen2.5-3b-instruct-q4_k_m.gguf

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Search
DEFAULT_TOP_K=5
RRF_K=60

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### 5. 서버 실행

```bash
# 개발 모드
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

서버가 실행되면 다음 URL에서 확인할 수 있습니다:
- **API 문서**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 사용 방법

### 문서 업로드

```bash
# 텍스트 파일 업로드
curl -X POST http://localhost:8000/api/documents \
  -F "file=@sample.txt"

# 마크다운 파일 업로드
curl -X POST http://localhost:8000/api/documents \
  -F "file=@README.md"

# JSON 파일 업로드
curl -X POST http://localhost:8000/api/documents \
  -F "file=@data.json"
```

### 문서 목록 조회

```bash
curl http://localhost:8000/api/documents
```

### 질문하기

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "이 문서의 주요 내용은 무엇인가요?"}'
```

### 검색만 수행 (답변 생성 없이)

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "hybrid search",
    "top_k": 5,
    "search_type": "hybrid"
  }'
```

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html

# 특정 테스트 카테고리
pytest tests/unit/          # 단위 테스트
pytest tests/integration/   # 통합 테스트
pytest tests/contract/      # 계약 테스트
```

## 📊 성능 지표

| 작업 | 예상 소요 시간 |
|------|---------------|
| 첫 모델 로딩 | 20-60초 |
| 문서 업로드 (1MB) | < 10초 |
| 질문 답변 생성 | 10-30초 |
| 검색만 수행 | < 2초 |

## 🏛️ 아키텍처

### 데이터 모델

```
Document (문서)
  ├── id: UUID
  ├── filename: VARCHAR(255) UNIQUE
  ├── content: TEXT
  ├── format: VARCHAR(10) ['txt', 'md', 'json']
  └── chunks: Chunk[]

Chunk (청크) - 선택된 임베딩 모델에 따라 다른 테이블 사용
  ├── chunks_768 (multilingual-e5-base용)
  │   ├── id: UUID
  │   ├── document_id: UUID (FK)
  │   ├── content: TEXT
  │   ├── chunk_index: INTEGER
  │   ├── embedding: VECTOR(768)
  │   └── search_vector: TSVECTOR
  │
  └── chunks_384 (all-MiniLM-L6-v2용)
      ├── id: UUID
      ├── document_id: UUID (FK)
      ├── content: TEXT
      ├── chunk_index: INTEGER
      ├── embedding: VECTOR(384)
      └── search_vector: TSVECTOR
```

### 검색 파이프라인

1. **문서 업로드** → 텍스트 추출 → 청킹 (512 토큰, 50 토큰 오버랩)
2. **임베딩 생성** → pgvector에 저장
3. **검색 인덱스 생성** → tsvector (BM25), trigram
4. **하이브리드 검색**:
   - Dense Search (벡터 유사도)
   - BM25 Search (키워드 매칭)
   - Trigram Search (부분 매칭)
   - RRF로 결과 통합
5. **답변 생성** → LLM에 컨텍스트 전달 → 자연어 답변 생성

## 🔧 문제 해결

### "Model not found" 오류
- `.env` 파일의 `LLM_MODEL_PATH`가 올바른지 확인
- 모델 파일이 완전히 다운로드되었는지 확인 (~2GB)

### "Database connection failed" 오류
- PostgreSQL이 실행 중인지 확인: `pg_isready`
- `.env`의 `DATABASE_URL` 확인
- pgvector 확장이 설치되었는지 확인:
  ```sql
  SELECT * FROM pg_extension WHERE extname = 'vector';
  ```

### "Out of memory" 오류
- 다른 애플리케이션을 종료하여 RAM 확보
- `.env`에서 `CHUNK_SIZE` 줄이기
- 더 작은 양자화 모델 사용 (Q3_K_M)

### 응답 속도가 느림
- 첫 요청은 모델 로딩으로 ~30초 소요 (이후 요청은 빠름)
- CPU를 많이 사용하는 다른 프로세스 종료
- Q4_0 양자화 모델 사용 고려 (속도 향상, 품질 약간 저하)

## 📝 테스트용 샘플 파일

**sample.txt**:
```
하이브리드 검색은 밀집 검색과 희소 검색을 결합한 방법입니다.
밀집 검색은 의미적 유사성을 기반으로 하고,
희소 검색은 키워드 매칭을 기반으로 합니다.
Reciprocal Rank Fusion을 사용하여 두 결과를 통합합니다.
```

**sample.json**:
```json
{
  "title": "RAG System Overview",
  "content": "Retrieval-Augmented Generation combines search with language models.",
  "topics": ["AI", "NLP", "Search"]
}
```

## 🎯 주요 기능 요구사항

- ✅ txt, md, json 파일 업로드 지원
- ✅ 의미 기반 검색 (Dense Search)
- ✅ 키워드 검색 (BM25)
- ✅ 트라이그램 검색 (부분 매칭)
- ✅ 하이브리드 검색 (RRF 통합)
- ✅ 자연어 질의응답
- ✅ 출처 참조 제공
- ✅ CPU 전용 동작 (GPU 불필요)
- ✅ 동시 요청 처리
- ✅ 중복 파일명 자동 교체

## 📄 라이선스

이 프로젝트는 로컬 개발 및 테스트 목적으로 제작되었습니다.

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

---

**개발 시작일**: 2026-01-27  
**상태**: 개발 중 (Draft)
