"""문서 관리 API 엔드포인트."""

from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.config import settings
from src.db.repositories import DocumentRepository, ChunkRepository
from src.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetail,
    ErrorResponse,
)
from src.services.ingestion import IngestionService

router = APIRouter(tags=["Documents"])

# 지원되는 파일 형식
SUPPORTED_FORMATS = {"txt", "md", "json"}


def get_file_format(filename: str) -> str | None:
    """파일 이름에서 파일 형식을 추출하고 유효성을 검사합니다."""
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext if ext in SUPPORTED_FORMATS else None


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """인덱싱을 위해 문서를 업로드합니다.

    지원되는 형식: txt, md, json.
    같은 이름의 파일이 존재하면 대체됩니다.
    """
    # 파일 형식 유효성 검사
    format = get_file_format(file.filename)
    if not format:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="INVALID_FILE_FORMAT",
                message=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            ).model_dump(),
        )

    # 콘텐츠 읽기
    content = await file.read()

    # 파일 크기 유효성 검사
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=ErrorResponse(
                error="FILE_TOO_LARGE",
                message=f"File size exceeds limit of {settings.max_file_size // (1024 * 1024)}MB",
            ).model_dump(),
        )

    # 비어 있지 않은지 유효성 검사
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="EMPTY_FILE",
                message="File is empty",
            ).model_dump(),
        )

    # 콘텐츠가 공백만 있지 않은지 유효성 검사
    try:
        text_content = content.decode("utf-8")
        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error="EMPTY_CONTENT",
                    message="File contains no meaningful content",
                ).model_dump(),
            )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="INVALID_ENCODING",
                message="File must be UTF-8 encoded",
            ).model_dump(),
        )

    # 서비스 가져오기
    pool = request.app.state.db_pool
    embedding_service = request.app.state.embedding_service

    document_repo = DocumentRepository(pool)
    chunk_repo = ChunkRepository(pool, settings.chunk_table)
    ingestion_service = IngestionService(document_repo, chunk_repo, embedding_service)

    try:
        result = await ingestion_service.process_document(
            filename=file.filename,
            content=content,
            format=format,
        )
        return DocumentUploadResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="PROCESSING_ERROR",
                message=str(e),
            ).model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message="Failed to process document",
                details={"error": str(e)},
            ).model_dump(),
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """모든 업로드된 문서 목록을 조회합니다."""
    pool = request.app.state.db_pool
    document_repo = DocumentRepository(pool)

    documents = await document_repo.list_all(settings.chunk_table)

    return DocumentListResponse(
        documents=[
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                format=doc.format,
                file_size=doc.file_size,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )
            for doc in documents
        ],
        total=len(documents),
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(request: Request, document_id: UUID) -> DocumentDetail:
    """ID로 문서 세부 정보를 조회합니다."""
    pool = request.app.state.db_pool
    document_repo = DocumentRepository(pool)

    document = await document_repo.get_by_id(document_id, settings.chunk_table)

    if not document:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="NOT_FOUND",
                message=f"Document with ID {document_id} not found",
            ).model_dump(),
        )

    return DocumentDetail(
        id=document.id,
        filename=document.filename,
        format=document.format,
        file_size=document.file_size,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
        content_preview=document.content[:500] if document.content else "",
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(request: Request, document_id: UUID) -> None:
    """문서와 모든 청크를 삭제합니다."""
    pool = request.app.state.db_pool
    document_repo = DocumentRepository(pool)

    # 문서가 존재하는지 확인
    document = await document_repo.get_by_id(document_id, settings.chunk_table)
    if not document:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="NOT_FOUND",
                message=f"Document with ID {document_id} not found",
            ).model_dump(),
        )

    # 문서 삭제 (청크 연쇄 삭제)
    await document_repo.delete(document_id)
