"""페이지 렌더링 라우터."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Pages"])

templates = Jinja2Templates(directory="src/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """채팅 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """문서 관리 페이지를 렌더링합니다."""
    return templates.TemplateResponse("documents.html", {"request": request})
