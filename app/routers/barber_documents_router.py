"""
Barber Documents Router
- Prefix: /api/barber-documents
- Hoàn toàn tách biệt với /api/chatbot
"""

from fastapi import APIRouter, HTTPException, Query, Path, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.api.dependencies import require_admin, require_system
from app.schemas.barber_rag_schema import (
    BarberDocumentCreate,
    BarberDocumentUpdate,
    BarberDocumentItem,
    BarberDocumentListResponse,
    BarberSearchResponse,
)
from app.services import barber_rag_service as svc

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/api/barber-documents",
    tags=["Barber Vector Documents"]
)


# ==========================================
# Search (dùng cho chatbot hoặc client)
# ==========================================

@router.get("/search", response_model=BarberSearchResponse)
@limiter.limit("60/minute")
async def search(
    request: Request,
    q:         str   = Query(..., min_length=1, description="Câu hỏi tìm kiếm"),
    top_k:     int   = Query(3, ge=1, le=10),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    """Vector search trong barber_documents"""
    results = svc.search_barber_documents(q, top_k, threshold)
    return {
        "query":   q,
        "total":   len(results),
        "results": results
    }


@router.get("/search/keyword/{keyword}")
@limiter.limit("60/minute")
async def keyword_search(
    request: Request,
    keyword: str = Path(..., min_length=1),
    current_user: dict = Depends(require_admin)
):
    """Tìm kiếm theo keyword trong content"""
    docs = svc.search_barber_documents_by_keyword(keyword)
    return {"keyword": keyword, "total": len(docs), "documents": docs}


# ==========================================
# CRUD - chỉ admin/system
# ==========================================

@router.get("", response_model=BarberDocumentListResponse)
@limiter.limit("120/minute")
async def get_all(
    request:  Request,
    limit:    int            = Query(100, ge=1, le=500),
    offset:   int            = Query(0, ge=0),
    doc_type: str | None     = Query(None, description="Filter: opening_hours | services | location | rating | recommendation | service_detail"),
    current_user: dict       = Depends(require_admin)
):
    """Lấy danh sách tất cả barber documents"""
    return svc.get_all_barber_documents(limit, offset, doc_type)


@router.get("/{doc_id}", response_model=BarberDocumentItem)
@limiter.limit("120/minute")
async def get_one(
    request:  Request,
    doc_id:   int  = Path(..., ge=1),
    current_user: dict = Depends(require_admin)
):
    """Lấy chi tiết 1 document"""
    doc = svc.get_barber_document_by_id(doc_id)
    if not doc:
        raise HTTPException(404, detail="Document không tồn tại")
    return doc


@router.post("", status_code=201)
@limiter.limit("30/minute")
async def create(
    request: Request,
    body:    BarberDocumentCreate,
    current_user: dict = Depends(require_system)
):
    """Tạo document mới"""
    doc = svc.insert_barber_document(
        body.content,
        body.output,
        body.extra_metadata
    )
    if not doc:
        raise HTTPException(500, detail="Không thể tạo document")
    return {"message": "Tạo thành công", "id": doc.get("id")}


@router.put("/{doc_id}")
@limiter.limit("30/minute")
async def update(
    request:  Request,
    doc_id:   int = Path(..., ge=1),
    body:     BarberDocumentUpdate = ...,
    current_user: dict = Depends(require_system)
):
    """Cập nhật document (tự động re-embed nếu content thay đổi)"""
    if not any([body.new_content, body.new_output, body.new_metadata]):
        raise HTTPException(400, detail="Cần ít nhất 1 trường để cập nhật")

    ok = svc.update_barber_document(
        doc_id,
        body.new_content,
        body.new_output,
        body.new_metadata
    )
    if not ok:
        raise HTTPException(404, detail="Document không tồn tại")

    return {"message": "Cập nhật thành công", "id": doc_id}


@router.delete("/{doc_id}")
@limiter.limit("30/minute")
async def delete_one(
    request:  Request,
    doc_id:   int = Path(..., ge=1),
    current_user: dict = Depends(require_system)
):
    """Xoá 1 document"""
    ok = svc.delete_barber_document(doc_id)
    if not ok:
        raise HTTPException(404, detail="Document không tồn tại")
    return {"message": "Đã xoá", "id": doc_id}


@router.delete("/barber/{barber_id}")
@limiter.limit("10/minute")
async def delete_by_barber(
    request:   Request,
    barber_id: str = Path(...),
    current_user: dict = Depends(require_system)
):
    """Xoá toàn bộ documents của 1 tiệm"""
    count = svc.delete_barber_documents_by_barber_id(barber_id)
    return {"message": f"Đã xoá {count} documents", "barber_id": barber_id}