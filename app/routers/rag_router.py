# Hybrid Chat Router
from fastapi import APIRouter, HTTPException, Query, Path, Request
from typing import List
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.services.rag_service import rag_service
from app.schemas.rag_schema import (
    ChatRequest,
    ChatResponse,
    Source,
    ChatMessage,
    ChatHistoryResponse,
    ChatSession,
    ChatSessionsResponse,
    CreateSessionRequest,
    UpdateSessionRequest,
)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/chatbot", tags=["Hybrid RAG Chatbot"])


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_healthy = True  # ollama_client.health_check()
    
    return {
        "status": "healthy",
        "strategy": "hybrid",
        "ollama": {
            "chat_model": rag_service.ollama_chat_model,
            "embed_model": rag_service.ollama_embed_model,
            "healthy": ollama_healthy
        },
        "gemini": {
            "model": "gemini-2.0-flash-exp"
        },
        "threshold": rag_service.similarity_threshold,
        "timestamp": datetime.now()
    }


# ==================== CHAT ====================

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request ,body: ChatRequest):
    """
    Chat với Hybrid AI (Ollama + Gemini)
    
    **Strategy:**
    - similarity >= 0.75: Dùng Ollama (local, fast)
    - similarity < 0.75: Dùng Gemini (cloud, powerful)
    
    **Parameters:**
    - **question**: Câu hỏi
    - **user_id**: ID user
    - **session_id**: Optional (tự tạo nếu None)
    - **top_k**: Số documents (1-10)
    - **return_sources**: Trả về sources?
    """
    try:
        # Validate
        if not body.question or len(body.question.strip()) == 0:
            raise HTTPException(400, detail="Câu hỏi không được để trống")
        
        if not body.user_id:
            raise HTTPException(400, detail="user_id là bắt buộc")
        
        # Query hybrid service
        response = rag_service.query(
            question=body.question,
            user_id=body.user_id,
            session_id=body.session_id,
            top_k=body.top_k or 3,
            return_sources=body.return_sources or False
        )
        
        # Convert sources to Source model
        if response.get('sources'):
            response['sources'] = [
                Source(
                    id=doc.get('id', 0),
                    content=doc.get('content', ''),
                    similarity=doc.get('similarity', 0),
                    metadata=doc.get('metadata', {})
                )
                for doc in response['sources']
            ]
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Chat error: {str(e)}")


# ==================== SESSIONS ====================

@router.post("/sessions", status_code=201)
@limiter.limit("30/minute")
async def create_session(request : Request, body: CreateSessionRequest):
    """Tạo session chat mới"""
    try:
        session_id = rag_service.create_chat_session(
            user_id=body.user_id,
            title=body.title
        )
        
        return {
            "session_id": session_id,
            "message": "Tạo session thành công"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/sessions/{user_id}", response_model=ChatSessionsResponse)
@limiter.limit("120/minute")
async def get_sessions(
    request:Request,
    user_id: str = Path(...),
    limit: int = Query(20, ge=1, le=100)
):
    """Lấy danh sách sessions của user"""
    try:
        sessions = rag_service.get_user_sessions(user_id)
        sessions = sessions[:limit]
        
        return {
            "sessions": [ChatSession(**s) for s in sessions],
            "total": len(sessions)
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
@limiter.limit("120/minute")
async def get_history(request:Request, session_id: str = Path(...)):
    """Lấy lịch sử chat"""
    try:
        messages = rag_service.get_chat_history(session_id)
        
        return {
            "session_id": session_id,
            "messages": [ChatMessage(**m) for m in messages],
            "total": len(messages)
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.put("/sessions/{session_id}")
@limiter.limit("30/minute")
async def update_title(
    request:Request,
    session_id: str = Path(...),
    body: UpdateSessionRequest = ...
):
    """Đổi tên session"""
    try:
        success = rag_service.update_session_title(session_id, body.title)
        
        if not success:
            raise HTTPException(404, detail="Session not found")
        
        return {    
            "message": "Cập nhật thành công",
            "session_id": session_id,
            "new_title": body.title
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.delete("/sessions/{session_id}")
@limiter.limit("30/minute")
async def delete_session(request : Request,session_id: str = Path(...)):
    """Xóa session"""
    try:
        success = rag_service.delete_session(session_id)
        
        if not success:
            raise HTTPException(404, detail="Session not found")
        
        return {
            "message": "Xóa session thành công",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ==================== DOCUMENTS ====================

@router.post("/documents", status_code=201)
@limiter.limit("10/minute")
async def create_document(request:Request, body: dict):
    """Tạo document mới"""
    try:
        if not body.get("content") or not body.get("output"):
            raise HTTPException(400, detail="content và output là bắt buộc")
        
        success = rag_service.create_document(
            content=body["content"],
            output=body["output"],
            extra_metadata=body.get("extra_metadata")
        )
        
        if not success:
            raise HTTPException(500, detail="Không thể tạo document")
        
        return {
            "message": "Tạo document thành công",
            "content": body["content"][:50] + "..."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/documents")
@limiter.limit("120/minute")
async def get_documents(
    request:Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Lấy danh sách documents"""
    try:
        documents = rag_service.get_all_documents(limit, offset)
        
        # Get total count
        count_result = rag_service.supabase.table("documents") \
            .select("*", count="exact") \
            .execute()
        
        return {
            "total": count_result.count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < count_result.count,
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/documents/{document_id}")
@limiter.limit("120/minute")
async def get_document(request:Request,document_id: int = Path(..., ge=1)):
    """Lấy chi tiết document"""
    try:
        document = rag_service.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(404, detail="Document not found")
        
        return {"document": document}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/documents/search/{keyword}")
@limiter.limit("120/minute")
async def search_documents(request:Request, keyword: str = Path(..., min_length=1)):
    """Tìm kiếm documents"""
    try:
        documents = rag_service.search_documents_by_keyword(keyword)
        
        return {
            "keyword": keyword,
            "total": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.put("/documents/{document_id}")
@limiter.limit("120/minute")
async def update_document(
    request:Request,
    document_id: int = Path(..., ge=1),
    body: dict = ...
):
    """Cập nhật document"""
    try:
        success = rag_service.update_document(
            document_id=document_id,
            new_content=body.get("new_content"),
            new_output=body.get("new_output"),
            new_metadata=body.get("new_metadata")
        )
        
        if not success:
            raise HTTPException(404, detail="Document not found")
        
        return {
            "message": "Cập nhật thành công",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.delete("/documents/{document_id}")
@limiter.limit("30/minute")
async def delete_document(request:Request,document_id: int = Path(..., ge=1)):
    """Xóa document"""
    try:
        success = rag_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(404, detail="Document not found")
        
        return {
            "message": "Xóa thành công",
            "document_id": document_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))