# app/routers/rag_router.py

from fastapi import APIRouter, HTTPException, Path, Query
from app.services.rag_service import rag_service
from app.schemas.rag_schema import (
    ChatRequest, ChatResponse,
    ChatHistoryResponse, ChatMessage,
    ChatSessionsResponse, ChatSession,
    CreateSessionRequest, UpdateSessionRequest
)

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot RAG"])

# ====================================
# 1. CHAT ENDPOINT 
# ====================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """
     Chat với BarberGo AI Assistant
    
    **Tự động quản lý session:**
    - Nếu `session_id` = null → Tạo session mới
    - Nếu có `session_id` → Tiếp tục chat trong session đó
    
    **Parameters:**
    - **user_id**: ID của user (bắt buộc)
    - **question**: Câu hỏi của bạn (bắt buộc)
    - **session_id**: ID session hiện tại (tùy chọn)
    - **top_k**: Số documents tham khảo (mặc định: 3)
    - **return_sources**: Hiển thị nguồn tham khảo (mặc định: false)
    
    **Returns:**
    - **session_id**: ID của session (mới hoặc hiện tại)
    - **answer**: Câu trả lời
    - **confidence**: Độ tin cậy (high/medium/low)
    - **sources**: Nguồn tham khảo (nếu return_sources=true)
    """
    try:
        # Validate input
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Câu hỏi không được để trống"
            )
        
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id là bắt buộc"
            )
        
        # Gọi RAG service (đã có logic tạo session tự động)
        result = rag_service.query(
            question=request.question,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            return_sources=request.return_sources
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {str(e)}"
        )

# ====================================
# 2. QUẢN LÝ SESSIONS
# ====================================

@router.post("/sessions", status_code=201)
async def create_session(request: CreateSessionRequest):
    """
     Tạo session chat mới
    
    **Parameters:**
    - **user_id**: ID của user
    - **title**: Tiêu đề session (VD: "Hỏi về đặt lịch")
    
    **Returns:**
    - **session_id**: ID của session vừa tạo
    """
    try:
        session_id = rag_service.create_chat_session(
            user_id=request.user_id,
            title=request.title
        )
        
        return {
            "session_id": session_id,
            "message": "Tạo session thành công"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo session: {str(e)}"
        )

@router.get("/sessions/{user_id}", response_model=ChatSessionsResponse)
async def get_user_sessions(
    user_id: str = Path(..., description="ID của user"),
    limit: int = Query(20, ge=1, le=100, description="Số lượng sessions")
):
    """
     Lấy danh sách tất cả chat sessions của user
    
    **Parameters:**
    - **user_id**: ID của user
    - **limit**: Số lượng sessions tối đa (mặc định: 20)
    
    **Returns:**
    - Danh sách sessions (sắp xếp theo thời gian tạo mới nhất)
    """
    try:
        sessions = rag_service.get_user_sessions(user_id)
        
        # Giới hạn số lượng
        sessions = sessions[:limit]
        
        return {
            "sessions": [
                ChatSession(
                    id=s["id"],
                    user_id=s.get("user_id", user_id),
                    title=s["title"],
                    created_at=s["created_at"]
                )
                for s in sessions
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy danh sách sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str = Path(..., description="ID của session")
):
    """
     Lấy lịch sử chat trong một session
    
    **Parameters:**
    - **session_id**: ID của session
    
    **Returns:**
    - Toàn bộ lịch sử chat (user + assistant)
    """
    try:
        messages = rag_service.get_chat_history(session_id)
        
        if not messages:
            return {
                "session_id": session_id,
                "messages": []
            }
        
        return {
            "session_id": session_id,
            "messages": [
                ChatMessage(
                    id=msg["id"],
                    role=msg["role"],
                    content=msg["content"],
                    confidence=msg.get("confidence"),
                    created_at=msg["created_at"]
                )
                for msg in messages
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy lịch sử chat: {str(e)}"
        )

@router.put("/sessions/{session_id}")
async def update_session_title(
    session_id: str = Path(..., description="ID của session"),
    request: UpdateSessionRequest = ...
):
    """
     Đổi tên session
    
    **Parameters:**
    - **session_id**: ID của session
    - **title**: Tiêu đề mới
    """
    try:
        result = rag_service.supabase.table("chat_sessions")\
            .update({"title": request.title})\
            .eq("id", session_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy session"
            )
        
        return {
            "message": "Cập nhật tên session thành công",
            "session_id": session_id,
            "new_title": request.title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi cập nhật session: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str = Path(..., description="ID của session")
):
    """
     Xóa session và toàn bộ messages
    
    **Parameters:**
    - **session_id**: ID của session cần xóa
    
    **Note:** Sẽ xóa cascade toàn bộ messages trong session
    """
    try:
        # 1. Xóa messages trước (vì có foreign key)
        rag_service.supabase.table("chat_messages")\
            .delete()\
            .eq("session_id", session_id)\
            .execute()
        
        # 2. Xóa session
        result = rag_service.supabase.table("chat_sessions")\
            .delete()\
            .eq("id", session_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy session"
            )
        
        return {
            "message": "Xóa session thành công",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa session: {str(e)}"
        )

# ====================================
# 3. QUẢN LÝ DOCUMENTS (Knowledge Base)
# ====================================

@router.post("/documents", status_code=201)
async def create_document(request: dict):
    """
     Tạo document mới trong Knowledge Base
    
    **Parameters:**
    - **content**: Nội dung chính của document (VD: "Cách đặt lịch trên app")
    - **output**: Câu trả lời hoặc thông tin liên quan (VD: "Bước 1: Mở app... Bước 2:...")
    - **extra_metadata** (optional): Metadata bổ sung (VD: {"category": "FAQ", "priority": 1})
    
    **Returns:**
    - Thông báo tạo thành công và document ID
    """
    try:
        # Validate
        if not request.get("content") or not request.get("output"):
            raise HTTPException(
                status_code=400,
                detail="content và output là bắt buộc"
            )
        
        result = rag_service.create_document(
            content=request["content"],
            output=request["output"],
            extra_metadata=request.get("extra_metadata")
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Không thể tạo document"
            )
        
        return {
            "message": "Tạo document thành công",
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo document: {str(e)}"
        )

@router.get("/documents")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000, description="Số lượng documents"),
    offset: int = Query(0, ge=0, description="Vị trí bắt đầu")
):
    """
     Lấy danh sách tất cả documents (có phân trang)
    
    **Parameters:**
    - **limit**: Số lượng documents tối đa (mặc định: 100)
    - **offset**: Vị trí bắt đầu (mặc định: 0)
    
    **Returns:**
    - Danh sách documents với content, metadata
    """
    try:
        documents = rag_service.get_all_documents(limit=limit, offset=offset)
        
        return {
            "total": len(documents),
            "limit": limit,
            "offset": offset,
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy danh sách documents: {str(e)}"
        )

@router.get("/documents/{document_id}")
async def get_document_detail(
    document_id: int = Path(..., ge=1, description="ID của document")
):
    """
     Lấy chi tiết một document theo ID
    
    **Parameters:**
    - **document_id**: ID của document
    
    **Returns:**
    - Chi tiết document (id, content, metadata)
    """
    try:
        document = rag_service.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy document với ID {document_id}"
            )
        
        return {
            "document": document
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy document: {str(e)}"
        )

@router.get("/documents/search/keyword")
async def search_documents(
    keyword: str = Query(..., min_length=1, description="Từ khóa tìm kiếm")
):
    """
     Tìm kiếm documents theo từ khóa
    
    **Parameters:**
    - **keyword**: Từ khóa cần tìm (sẽ tìm trong content)
    
    **Returns:**
    - Danh sách documents chứa từ khóa
    """
    try:
        documents = rag_service.search_documents_by_keyword(keyword)
        
        return {
            "keyword": keyword,
            "total": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tìm kiếm documents: {str(e)}"
        )

@router.put("/documents/{document_id}")
async def update_document(
    document_id: int = Path(..., ge=1, description="ID của document"),
    request: dict = ...
):
    """
     Cập nhật document
    
    **Parameters:**
    - **document_id**: ID của document
    - **new_content** (optional): Nội dung mới
    - **new_output** (optional): Câu trả lời/thông tin mới
    - **new_metadata** (optional): Metadata mới (sẽ merge với metadata cũ)
    
    **Note:**
    - Nếu `new_content` thay đổi → embedding sẽ được tạo lại tự động
    - Metadata mới sẽ merge với metadata cũ (không ghi đè hoàn toàn)
    """
    try:
        result = rag_service.update_document(
            document_id=document_id,
            new_content=request.get("new_content"),
            new_output=request.get("new_output"),
            new_metadata=request.get("new_metadata")
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy document với ID {document_id}"
            )
        
        return {
            "message": "Cập nhật document thành công",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi cập nhật document: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int = Path(..., ge=1, description="ID của document")
):
    """
     Xóa document
    
    **Parameters:**
    - **document_id**: ID của document cần xóa
    
    **Note:** Xóa document sẽ làm mất embedding và metadata liên quan
    """
    try:
        result = rag_service.delete_document(document_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy document với ID {document_id}"
            )
        
        return {
            "message": "Xóa document thành công",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xóa document: {str(e)}"
        )

# ====================================
# 4. HEALTH CHECK & TEST
# ====================================

@router.get("/health")
async def health_check():
    """ Health check endpoint"""
    return {
        "status": "healthy",
        "service": "BarberGo RAG Chatbot",
        "model": "Gemini 2.5 Flash + Supabase Vector DB",
        "features": [
            "Multi-session chat",
            "Chat history",
            "RAG with embeddings",
            "Smart fallback answers"
        ]
    }

@router.get("/test")
async def test_chatbot():
    """
     Test endpoint với câu hỏi mẫu
    
    **Note:** Chỉ dùng để test, không lưu vào database
    """
    test_questions = [
        "Làm thế nào để đặt lịch?",
        "Tôi muốn hủy lịch thì làm sao?",
        "Đặt lịch trên app có mất phí không?"
    ]
    
    results = []
    for q in test_questions:
        # Test RAG retrieval only (không lưu DB)
        docs = rag_service.search_similar_documents(q, top_k=2)
        answer = rag_service.generate_answer(q, docs)
        
        results.append({
            "question": q,
            "answer": answer,
            "found_docs": len(docs),
            "similarity": docs[0]["similarity"] if docs else 0
        })
    
    return {
        "message": "Test chatbot with sample questions",
        "results": results
    }

@router.get("/all_documents")
async def get_all_documents():
    """
     Lấy tất cả documents trong vector DB
    
    **Note:** Chỉ dùng để kiểm tra dữ liệu
    """
    try:
        documents = rag_service.supabase.table("documents")\
            .select("*")\
            .execute()
        
        return {
            "total_documents": len(documents.data),
            "documents": documents.data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy documents: {str(e)}"
        )
