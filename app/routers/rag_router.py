from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_service
from typing import Optional, List, Dict

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot RAG"])

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    return_sources: Optional[bool] = False

class Source(BaseModel):
    question: str
    answer: str
    similarity: float

class ChatResponse(BaseModel):
    answer: str
    confidence: str
    sources: Optional[List[Source]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """
    💬 Chatbot BarberGo sử dụng RAG (Retrieval-Augmented Generation)
    
    Chatbot này có thể trả lời các câu hỏi về:
    - Cách đặt lịch, hủy lịch
    - Chính sách thanh toán, đặt cọc
    - Tính năng ứng dụng
    - Hợp tác đối tác
    - Và nhiều thông tin khác về BarberGo
    
    **Parameters:**
    - **question**: Câu hỏi của bạn
    - **top_k**: Số lượng documents liên quan để tham khảo (mặc định: 3)
    - **return_sources**: Có muốn xem nguồn tham khảo không (mặc định: false)
    
    **Returns:**
    - **answer**: Câu trả lời từ chatbot
    - **confidence**: Độ tin cậy (high/medium/low)
    - **sources**: Các nguồn tham khảo (nếu return_sources=true)
    """
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Câu hỏi không được để trống"
            )
        
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            return_sources=request.return_sources
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for RAG Chatbot"""
    return {
        "status": "healthy",
        "service": "BarberGo RAG Chatbot",
        "model": "Gemini 2.0 Flash + Supabase Vector DB"
    }

@router.get("/test")
async def test_chatbot():
    """
    Test endpoint với câu hỏi mẫu
    """
    test_questions = [
        "Làm thế nào để đặt lịch?",
        "Tôi muốn hủy lịch thì làm sao?",
        "Đặt lịch trên app có mất phí không?"
    ]
    
    results = []
    for q in test_questions:
        result = rag_service.query(q, top_k=2, return_sources=True)
        results.append({
            "question": q,
            "answer": result["answer"],
            "confidence": result["confidence"]
        })
    
    return {
        "message": "Test chatbot with sample questions",
        "results": results
    }