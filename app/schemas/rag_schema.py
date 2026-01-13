from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="Câu hỏi của người dùng, tối đa 4000 ký tự")
    user_id: str  
    session_id: Optional[str] = None  
    top_k: Optional[int] = 3
    return_sources: Optional[bool] = False

class Source(BaseModel):
    content: str
    metadata: dict = {}
    similarity: float

class ChatResponse(BaseModel):
    session_id: str 
    answer: str
    confidence: str
    sources: Optional[List[Source]] = None

class ChatMessage(BaseModel):
    id: str
    role: str   
    content: str
    confidence: Optional[str] = None
    created_at: datetime

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class ChatSession(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime

class ChatSessionsResponse(BaseModel):
    sessions: List[ChatSession]

class CreateSessionRequest(BaseModel):
    user_id: str
    title: str

class UpdateSessionRequest(BaseModel):
    title: str
