from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
from decimal import Decimal

# Tạo rating mới
class RatingCreate(BaseModel):
    barber_id: str  # UUID
    user_id: str  # UUID
    score: Decimal
    comment: Optional[str] = None
    
    @validator('score')
    def validate_score(cls, v):
        if v < 0 or v > 5:
            raise ValueError('Điểm đánh giá phải từ 0 đến 5')
        return v

# Cập nhật rating
class RatingUpdate(BaseModel):
    score: Optional[Decimal] = None
    comment: Optional[str] = None
    
    @validator('score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError('Điểm đánh giá phải từ 0 đến 5')
        return v

# Response rating
class Rating(BaseModel):
    id: int
    barber_id: Optional[str] = None
    user_id: Optional[str] = None
    score: Optional[Decimal] = None
    comment: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True