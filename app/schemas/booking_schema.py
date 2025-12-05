from pydantic import BaseModel, field_validator
from typing import Optional, List
from datetime import datetime

# Tạo booking mới
class BookingCreate(BaseModel):
    user_id: str  # UUID
    time_slot_id: int
    service_ids: List[int]  # Danh sách service IDs
    total_duration_min: int
    total_price: int
    status: Optional[str] = "confirmed"
    
    @field_validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['confirmed', 'completed', 'cancelled']
        if v not in allowed_statuses:
            raise ValueError(f'Status phải là một trong: {allowed_statuses}')
        return v
    
    @field_validator('service_ids')
    def validate_service_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Phải chọn ít nhất một dịch vụ')
        return v

# Response booking với thông tin đầy đủ
class BookingResponse(BaseModel):
    id: int
    user_id: str
    status: str
    time_slot_id: int
    total_duration_min: int
    total_price: int
    # Thông tin bổ sung từ joins
    user: Optional[dict] = None
    time_slot: Optional[dict] = None
    barber: Optional[dict] = None
    services: Optional[List[dict]] = None

    class Config:
        from_attributes = True

# Response booking cơ bản
class Booking(BaseModel):
    id: int
    user_id: str
    status: str
    time_slot_id: int
    total_duration_min: int
    total_price: int

    class Config:
        from_attributes = True