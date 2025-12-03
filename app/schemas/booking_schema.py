from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

# Tạo booking mới
class BookingCreate(BaseModel):
    barber_id: str  # UUID
    user_id: str  # UUID
    service_id: int
    date_time: datetime
    status: Optional[str] = "pending"
    
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
        if v not in allowed_statuses:
            raise ValueError(f'Status phải là một trong: {allowed_statuses}')
        return v

# Cập nhật booking
class BookingUpdate(BaseModel):
    barber_id: Optional[str] = None
    user_id: Optional[str] = None
    service_id: Optional[int] = None
    date_time: Optional[datetime] = None
    status: Optional[str] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
            if v not in allowed_statuses:
                raise ValueError(f'Status phải là một trong: {allowed_statuses}')
        return v

# Response booking
class Booking(BaseModel):
    id: int
    barber_id: Optional[str] = None
    user_id: Optional[str] = None
    service_id: Optional[int] = None
    date_time: Optional[datetime] = None
    status: Optional[str] = None

    class Config:
        from_attributes = True