from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import date, time

# Tạo time slot mới
class TimeSlotCreate(BaseModel):
    barber_id: str  # UUID
    slot_date: date
    start_time: time
    end_time: time
    is_available: Optional[bool] = True
    
    @field_validator('end_time')
    def validate_time_range(cls, v, info):
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError('end_time phải lớn hơn start_time')
        return v

# Tạo nhiều time slots cùng lúc
class TimeSlotBulkCreate(BaseModel):
    barber_id: str  # UUID
    slot_date: date
    time_ranges: list[dict]  # [{"start_time": "09:00", "end_time": "10:00"}, ...]
    
    @field_validator('time_ranges')
    def validate_time_ranges(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Phải có ít nhất một khoảng thời gian')
        return v

# Cập nhật time slot
class TimeSlotUpdate(BaseModel):
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_available: Optional[bool] = None
    slot_date: Optional[date] = None

# Response time slot
class TimeSlot(BaseModel):
    id: int
    barber_id: str
    slot_date: date
    start_time: time
    end_time: time
    is_available: bool

    class Config:
        from_attributes = True

# Response time slot với thông tin barber
class TimeSlotWithBarber(TimeSlot):
    barber: Optional[dict] = None