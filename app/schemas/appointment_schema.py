from pydantic import BaseModel, field_validator, EmailStr
from typing import Optional
from datetime import datetime

# Tạo appointment mới
class AppointmentCreate(BaseModel):
    user_id: str  # UUID
    name_barber: str
    phone: str
    email: EmailStr
    status: Optional[str] = "pending"
    
    @field_validator('status')
    def validate_status(cls, v):
        allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
        if v not in allowed_statuses:
            raise ValueError(f'Status phải là một trong: {allowed_statuses}')
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        # Loại bỏ khoảng trắng và ký tự đặc biệt
        phone = ''.join(filter(str.isdigit, v))
        if len(phone) < 10:
            raise ValueError('Số điện thoại phải có ít nhất 10 chữ số')
        return v

# Cập nhật appointment (cho admin)
class AppointmentUpdate(BaseModel):
    status: Optional[str] = None
    admin_note: Optional[str] = None
    admin_id: Optional[str] = None  # UUID
    
    @field_validator('status')
    def validate_status(cls, v):
        if v is not None:
            allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
            if v not in allowed_statuses:
                raise ValueError(f'Status phải là một trong: {allowed_statuses}')
        return v

# Response appointment
class Appointment(BaseModel):
    id: str
    user_id: str
    name_barber: str
    phone: str
    email: str
    status: str
    created_at: datetime
    updated_at: datetime
    admin_note: Optional[str] = None
    admin_id: Optional[str] = None

    class Config:
        from_attributes = True

# Response appointment với thông tin user và admin
class AppointmentWithDetails(Appointment):
    user: Optional[dict] = None
    admin: Optional[dict] = None