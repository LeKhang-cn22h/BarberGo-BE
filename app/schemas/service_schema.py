from pydantic import BaseModel
from typing import Optional

# Tạo service mới
class ServiceCreate(BaseModel):
    barber_id: str  # UUID
    service_name: str
    price: int
    duration_min: int

# Cập nhật service
class ServiceUpdate(BaseModel):
    barber_id: Optional[str] = None
    service_name: Optional[str] = None
    price: Optional[int] = None
    duration_min: Optional[int] = None

# Response service
class Service(BaseModel):
    id: int
    barber_id: Optional[str] = None
    service_name: Optional[str] = None
    price: Optional[int] = None
    duration_min: Optional[int] = None

    class Config:
        from_attributes = True