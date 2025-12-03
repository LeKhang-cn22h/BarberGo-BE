from typing import Optional
from pydantic import BaseModel
from decimal import Decimal
from uuid import UUID
from datetime import datetime

class BarberBase(BaseModel):
    name: str
    location: Optional[str] = None
    area: Optional[str] = None
    address: Optional[str] = None
    imagepath: Optional[str] = None

class BarberCreate(BarberBase):
    user_id: UUID

class BarberUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    area: Optional[str] = None
    address: Optional[str] = None
    rank: Optional[Decimal] = None
    status: Optional[bool] = None

class BarberResponse(BarberBase):
    id: UUID
    rank:Optional[ Decimal] =None
    user_id: UUID
    status:Optional [bool]=None
    
    class Config:
        from_attributes = True