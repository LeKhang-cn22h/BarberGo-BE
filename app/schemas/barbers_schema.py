from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator
from decimal import Decimal
from uuid import UUID
import struct

# 1. Class phụ để hứng dữ liệu update lồng nhau
class LocationUpdate(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

# 2. Base
class BarberBase(BaseModel):
    name: str

# 3. Create (Input)
class BarberCreate(BarberBase):
    user_id: UUID

# 4. Update (Input)
class BarberUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[LocationUpdate] = None 
    area: Optional[str] = None
    address: Optional[str] = None
    rank: Optional[Decimal] = None
    status: Optional[bool] = None

# 5. Response (Output)
class BarberResponse(BarberBase):
    id: UUID
    rank: Optional[Decimal] = None
    user_id: UUID
    status: Optional[bool] = None
    imagepath: Optional[str] = None
    address: Optional[str] = None
    area: Optional[str] = None

    # Trường lat/lng để trả về Flutter
    lat: Optional[float] = None
    lng: Optional[float] = None

    # Trường location ẩn (nhận WKB từ DB)
    location: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        from_attributes = True

    @model_validator(mode='after')
    def parse_location(self):
        """Parse location từ WKB hoặc POINT format"""
        raw_loc = self.location
        
        if not raw_loc:
            return self
        
        try:
            # Case 1: WKB Hex String (như của bạn)
            if isinstance(raw_loc, str) and len(raw_loc) > 20 and not raw_loc.startswith('POINT'):
                # Decode WKB hex string
                wkb_bytes = bytes.fromhex(raw_loc)
                
                # WKB format cho POINT:
                # - Byte 0: byte order (01 = little endian)
                # - Bytes 1-4: geometry type (01000000 = Point)
                # - Bytes 5-8: SRID (E6100000 = 4326)
                # - Bytes 9-16: X coordinate (longitude) - double
                # - Bytes 17-24: Y coordinate (latitude) - double
                
                # Skip first 9 bytes (byte order + type + SRID)
                # Then read 2 doubles (8 bytes each)
                lng = struct.unpack('<d', wkb_bytes[9:17])[0]  # X = longitude
                lat = struct.unpack('<d', wkb_bytes[17:25])[0]  # Y = latitude
                
                self.lng = round(lng, 7)
                self.lat = round(lat, 7)
            
            # Case 2: POINT text format (backup)
            elif isinstance(raw_loc, str) and raw_loc.startswith('POINT'):
                content = raw_loc.replace('POINT(', '').replace(')', '').strip()
                parts = content.split()
                if len(parts) == 2:
                    self.lng = float(parts[0])
                    self.lat = float(parts[1])
        
        except Exception as e:
            # Log error nhưng không crash
            print(f"Error parsing location: {e}")
            self.lat = None
            self.lng = None
        
        return self