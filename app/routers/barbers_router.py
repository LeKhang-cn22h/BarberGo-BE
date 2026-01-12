from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.schemas.barbers_schema import BarberCreate, BarberUpdate, BarberResponse, LocationUpdate
from app.services import barbers_service
from typing import List, Optional
from uuid import UUID
from decimal import Decimal
import json

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/barbers",
    tags=["Barbers"]
)

@router.post("/", response_model=BarberResponse, status_code=201)
@limiter.limit("10/minute")

async def create_barber(
    request:Request,
    name: str = Form(...),
    user_id: str = Form(...),
):
    """Tạo barber mới"""
    print("=" * 60)
    print("🔵 CREATE BARBER REQUEST")
    print(f"name: {name}")
    print(f"user_id: {user_id}")
    print(f"name type: {type(name)}")
    print(f"user_id type: {type(user_id)}")
    print("=" * 60)
    
    try:
        barber_data = BarberCreate(
            name=name,
            user_id=UUID(user_id)
        )
        
        print(f"BarberCreate validated: {barber_data}")
        
        result = barbers_service.create_barber(barber_data)
        
        print(f"Barber created: {result}")
        print("=" * 60)
        
        return result
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"Exception: {str(e)}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/", response_model=List[BarberResponse])
@limiter.limit("120/minute")
def get_all_barbers(
    request:Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[bool] = Query(None),
    area: Optional[str] = Query(None)
):
    """Lấy danh sách tất cả barbers"""
    return barbers_service.get_all_barbers(skip=skip, limit=limit, status=status, area=area)


@router.get("/top", response_model=List[BarberResponse])
@limiter.limit("120/minute")
def get_top_barbers(request:Request,limit: int = Query(2, ge=1, le=10)):
    """Lấy danh sách barbers có rank cao nhất"""
    return barbers_service.get_top_barbers(limit=limit)


@router.get("/locations", response_model=List[str])
@limiter.limit("120/minute")
def get_locations(request:Request):
    """Lấy danh sách các location duy nhất"""
    return barbers_service.get_unique_locations()


@router.get("/areas", response_model=List[str])
@limiter.limit("120/minute")
def get_areas(request:Request):
    """Lấy danh sách các area duy nhất"""
    return barbers_service.get_unique_areas()


@router.get("/location/{location}", response_model=List[BarberResponse])
@limiter.limit("120/minute")
def get_barbers_by_location(request:Request, location: str):
    """Lấy tất cả barbers theo location"""
    return barbers_service.get_barbers_by_location(location)


@router.get("/area", response_model=List[BarberResponse])
@limiter.limit("120/minute")
def get_barbers_by_area(request:Request, area: str = Query(...)):
    """Lấy tất cả barber theo area"""
    return barbers_service.get_barbers_by_area(area)


@router.get("/user/{user_id}", response_model=List[BarberResponse])
@limiter.limit("120/minute")
def get_user_barbers(request:Request, user_id: UUID):
    """Lấy danh sách barbers của một user"""
    return barbers_service.get_barbers_by_user(user_id)


@router.get("/{barber_id}", response_model=BarberResponse)
@limiter.limit("120/minute")
def get_barber(request:Request, barber_id: UUID):
    """Lấy thông tin barber theo ID"""
    return barbers_service.get_barber_by_id(barber_id)


@router.put("/{barber_id}", response_model=BarberResponse)
@limiter.limit("30/minute")
async def update_barber(
    request:Request,
    barber_id: UUID,
    name: Optional[str] = Form(None),
    area: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    rank: Optional[str] = Form(None),
    status: Optional[bool] = Form(None),
    location: Optional[str] = Form(None, description='JSON string: {"lat": 10.8520, "lng": 106.6190}'),
    image: Optional[UploadFile] = File(None)
):
    """
    Update barber với option upload ảnh mới và location
    
    - **location**: JSON string chứa lat và lng, ví dụ: {"lat": 10.8520, "lng": 106.6190}
    """
    try:
        # Parse location nếu có
        location_obj = None
        if location:
            try:
                location_data = json.loads(location)
                location_obj = LocationUpdate(**location_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid location format. Expected JSON with lat and lng: {str(e)}"
                )
        
        update_data = BarberUpdate(
            name=name,
            location=location_obj,
            area=area,
            address=address,
            rank=Decimal(rank) if rank else None,
            status=status
        )
        return barbers_service.update_barber(barber_id, update_data, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")


# Update chỉ location
@router.patch("/location/{barber_id}", response_model=BarberResponse)
@limiter.limit("30/minute")
async def update_barber_location(
    request:Request,
    barber_id: UUID,
    lat: float = Form(..., ge=-90, le=90, description="Vĩ độ (Latitude)"),
    lng: float = Form(..., ge=-180, le=180, description="Kinh độ (Longitude)")
):
    """
    Update location (tọa độ) của barber
    
    - **lat**: Vĩ độ (10.8520 cho HCMC)
    - **lng**: Kinh độ (106.6190 cho HCMC)
    """
    return barbers_service.update_barber_location(barber_id, lat, lng)


@router.patch("/{barber_id}/deactivate", response_model=BarberResponse)
@limiter.limit("30/minute")
def deactivate_barber(request:Request, barber_id: UUID):
    """Soft delete - Vô hiệu hóa barber"""
    return barbers_service.soft_delete_barber(barber_id)

@router.patch("/{barber_id}/active", response_model=BarberResponse)
@limiter.limit("30/minute")
def deactivate_barber(request:Request,barber_id: UUID):
    """kích hoạt barber"""
    return barbers_service.active_barber(barber_id)

@router.delete("/{barber_id}")
@limiter.limit("30/minute")
def delete_barber(request:Request,barber_id: UUID):
    """Xóa barber vĩnh viễn (bao gồm cả ảnh)"""
    return barbers_service.delete_barber(barber_id)