from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from app.schemas.barbers_schema import BarberCreate, BarberUpdate, BarberResponse
from app.services import barbers_service
from typing import List, Optional
from uuid import UUID
from decimal import Decimal

router = APIRouter(
    prefix="/barbers",
    tags=["Barbers"]
)

@router.post("/", response_model=BarberResponse, status_code=201)
async def create_barber(
    name: str = Form(...),
    user_id: str = Form(...),
    location: Optional[str] = Form(None),
    area: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Tạo barber mới với upload ảnh"""
    try:
        barber_data = BarberCreate(
            name=name,
            location=location,
            area=area,
            address=address,
            user_id=UUID(user_id)
        )
        return barbers_service.create_barber(barber_data, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")


@router.get("/", response_model=List[BarberResponse])
def get_all_barbers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[bool] = Query(None),
    area: Optional[str] = Query(None)
):
    """Lấy danh sách tất cả barbers"""
    return barbers_service.get_all_barbers(skip=skip, limit=limit, status=status, area=area)


@router.get("/top", response_model=List[BarberResponse])
def get_top_barbers(limit: int = Query(2, ge=1, le=10)):
    """Lấy danh sách barbers có rank cao nhất"""
    return barbers_service.get_top_barbers(limit=limit)


@router.get("/locations", response_model=List[str])
def get_locations():
    """Lấy danh sách các location duy nhất"""
    return barbers_service.get_unique_locations()


@router.get("/areas", response_model=List[str])
def get_areas():
    """Lấy danh sách các area duy nhất"""
    return barbers_service.get_unique_areas()


@router.get("/location/{location}", response_model=List[BarberResponse])
def get_barbers_by_location(location: str):
    """Lấy tất cả barbers theo location"""
    return barbers_service.get_barbers_by_location(location)


@router.get("/area/{area}", response_model=List[BarberResponse])
def get_barbers_by_area(area: str):
    """Lấy tất cả barbers theo area"""
    return barbers_service.get_barbers_by_area(area)


@router.get("/user/{user_id}", response_model=List[BarberResponse])
def get_user_barbers(user_id: UUID):
    """Lấy danh sách barbers của một user"""
    return barbers_service.get_barbers_by_user(user_id)


@router.get("/{barber_id}", response_model=BarberResponse)
def get_barber(barber_id: UUID):
    """Lấy thông tin barber theo ID"""
    return barbers_service.get_barber_by_id(barber_id)


@router.put("/{barber_id}", response_model=BarberResponse)
async def update_barber(
    barber_id: UUID,
    name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    area: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    rank: Optional[str] = Form(None),
    status: Optional[bool] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Update barber với option upload ảnh mới"""
    try:
        update_data = BarberUpdate(
            name=name,
            location=location,
            area=area,
            address=address,
            rank=Decimal(rank) if rank else None,
            status=status
        )
        return barbers_service.update_barber(barber_id, update_data, image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")


@router.patch("/{barber_id}/deactivate", response_model=BarberResponse)
def deactivate_barber(barber_id: UUID):
    """Soft delete - Vô hiệu hóa barber"""
    return barbers_service.soft_delete_barber(barber_id)


@router.delete("/{barber_id}")
def delete_barber(barber_id: UUID):
    """Xóa barber vĩnh viễn (bao gồm cả ảnh)"""
    return barbers_service.delete_barber(barber_id)