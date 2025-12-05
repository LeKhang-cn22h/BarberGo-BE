from fastapi import APIRouter, Depends, Query
from app.schemas.time_slot_schema import TimeSlotCreate, TimeSlotUpdate, TimeSlotBulkCreate
from app.services import time_slot_service
from app.dependencies.current_user import get_current_user
from typing import Optional

router = APIRouter(prefix="/time-slots", tags=["Time Slots"])

# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_time_slot(data: TimeSlotCreate):
    """
    Tạo time slot mới
    - Kiểm tra barber, working days, opening hours
    - Yêu cầu đăng nhập
    """
    return time_slot_service.create_time_slot(data)


@router.post("/bulk", dependencies=[Depends(get_current_user)])
def create_time_slots_bulk(data: TimeSlotBulkCreate):
    """
    Tạo nhiều time slots cùng lúc cho 1 ngày
    - Yêu cầu đăng nhập
    """
    return time_slot_service.create_time_slots_bulk(data)


# ==================== Read ====================

@router.get("/")
def get_all_time_slots():
    """
    Lấy tất cả time slots
    """
    return time_slot_service.get_all_time_slots()


@router.get("/{time_slot_id}")
def get_time_slot(time_slot_id: int):
    """
    Lấy thông tin chi tiết 1 time slot
    """
    return time_slot_service.get_time_slot_by_id(time_slot_id)


@router.get("/barber/{barber_id}")
def get_barber_time_slots(
    barber_id: str,
    slot_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    is_available: Optional[bool] = Query(None, description="Filter by availability")
):
    """
    Lấy time slots của barber
    - Có thể filter theo ngày và availability
    """
    return time_slot_service.get_time_slots_by_barber(barber_id, slot_date, is_available)


@router.get("/available/list")
def get_available_time_slots(
    barber_id: Optional[str] = Query(None),
    slot_date: Optional[str] = Query(None, description="Format: YYYY-MM-DD")
):
    """
    Lấy các time slots còn trống (available)
    - Có thể filter theo barber và ngày
    """
    return time_slot_service.get_available_time_slots(barber_id, slot_date)


# ==================== Update ====================

@router.put("/{time_slot_id}", dependencies=[Depends(get_current_user)])
def update_time_slot(time_slot_id: int, data: TimeSlotUpdate):
    """
    Cập nhật thông tin time slot
    - Không cho phép sửa nếu đang có booking
    - Yêu cầu đăng nhập
    """
    return time_slot_service.update_time_slot(time_slot_id, data)


@router.patch("/{time_slot_id}/toggle", dependencies=[Depends(get_current_user)])
def toggle_time_slot_availability(time_slot_id: int):
    """
    Chuyển đổi trạng thái available/unavailable
    - Yêu cầu đăng nhập
    """
    return time_slot_service.toggle_time_slot_availability(time_slot_id)


# ==================== Delete ====================

@router.delete("/{time_slot_id}", dependencies=[Depends(get_current_user)])
def delete_time_slot(time_slot_id: int):
    """
    Xóa time slot
    - Chỉ xóa được nếu chưa có booking
    - Yêu cầu đăng nhập
    """
    return time_slot_service.delete_time_slot(time_slot_id)