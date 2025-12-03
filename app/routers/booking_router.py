from fastapi import APIRouter, Depends, Query
from app.schemas.booking_schema import BookingCreate, BookingUpdate
from app.services import booking_service
from app.dependencies.current_user import get_current_user

router = APIRouter(prefix="/bookings", tags=["Bookings"])

# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_booking(data: BookingCreate):
    """
    Tạo booking mới
    - Yêu cầu đăng nhập
    """
    return booking_service.create_booking(data)


# ==================== Read ====================

@router.get("/", dependencies=[Depends(get_current_user)])
def get_all_bookings():
    """
    Lấy danh sách tất cả bookings
    - Yêu cầu đăng nhập (admin)
    """
    return booking_service.get_all_bookings()


@router.get("/{booking_id}", dependencies=[Depends(get_current_user)])
def get_booking(booking_id: int):
    """
    Lấy thông tin chi tiết 1 booking
    - Yêu cầu đăng nhập
    """
    return booking_service.get_booking_by_id(booking_id)


@router.get("/user/{user_id}", dependencies=[Depends(get_current_user)])
def get_user_bookings(user_id: str):
    """
    Lấy danh sách bookings của 1 user
    - Yêu cầu đăng nhập
    """
    return booking_service.get_bookings_by_user(user_id)


@router.get("/barber/{barber_id}", dependencies=[Depends(get_current_user)])
def get_barber_bookings(barber_id: str):
    """
    Lấy danh sách bookings của 1 barber
    - Yêu cầu đăng nhập
    """
    return booking_service.get_bookings_by_barber(barber_id)


@router.get("/status/{status}")
def get_bookings_by_status(status: str):
    """
    Lấy danh sách bookings theo status
    - status: pending, confirmed, completed, cancelled
    """
    return booking_service.get_bookings_by_status(status)


# ==================== Update ====================

@router.put("/{booking_id}", dependencies=[Depends(get_current_user)])
def update_booking(booking_id: int, data: BookingUpdate):
    """
    Cập nhật thông tin booking
    - Yêu cầu đăng nhập
    """
    return booking_service.update_booking(booking_id, data)


@router.patch("/{booking_id}/status", dependencies=[Depends(get_current_user)])
def update_booking_status(booking_id: int, status: str = Query(..., description="pending, confirmed, completed, cancelled")):
    """
    Cập nhật trạng thái booking
    - Yêu cầu đăng nhập
    """
    return booking_service.update_booking_status(booking_id, status)


@router.patch("/{booking_id}/cancel", dependencies=[Depends(get_current_user)])
def cancel_booking(booking_id: int):
    """
    Hủy booking (set status = cancelled)
    - Yêu cầu đăng nhập
    """
    return booking_service.cancel_booking(booking_id)


# ==================== Delete ====================

@router.delete("/{booking_id}", dependencies=[Depends(get_current_user)])
def delete_booking(booking_id: int):
    """
    Xóa booking
    - Yêu cầu đăng nhập (admin)
    """
    return booking_service.delete_booking(booking_id)