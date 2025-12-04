from fastapi import APIRouter, Depends, Query
from app.schemas.booking_schema import BookingCreate
from app.services import booking_service
from app.dependencies.current_user import get_current_user

router = APIRouter(prefix="/bookings", tags=["Bookings"])

# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_booking(data: BookingCreate):
    """
    Tạo booking mới với nhiều dịch vụ
    - Yêu cầu đăng nhập
    - Tự động set time_slot thành unavailable
    """
    return booking_service.create_booking(data)


# ==================== Read ====================
#  Thứ tự quan trọng: Routes CỤ THỂ phải đứng TRƯỚC routes CHUNG

@router.get("/", dependencies=[Depends(get_current_user)])
def get_all_bookings():
    """
    Lấy danh sách tất cả bookings với đầy đủ thông tin
    - Yêu cầu đăng nhập (admin)
    """
    return booking_service.get_all_bookings()


@router.get("/status/{status}")
def get_bookings_by_status(status: str):
    """
    Lấy danh sách bookings theo status
    - status: confirmed, completed, cancelled
    """
    return booking_service.get_bookings_by_status(status)


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


#  Route với path parameter động phải đặt CUỐI CÙNG
@router.get("/{booking_id}", dependencies=[Depends(get_current_user)])
def get_booking(booking_id: int):
    """
    Lấy thông tin chi tiết 1 booking
    - Bao gồm user, barber, time_slot, và danh sách services
    - Yêu cầu đăng nhập
    """
    return booking_service.get_booking_by_id(booking_id)


# ==================== Update ====================
#  Routes với path cụ thể (/{booking_id}/status) phải đứng TRƯỚC /{booking_id}

@router.patch("/{booking_id}/status", dependencies=[Depends(get_current_user)])
def update_booking_status(
    booking_id: int, 
    status: str = Query(..., description="confirmed, completed, cancelled")
):
    """
    Cập nhật trạng thái booking
    - Nếu cancel, tự động set time_slot thành available
    - Yêu cầu đăng nhập
    """
    return booking_service.update_booking_status(booking_id, status)


@router.patch("/{booking_id}/cancel", dependencies=[Depends(get_current_user)])
def cancel_booking(booking_id: int):
    """
    Hủy booking (set status = cancelled và time_slot = available)
    - Yêu cầu đăng nhập
    """
    return booking_service.cancel_booking(booking_id)