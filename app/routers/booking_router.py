from fastapi import APIRouter, Depends, Query, Request
from app.schemas.booking_schema import BookingCreate
from app.services import booking_service
from app.dependencies.current_user import get_current_user
from app.api.dependencies import (require_admin,require_owner,verify_self_or_admin)
from slowapi import Limiter
from slowapi.util import get_remote_address
router = APIRouter(prefix="/bookings", tags=["Bookings"])

limiter = Limiter(key_func=get_remote_address)
# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
@limiter.limit("30/minute")
def create_booking(request:Request,data: BookingCreate):
    """
    Tạo booking mới với nhiều dịch vụ
    - Yêu cầu đăng nhập
    - Tự động set time_slot thành unavailable
    """
    return booking_service.create_booking(data)


# ==================== Read ====================

@router.get("/")
@limiter.limit("120/minute")
def get_all_bookings(request:Request, current_user:dict=Depends(require_admin)):
    """
    Lấy danh sách tất cả bookings với đầy đủ thông tin
    - Yêu cầu đăng nhập (admin)
    """
    return booking_service.get_all_bookings()


@router.get("/status/{status}")
@limiter.limit("120/minute")
def get_bookings_by_status(request:Request,status: str, current_user:dict=Depends(require_admin)):
    """
    Lấy danh sách bookings theo status
    - status: confirmed, completed, cancelled
    """
    return booking_service.get_bookings_by_status(status)


@router.get("/user/{user_id}", dependencies=[Depends(get_current_user)])
@limiter.limit("120/minute")

def get_user_bookings(request:Request,user_id: str):
    """
    Lấy danh sách bookings của 1 user
    - Yêu cầu đăng nhập
    """
    return booking_service.get_bookings_by_user(user_id)


@router.get("/barber/{barber_id}")
@limiter.limit("120/minute")

def get_barber_bookings(request:Request,barber_id: str, current_user:dict=Depends(require_admin)):
    """
    Lấy danh sách bookings của 1 barber
    - Yêu cầu đăng nhập
    """
    return booking_service.get_bookings_by_barber(barber_id)


#  Route với path parameter động phải đặt CUỐI CÙNG
@router.get("/{booking_id}")
@limiter.limit("120/minute")

def get_booking(request:Request,booking_id: int,current_user:dict=Depends(require_owner)):
    """
    Lấy thông tin chi tiết 1 booking
    - Bao gồm user, barber, time_slot, và danh sách services
    - Yêu cầu đăng nhập
    """
    return booking_service.get_booking_by_id(booking_id)


# ==================== Update ====================

@router.patch("/{booking_id}/status", dependencies=[Depends(get_current_user)])
@limiter.limit("30/minute")

def update_booking_status(request:Request,
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
@limiter.limit("30/minute")
def cancel_booking(request:Request,booking_id: int):
    """
    Hủy booking (set status = cancelled và time_slot = available)
    - Yêu cầu đăng nhập
    """
    return booking_service.cancel_booking(booking_id)

@router.patch("/{booking_id}/boom")
@limiter.limit("30/minute")
def boom_booking(request: Request, booking_id:int, current_user:dict=Depends(require_owner)):
    # chỉ có owner mới được hủy, trường hợp khách hàng ko đến
    return booking_service.boom_booking(booking_id)
