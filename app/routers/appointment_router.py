from fastapi import APIRouter, Depends, Query
from app.schemas.appointment_schema import AppointmentCreate, AppointmentUpdate
from app.services import appointment_service
from app.dependencies.current_user import get_current_user
from typing import Optional

router = APIRouter(prefix="/appointments", tags=["Appointments"])

# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_appointment(data: AppointmentCreate):
    """
    Tạo appointment mới (yêu cầu tư vấn)
    - User gửi thông tin để được tư vấn về barber
    - Yêu cầu đăng nhập
    """
    return appointment_service.create_appointment(data)


# ==================== Read ====================

@router.get("/", dependencies=[Depends(get_current_user)])
def get_all_appointments():
    """
    Lấy tất cả appointments
    - Chỉ admin mới nên truy cập
    - Yêu cầu đăng nhập
    """
    return appointment_service.get_all_appointments()


@router.get("/pending", dependencies=[Depends(get_current_user)])
def get_pending_appointments():
    """
    Lấy các appointments đang chờ xử lý
    - Cho admin kiểm tra và xử lý
    - Yêu cầu đăng nhập
    """
    return appointment_service.get_pending_appointments()


@router.get("/{appointment_id}", dependencies=[Depends(get_current_user)])
def get_appointment(appointment_id: str):
    """
    Lấy thông tin chi tiết 1 appointment
    - Yêu cầu đăng nhập
    """
    return appointment_service.get_appointment_by_id(appointment_id)


@router.get("/user/{user_id}", dependencies=[Depends(get_current_user)])
def get_user_appointments(user_id: str):
    """
    Lấy appointments của 1 user
    - Yêu cầu đăng nhập
    """
    return appointment_service.get_appointments_by_user(user_id)


@router.get("/status/{status}", dependencies=[Depends(get_current_user)])
def get_appointments_by_status(status: str):
    """
    Lấy appointments theo status
    - status: pending, confirmed, completed, cancelled
    - Yêu cầu đăng nhập
    """
    return appointment_service.get_appointments_by_status(status)


# ==================== Update ====================

@router.put("/{appointment_id}", dependencies=[Depends(get_current_user)])
def update_appointment(
    appointment_id: str, 
    data: AppointmentUpdate,
    current_user = Depends(get_current_user)
):
    """
    Cập nhật appointment (cho admin)
    - Tự động gán admin_id là người đang cập nhật
    - Yêu cầu đăng nhập
    """
    current_user_id = current_user.get('id') if isinstance(current_user, dict) else None
    return appointment_service.update_appointment(appointment_id, data, current_user_id)


@router.patch("/{appointment_id}/status", dependencies=[Depends(get_current_user)])
def update_appointment_status(
    appointment_id: str,
    status: str = Query(..., description="pending, confirmed, completed, cancelled"),
    current_user = Depends(get_current_user)
):
    """
    Cập nhật status của appointment
    - Yêu cầu đăng nhập
    """
    admin_id = current_user.get('id') if isinstance(current_user, dict) else None
    return appointment_service.update_appointment_status(appointment_id, status, admin_id)


@router.patch("/{appointment_id}/confirm", dependencies=[Depends(get_current_user)])
def confirm_appointment(
    appointment_id: str,
    admin_note: Optional[str] = Query(None, description="Ghi chú của admin"),
    current_user = Depends(get_current_user)
):
    """
    Xác nhận appointment
    - Set status = confirmed
    - Yêu cầu đăng nhập
    """
    admin_id = current_user.get('id') if isinstance(current_user, dict) else None
    return appointment_service.confirm_appointment(appointment_id, admin_id, admin_note)


@router.patch("/{appointment_id}/cancel", dependencies=[Depends(get_current_user)])
def cancel_appointment(
    appointment_id: str,
    admin_note: Optional[str] = Query(None, description="Lý do hủy")
):
    """
    Hủy appointment
    - Set status = cancelled
    - Yêu cầu đăng nhập
    """
    return appointment_service.cancel_appointment(appointment_id, admin_note)


# ==================== Delete ====================

@router.delete("/{appointment_id}", dependencies=[Depends(get_current_user)])
def delete_appointment(appointment_id: str):
    """
    Xóa appointment (soft delete)
    - Thực chất là set status = cancelled
    - Yêu cầu đăng nhập (admin)
    """
    return appointment_service.delete_appointment(appointment_id)