from fastapi import APIRouter, Depends
from app.schemas.service_schema import ServiceCreate, ServiceUpdate
from app.services import service_service
from app.dependencies.current_user import get_current_user

router = APIRouter(prefix="/services", tags=["Services"])

# ==================== Create ====================

@router.post("/", dependencies=[Depends(get_current_user)])
def create_service(data: ServiceCreate):
    """
    Tạo dịch vụ mới
    - Yêu cầu đăng nhập
    """
    return service_service.create_service(data)


# ==================== Read ====================

@router.get("/")
def get_all_services():
    """
    Lấy danh sách tất cả dịch vụ
    - Không cần đăng nhập
    """
    return service_service.get_all_services()


@router.get("/{service_id}")
def get_service(service_id: int):
    """
    Lấy thông tin chi tiết 1 dịch vụ
    - Không cần đăng nhập
    """
    return service_service.get_service_by_id(service_id)


@router.get("/barber/{barber_id}")
def get_services_by_barber(barber_id: str):
    """
    Lấy danh sách dịch vụ của 1 barber
    - Không cần đăng nhập
    """
    return service_service.get_services_by_barber(barber_id)


# ==================== Update ====================

@router.put("/{service_id}", dependencies=[Depends(get_current_user)])
def update_service(service_id: int, data: ServiceUpdate):
    """
    Cập nhật thông tin dịch vụ
    - Yêu cầu đăng nhập
    """
    return service_service.update_service(service_id, data)


# ==================== Delete ====================

@router.patch("/{service_id}/delete", dependencies=[Depends(get_current_user)])
def delete_service(service_id: int):
    """
    Xóa mềm dịch vụ (set status = false)
    - Không xóa hẳn khỏi database
    - Yêu cầu đăng nhập
    """
    return service_service.delete_service(service_id)


@router.patch("/{service_id}/restore", dependencies=[Depends(get_current_user)])
def restore_service(service_id: int):
    """
    Khôi phục dịch vụ đã xóa (set status = true)
    - Yêu cầu đăng nhập
    """
    return service_service.restore_service(service_id)


@router.patch("/{service_id}/toggle-status", dependencies=[Depends(get_current_user)])
def toggle_service_status(service_id: int):
    """
    Chuyển đổi trạng thái active/inactive của dịch vụ
    - Nếu đang active (true) -> inactive (false)
    - Nếu đang inactive (false) -> active (true)
    - Yêu cầu đăng nhập
    """
    return service_service.toggle_service_status(service_id)