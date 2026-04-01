from fastapi import APIRouter, Depends, Request
from app.schemas.service_schema import ServiceCreate, ServiceUpdate
from app.services import service_service
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/services", tags=["Services"])
from app.api.dependencies import (
    require_admin, 
    require_owner
)
# ==================== Create ====================

@router.post("/")
@limiter.limit("10/minute")

def create_service(request:Request,data: ServiceCreate, current_user:dict=Depends(require_owner)):
    """
    Tạo dịch vụ mới
    """
    return service_service.create_service(data)


# ==================== Read ====================

@router.get("/")
@limiter.limit("120/minute")
def get_all_services(request:Request):
    """
    Lấy danh sách tất cả dịch vụ
    """
    return service_service.get_all_services()


@router.get("/{service_id}")
@limiter.limit("120/minute")
def get_service(request:Request,service_id: int):
    """
    Lấy thông tin chi tiết 1 dịch vụ
    - Không cần đăng nhập
    """
    return service_service.get_service_by_id(service_id)


@router.get("/barber/{barber_id}")
@limiter.limit("120/minute")
def get_services_by_barber(request:Request,barber_id: str):
    """
    Lấy danh sách dịch vụ của 1 barber
    - Không cần đăng nhập
    """
    return service_service.get_services_by_barber(barber_id)

@router.get("/pricerange/{barber_id}")
@limiter.limit("120/minute")
def get_price_range(request:Request,barber_id:str):
    "Lấy khoảng giá barber"
    return service_service.get_min_max_price_by_barber(barber_id)

# ==================== Update ====================

@router.put("/{service_id}")
@limiter.limit("30/minute")
def update_service(request:Request, service_id: int, data: ServiceUpdate,current_user:dict=Depends(require_owner)):
    """
    Cập nhật thông tin dịch vụ
    - Yêu cầu đăng nhập
    """
    return service_service.update_service(service_id, data)


# ==================== Delete ====================

@router.patch("/{service_id}/delete")
@limiter.limit("30/minute")
def delete_service(request:Request,service_id: int,current_user:dict=Depends(require_owner)):
    """
    Xóa mềm dịch vụ (set status = false)
    - Không xóa hẳn khỏi database
    - Yêu cầu đăng nhập
    """
    return service_service.delete_service(service_id)


@router.patch("/{service_id}/restore")
@limiter.limit("30/minute")
def restore_service(request:Request,service_id: int, current_user:dict=Depends(require_owner)):
    """
    Khôi phục dịch vụ đã xóa (set status = true)
    - Yêu cầu đăng nhập
    """
    return service_service.restore_service(service_id)


@router.patch("/{service_id}/toggle-status")
@limiter.limit("30/minute")
def toggle_service_status(request:Request,service_id: int,current_user:dict=Depends(require_owner)):
    """
    Chuyển đổi trạng thái active/inactive của dịch vụ
    - Nếu đang active (true) -> inactive (false)
    - Nếu đang inactive (false) -> active (true)
    - Yêu cầu đăng nhập
    """
    return service_service.toggle_service_status(service_id)