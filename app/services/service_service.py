from app.database.supabase_client import supabase
from app.schemas.service_schema import ServiceCreate, ServiceUpdate
from fastapi import HTTPException

# ==================== Create Service ====================

def create_service(data: ServiceCreate):
    """Tạo dịch vụ mới"""
    try:
        # Kiểm tra barber có tồn tại không
        barber_check = supabase.table("barbers").select("id").eq("id", data.barber_id).execute()
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        # Tạo service
        service_data = {
            "barber_id": data.barber_id,
            "service_name": data.service_name,
            "price": data.price,
            "duration_min": data.duration_min
        }
        
        response = supabase.table("services").insert(service_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo dịch vụ thất bại")
        
        return {
            "message": "Tạo dịch vụ thành công",
            "service": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo dịch vụ thất bại: {str(e)}")


# ==================== Get Services ====================

def get_all_services():
    """Lấy danh sách tất cả dịch vụ"""
    try:
        response = supabase.table("services").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_service_by_id(service_id: int):
    """Lấy thông tin dịch vụ theo ID"""
    try:
        response = supabase.table("services").select("*").eq("id", service_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy dịch vụ")
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_services_by_barber(barber_id: str):
    """Lấy danh sách dịch vụ của 1 barber"""
    try:
        response = supabase.table("services").select("*").eq("barber_id", barber_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


# ==================== Update Service ====================

def update_service(service_id: int, data: ServiceUpdate):
    """Cập nhật thông tin dịch vụ"""
    try:
        # Chỉ update các field không None
        update_data = {k: v for k, v in data.dict().items() if v is not None}
        
        if not update_data:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
        
        # Nếu update barber_id, kiểm tra barber có tồn tại không
        if "barber_id" in update_data:
            barber_check = supabase.table("barbers").select("id").eq("id", update_data["barber_id"]).execute()
            if not barber_check.data:
                raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        response = supabase.table("services").update(update_data).eq("id", service_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy dịch vụ")
        
        return {
            "message": "Cập nhật dịch vụ thành công",
            "service": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


# ==================== Delete Service ====================

def delete_service(service_id: int):
    """Xóa dịch vụ"""
    try:
        response = supabase.table("services").delete().eq("id", service_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy dịch vụ")
        
        return {"message": "Xóa dịch vụ thành công"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xóa thất bại: {str(e)}")