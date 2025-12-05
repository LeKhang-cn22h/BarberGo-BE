from app.database.supabase_client import supabase
from app.schemas.appointment_schema import AppointmentCreate, AppointmentUpdate
from fastapi import HTTPException
from datetime import datetime

# ==================== Create Appointment ====================

def create_appointment(data: AppointmentCreate):
    """Tạo appointment mới (yêu cầu tư vấn)"""
    try:
        # Kiểm tra user có tồn tại không
        user_check = supabase.table("users")\
            .select("id, full_name, email")\
            .eq("id", data.user_id)\
            .execute()
        
        if not user_check.data:
            raise HTTPException(status_code=404, detail="User không tồn tại")
        
        # Tạo appointment
        appointment_data = {
            "user_id": data.user_id,
            "name_barber": data.name_barber,
            "phone": data.phone,
            "email": data.email,
            "status": data.status
        }
        
        response = supabase.table("appointments").insert(appointment_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo appointment thất bại")
        
        return {
            "message": "Gửi yêu cầu tư vấn thành công. Chúng tôi sẽ liên hệ với bạn sớm!",
            "appointment": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo appointment thất bại: {str(e)}")


# ==================== Get Appointments ====================

def get_all_appointments():
    """Lấy tất cả appointments (cho admin)"""
    try:
        response = supabase.table("appointments")\
            .select("""
                *,
                users!appointments_user_id_fkey(full_name, email, phone),
                admin:users!appointments_admin_id_fkey(full_name, email)
            """)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_appointment_by_id(appointment_id: str):
    """Lấy thông tin appointment theo ID"""
    try:
        response = supabase.table("appointments")\
            .select("""
                *,
                users!appointments_user_id_fkey(full_name, email, phone),
                admin:users!appointments_admin_id_fkey(full_name, email)
            """)\
            .eq("id", appointment_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy appointment")
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_appointments_by_user(user_id: str):
    """Lấy appointments của 1 user"""
    try:
        response = supabase.table("appointments")\
            .select("""
                *,
                admin:users!appointments_admin_id_fkey(full_name)
            """)\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_appointments_by_status(status: str):
    """Lấy appointments theo status"""
    try:
        allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
        if status not in allowed_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Status không hợp lệ. Phải là: {allowed_statuses}"
            )
        
        response = supabase.table("appointments")\
            .select("""
                *,
                users!appointments_user_id_fkey(full_name, email, phone)
            """)\
            .eq("status", status)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_pending_appointments():
    """Lấy các appointments đang chờ xử lý"""
    return get_appointments_by_status("pending")


# ==================== Update Appointment ====================

def update_appointment(appointment_id: str, data: AppointmentUpdate, current_user_id: str):
    """Cập nhật appointment (cho admin)"""
    try:
        # Kiểm tra appointment có tồn tại không
        appointment_check = supabase.table("appointments")\
            .select("id")\
            .eq("id", appointment_id)\
            .execute()
        
        if not appointment_check.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy appointment")
        
        # Chuẩn bị dữ liệu update
        update_data = {"updated_at": datetime.now().isoformat()}
        
        if data.status is not None:
            update_data["status"] = data.status
        
        if data.admin_note is not None:
            update_data["admin_note"] = data.admin_note
        
        if data.admin_id is not None:
            # Kiểm tra admin có tồn tại không
            admin_check = supabase.table("users")\
                .select("id, role")\
                .eq("id", data.admin_id)\
                .execute()
            
            if not admin_check.data:
                raise HTTPException(status_code=404, detail="Admin không tồn tại")
            
            if admin_check.data[0].get('role') not in ['admin', 'staff']:
                raise HTTPException(status_code=403, detail="User không có quyền admin")
            
            update_data["admin_id"] = data.admin_id
        elif current_user_id:
            # Tự động set admin_id là người đang cập nhật
            update_data["admin_id"] = current_user_id
        
        response = supabase.table("appointments")\
            .update(update_data)\
            .eq("id", appointment_id)\
            .execute()
        
        return {
            "message": "Cập nhật appointment thành công",
            "appointment": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


def update_appointment_status(appointment_id: str, status: str, admin_id: str = None):
    """Cập nhật status của appointment"""
    try:
        allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
        if status not in allowed_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Status không hợp lệ. Phải là: {allowed_statuses}"
            )
        
        update_data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if admin_id:
            update_data["admin_id"] = admin_id
        
        response = supabase.table("appointments")\
            .update(update_data)\
            .eq("id", appointment_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy appointment")
        
        return {
            "message": f"Cập nhật trạng thái thành '{status}' thành công",
            "appointment": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


def confirm_appointment(appointment_id: str, admin_id: str = None, admin_note: str = None):
    """Xác nhận appointment"""
    try:
        update_data = {
            "status": "confirmed",
            "updated_at": datetime.now().isoformat()
        }
        
        if admin_id:
            update_data["admin_id"] = admin_id
        
        if admin_note:
            update_data["admin_note"] = admin_note
        
        response = supabase.table("appointments")\
            .update(update_data)\
            .eq("id", appointment_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy appointment")
        
        return {
            "message": "Xác nhận appointment thành công",
            "appointment": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xác nhận thất bại: {str(e)}")


def cancel_appointment(appointment_id: str, admin_note: str = None):
    """Hủy appointment"""
    try:
        update_data = {
            "status": "cancelled",
            "updated_at": datetime.now().isoformat()
        }
        
        if admin_note:
            update_data["admin_note"] = admin_note
        
        response = supabase.table("appointments")\
            .update(update_data)\
            .eq("id", appointment_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy appointment")
        
        return {
            "message": "Hủy appointment thành công",
            "appointment": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hủy thất bại: {str(e)}")


# ==================== Delete Appointment ====================

def delete_appointment(appointment_id: str):
    """Xóa appointment (soft delete bằng cách set status = cancelled thay vì xóa hẳn)"""
    return cancel_appointment(appointment_id, "Đã xóa bởi admin")