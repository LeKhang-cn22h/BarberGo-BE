from app.database.supabase_client import supabase
from app.schemas.booking_schema import BookingCreate, BookingUpdate
from fastapi import HTTPException
from datetime import datetime

# ==================== Create Booking ====================

def create_booking(data: BookingCreate):
    """Tạo booking mới"""
    try:
        # Kiểm tra barber có tồn tại không
        barber_check = supabase.table("barbers").select("id").eq("id", data.barber_id).execute()
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        # Kiểm tra user có tồn tại không
        user_check = supabase.table("users").select("id").eq("id", data.user_id).execute()
        if not user_check.data:
            raise HTTPException(status_code=404, detail="User không tồn tại")
        
        # Kiểm tra service có tồn tại không
        service_check = supabase.table("services").select("id").eq("id", data.service_id).execute()
        if not service_check.data:
            raise HTTPException(status_code=404, detail="Dịch vụ không tồn tại")
        
        # Kiểm tra trùng lịch (cùng barber, cùng thời gian)
        existing_booking = supabase.table("bookings")\
            .select("*")\
            .eq("barber_id", data.barber_id)\
            .eq("date_time", data.date_time.isoformat())\
            .in_("status", ["pending", "confirmed"])\
            .execute()
        
        if existing_booking.data:
            raise HTTPException(status_code=400, detail="Barber đã có lịch hẹn vào thời gian này")
        
        # Tạo booking
        booking_data = {
            "barber_id": data.barber_id,
            "user_id": data.user_id,
            "service_id": data.service_id,
            "date_time": data.date_time.isoformat(),
            "status": data.status
        }
        
        response = supabase.table("bookings").insert(booking_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo booking thất bại")
        
        return {
            "message": "Đặt lịch thành công",
            "booking": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo booking thất bại: {str(e)}")


# ==================== Get Bookings ====================

def get_all_bookings():
    """Lấy danh sách tất cả bookings"""
    try:
        response = supabase.table("bookings")\
            .select("*, barbers(full_name), users(full_name, email), services(service_name, price)")\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_booking_by_id(booking_id: int):
    """Lấy thông tin booking theo ID"""
    try:
        response = supabase.table("bookings")\
            .select("*, barbers(full_name), users(full_name, email), services(service_name, price, duration_min)")\
            .eq("id", booking_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_bookings_by_user(user_id: str):
    """Lấy danh sách bookings của 1 user"""
    try:
        response = supabase.table("bookings")\
            .select("*, barbers(full_name), services(service_name, price, duration_min)")\
            .eq("user_id", user_id)\
            .order("date_time", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_bookings_by_barber(barber_id: str):
    """Lấy danh sách bookings của 1 barber"""
    try:
        response = supabase.table("bookings")\
            .select("*, users(full_name, email, phone), services(service_name, price, duration_min)")\
            .eq("barber_id", barber_id)\
            .order("date_time", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_bookings_by_status(status: str):
    """Lấy danh sách bookings theo status"""
    try:
        response = supabase.table("bookings")\
            .select("*, barbers(full_name), users(full_name, email), services(service_name, price)")\
            .eq("status", status)\
            .order("date_time", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


# ==================== Update Booking ====================

def update_booking(booking_id: int, data: BookingUpdate):
    """Cập nhật thông tin booking"""
    try:
        # Chỉ update các field không None
        update_data = {}
        
        if data.barber_id is not None:
            # Kiểm tra barber có tồn tại không
            barber_check = supabase.table("barbers").select("id").eq("id", data.barber_id).execute()
            if not barber_check.data:
                raise HTTPException(status_code=404, detail="Barber không tồn tại")
            update_data["barber_id"] = data.barber_id
        
        if data.user_id is not None:
            # Kiểm tra user có tồn tại không
            user_check = supabase.table("users").select("id").eq("id", data.user_id).execute()
            if not user_check.data:
                raise HTTPException(status_code=404, detail="User không tồn tại")
            update_data["user_id"] = data.user_id
        
        if data.service_id is not None:
            # Kiểm tra service có tồn tại không
            service_check = supabase.table("services").select("id").eq("id", data.service_id).execute()
            if not service_check.data:
                raise HTTPException(status_code=404, detail="Dịch vụ không tồn tại")
            update_data["service_id"] = data.service_id
        
        if data.date_time is not None:
            update_data["date_time"] = data.date_time.isoformat()
        
        if data.status is not None:
            update_data["status"] = data.status
        
        if not update_data:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
        
        response = supabase.table("bookings").update(update_data).eq("id", booking_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        return {
            "message": "Cập nhật booking thành công",
            "booking": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


def update_booking_status(booking_id: int, status: str):
    """Cập nhật status của booking"""
    try:
        allowed_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
        if status not in allowed_statuses:
            raise HTTPException(status_code=400, detail=f"Status không hợp lệ. Phải là: {allowed_statuses}")
        
        response = supabase.table("bookings").update({"status": status}).eq("id", booking_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        return {
            "message": f"Cập nhật trạng thái thành '{status}' thành công",
            "booking": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


# ==================== Delete Booking ====================

def delete_booking(booking_id: int):
    """Xóa booking"""
    try:
        response = supabase.table("bookings").delete().eq("id", booking_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        return {"message": "Xóa booking thành công"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xóa thất bại: {str(e)}")


def cancel_booking(booking_id: int):
    """Hủy booking (update status thành cancelled)"""
    return update_booking_status(booking_id, "cancelled")