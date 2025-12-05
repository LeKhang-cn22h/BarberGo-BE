from app.database.supabase_client import supabase
from app.schemas.time_slot_schema import TimeSlotCreate, TimeSlotUpdate, TimeSlotBulkCreate
from fastapi import HTTPException
from datetime import datetime, time, timedelta

# ==================== Create Time Slot ====================

def create_time_slot(data: TimeSlotCreate):
    """Tạo time slot mới"""
    try:
        # Kiểm tra barber có tồn tại không
        barber_check = supabase.table("barbers")\
            .select("id, status, opening_time, closing_time, working_days")\
            .eq("id", data.barber_id)\
            .execute()
        
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        barber = barber_check.data[0]
        
        # Kiểm tra barber có đang hoạt động không
        if not barber.get('status'):
            raise HTTPException(status_code=400, detail="Barber hiện không hoạt động")
        
        # Kiểm tra ngày có trong working_days không
        day_name = data.slot_date.strftime('%A')
        working_days = barber.get('working_days', [])
        if day_name not in working_days:
            raise HTTPException(status_code=400, detail=f"Barber không làm việc vào {day_name}")
        
        # Kiểm tra thời gian có nằm trong giờ mở cửa không
        opening_time = barber.get('opening_time')
        closing_time = barber.get('closing_time')
        
        if opening_time and closing_time:
            if data.start_time < opening_time or data.end_time > closing_time:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Thời gian phải nằm trong giờ mở cửa: {opening_time} - {closing_time}"
                )
        
        # Kiểm tra trùng lặp time slot
        overlap_check = supabase.table("time_slots")\
            .select("id")\
            .eq("barber_id", data.barber_id)\
            .eq("slot_date", data.slot_date.isoformat())\
            .or_(f"and(start_time.lte.{data.start_time},end_time.gt.{data.start_time}),"
                 f"and(start_time.lt.{data.end_time},end_time.gte.{data.end_time}),"
                 f"and(start_time.gte.{data.start_time},end_time.lte.{data.end_time})")\
            .execute()
        
        if overlap_check.data:
            raise HTTPException(status_code=400, detail="Time slot bị trùng với slot đã có")
        
        # Tạo time slot
        time_slot_data = {
            "barber_id": data.barber_id,
            "slot_date": data.slot_date.isoformat(),
            "start_time": data.start_time.isoformat(),
            "end_time": data.end_time.isoformat(),
            "is_available": data.is_available
        }
        
        response = supabase.table("time_slots").insert(time_slot_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo time slot thất bại")
        
        return {
            "message": "Tạo time slot thành công",
            "time_slot": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo time slot thất bại: {str(e)}")


def create_time_slots_bulk(data: TimeSlotBulkCreate):
    """Tạo nhiều time slots cùng lúc cho 1 ngày"""
    try:
        # Kiểm tra barber
        barber_check = supabase.table("barbers")\
            .select("id, status")\
            .eq("id", data.barber_id)\
            .execute()
        
        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")
        
        if not barber_check.data[0].get('status'):
            raise HTTPException(status_code=400, detail="Barber hiện không hoạt động")
        
        # Tạo danh sách time slots
        time_slots = []
        for time_range in data.time_ranges:
            time_slots.append({
                "barber_id": data.barber_id,
                "slot_date": data.slot_date.isoformat(),
                "start_time": time_range['start_time'],
                "end_time": time_range['end_time'],
                "is_available": True
            })
        
        response = supabase.table("time_slots").insert(time_slots).execute()
        
        return {
            "message": f"Tạo thành công {len(response.data)} time slots",
            "time_slots": response.data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo time slots thất bại: {str(e)}")


# ==================== Get Time Slots ====================

def get_all_time_slots():
    """Lấy tất cả time slots"""
    try:
        response = supabase.table("time_slots")\
            .select("*, barbers(id, name, address)")\
            .order("slot_date", desc=False)\
            .order("start_time", desc=False)\
            .execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_time_slot_by_id(time_slot_id: int):
    """Lấy thông tin time slot theo ID"""
    try:
        response = supabase.table("time_slots")\
            .select("*, barbers(id, name, address)")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        return response.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_time_slots_by_barber(barber_id: str, slot_date: str = None, is_available: bool = None):
    """Lấy time slots của barber, có thể filter theo ngày và availability"""
    try:
        query = supabase.table("time_slots")\
            .select("*")\
            .eq("barber_id", barber_id)
        
        if slot_date:
            query = query.eq("slot_date", slot_date)
        
        if is_available is not None:
            query = query.eq("is_available", is_available)
        
        response = query.order("slot_date", desc=False)\
            .order("start_time", desc=False)\
            .execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_available_time_slots(barber_id: str = None, slot_date: str = None):
    """Lấy các time slots còn trống (available)"""
    try:
        query = supabase.table("time_slots")\
            .select("*, barbers(id, name, address)")\
            .eq("is_available", True)
        
        if barber_id:
            query = query.eq("barber_id", barber_id)
        
        if slot_date:
            query = query.eq("slot_date", slot_date)
        
        response = query.order("slot_date", desc=False)\
            .order("start_time", desc=False)\
            .execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


# ==================== Update Time Slot ====================

def update_time_slot(time_slot_id: int, data: TimeSlotUpdate):
    """Cập nhật thông tin time slot"""
    try:
        # Lấy time slot hiện tại
        current_slot = supabase.table("time_slots")\
            .select("*")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not current_slot.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        # Kiểm tra xem có booking nào đang sử dụng slot này không
        if data.is_available == False or data.start_time or data.end_time or data.slot_date:
            booking_check = supabase.table("bookings")\
                .select("id")\
                .eq("time_slot_id", time_slot_id)\
                .in_("status", ["confirmed"])\
                .execute()
            
            if booking_check.data:
                raise HTTPException(
                    status_code=400, 
                    detail="Không thể sửa time slot đang có booking"
                )
        
        # Chuẩn bị dữ liệu update
        update_data = {}
        if data.start_time is not None:
            update_data["start_time"] = data.start_time.isoformat()
        if data.end_time is not None:
            update_data["end_time"] = data.end_time.isoformat()
        if data.is_available is not None:
            update_data["is_available"] = data.is_available
        if data.slot_date is not None:
            update_data["slot_date"] = data.slot_date.isoformat()
        
        if not update_data:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
        
        response = supabase.table("time_slots")\
            .update(update_data)\
            .eq("id", time_slot_id)\
            .execute()
        
        return {
            "message": "Cập nhật time slot thành công",
            "time_slot": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


def toggle_time_slot_availability(time_slot_id: int):
    """Chuyển đổi trạng thái available/unavailable"""
    try:
        current_slot = supabase.table("time_slots")\
            .select("is_available")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not current_slot.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        new_status = not current_slot.data[0]['is_available']
        
        response = supabase.table("time_slots")\
            .update({"is_available": new_status})\
            .eq("id", time_slot_id)\
            .execute()
        
        return {
            "message": f"Đã chuyển trạng thái thành {'available' if new_status else 'unavailable'}",
            "time_slot": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cập nhật thất bại: {str(e)}")


# ==================== Delete Time Slot ====================

def delete_time_slot(time_slot_id: int):
    """Xóa time slot (chỉ xóa nếu chưa có booking)"""
    try:
        # Kiểm tra có booking nào sử dụng slot này không
        booking_check = supabase.table("bookings")\
            .select("id")\
            .eq("time_slot_id", time_slot_id)\
            .execute()
        
        if booking_check.data:
            raise HTTPException(
                status_code=400, 
                detail="Không thể xóa time slot đã có booking. Hãy set is_available = False thay vì xóa."
            )
        
        response = supabase.table("time_slots")\
            .delete()\
            .eq("id", time_slot_id)\
            .execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        return {"message": "Xóa time slot thành công"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Xóa thất bại: {str(e)}")