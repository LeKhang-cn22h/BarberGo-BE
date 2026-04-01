from app.database.supabase_client import supabase
from app.schemas.time_slot_schema import TimeSlotCreate, TimeSlotUpdate, TimeSlotBulkCreate
from fastapi import HTTPException
from datetime import datetime, time, timedelta

# ==================== Create Time Slot ====================

def create_time_slot(data: TimeSlotCreate):
    """Tạo time slot mới"""
    try:
        now = datetime.now()
        today = now.date()
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
        
        if data.slot_date < today:
            raise HTTPException(
                status_code=400,
                detail="Không thể tạo time slot cho ngày trong quá khứ"
            )
        #kiểm tra giờ start<end
        if data.start_time>=data.end_time:
            raise HTTPException(
                status_code=400,
                detail="Không thể tạo giờ bắt đầu và kết thúc bất hợp lí"
            )
        #kiểm tra giờ bị lố
        duration = datetime.combine(datetime.min, data.end_time) - datetime.combine(datetime.min, data.start_time)
        if duration.total_seconds() > 14400:  # 4 giờ = 14400 giây
            raise HTTPException(
                status_code=400,
                detail="Thời gian slot không được vượt quá 4 giờ"
            )
        
        #  Nếu là hôm nay, kiểm tra giờ không được trong quá khứ
        if data.slot_date == today:
            slot_start = datetime.combine(data.slot_date, data.start_time)
            min_allowed_time=now + timedelta(minutes=15)
            if slot_start < min_allowed_time:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Time slot phải bắt đầu sau "
                        f"{min_allowed_time.strftime('%H:%M')}"
                     )
                )
    
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
    """Tạo nhiều time slots cùng lúc cho 1 ngày (đã validate đầy đủ)"""
    try:
        now = datetime.now()
        today = now.date()

        # ==================== Check barber ====================
        barber_check = supabase.table("barbers")\
            .select("id, status, opening_time, closing_time, working_days")\
            .eq("id", data.barber_id)\
            .execute()

        if not barber_check.data:
            raise HTTPException(status_code=404, detail="Barber không tồn tại")

        barber = barber_check.data[0]

        if not barber.get("status"):
            raise HTTPException(status_code=400, detail="Barber hiện không hoạt động")

        # ==================== Check working day ====================
        day_name = data.slot_date.strftime("%A")
        working_days = barber.get("working_days", [])

        if day_name not in working_days:
            raise HTTPException(
                status_code=400,
                detail=f"Barber không làm việc vào {day_name}"
            )

        # ==================== Check past date ====================
        if data.slot_date < today:
            raise HTTPException(
                status_code=400,
                detail="Không thể tạo time slot cho ngày trong quá khứ"
            )

        opening_time = barber.get("opening_time")
        closing_time = barber.get("closing_time")

        prepared_slots = []
        used_ranges = []

        # ==================== Validate từng time range ====================
        for idx, time_range in enumerate(data.time_ranges):
            try:
                start_time = time.fromisoformat(time_range["start_time"])
                end_time = time.fromisoformat(time_range["end_time"])
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Định dạng thời gian không hợp lệ ở slot thứ {idx + 1}"
                )

            # start < end
            if start_time >= end_time:
                raise HTTPException(
                    status_code=400,
                    detail=f"start_time phải nhỏ hơn end_time (slot thứ {idx + 1})"
                )

            # within opening hours
            if opening_time and closing_time:
                if start_time < opening_time or end_time > closing_time:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Slot {start_time}-{end_time} nằm ngoài "
                            f"giờ làm việc {opening_time}-{closing_time}"
                        )
                    )

            # past time (today)
            if data.slot_date == today:
                slot_start = datetime.combine(data.slot_date, start_time)
                if slot_start <= now:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Slot {start_time}-{end_time} "
                            f"đã nằm trong quá khứ"
                        )
                    )

            # ==================== Check overlap trong batch ====================
            for used_start, used_end in used_ranges:
                if not (end_time <= used_start or start_time >= used_end):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Slot {start_time}-{end_time} "
                            f"bị trùng với slot {used_start}-{used_end} trong batch"
                        )
                    )

            used_ranges.append((start_time, end_time))

            prepared_slots.append({
                "barber_id": data.barber_id,
                "slot_date": data.slot_date.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "is_available": True
            })

        # ==================== Check overlap với DB ====================
        for slot in prepared_slots:
            overlap_check = supabase.table("time_slots")\
                .select("id")\
                .eq("barber_id", data.barber_id)\
                .eq("slot_date", data.slot_date.isoformat())\
                .or_(
                    f"and(start_time.lte.{slot['start_time']},end_time.gt.{slot['start_time']}),"
                    f"and(start_time.lt.{slot['end_time']},end_time.gte.{slot['end_time']}),"
                    f"and(start_time.gte.{slot['start_time']},end_time.lte.{slot['end_time']})"
                )\
                .execute()

            if overlap_check.data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Slot {slot['start_time']}-{slot['end_time']} bị trùng với slot đã tồn tại"
                )

        # ==================== Insert ====================
        response = supabase.table("time_slots")\
            .insert(prepared_slots)\
            .execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo time slots thất bại")

        return {
            "message": f"Tạo thành công {len(response.data)} time slots",
            "time_slots": response.data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Tạo time slots thất bại: {str(e)}"
        )

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
        now = datetime.now()
        today = now.date()
        
        # Lấy time slot hiện tại
        current_slot = supabase.table("time_slots")\
            .select("*")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not current_slot.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        slot = current_slot.data[0]
        
        #  Nếu update slot_date hoặc start_time, check past time
        new_date = data.slot_date if data.slot_date else datetime.fromisoformat(slot['slot_date']).date()
        new_start = data.start_time if data.start_time else time.fromisoformat(slot['start_time'])
        
        if new_date < today:
            raise HTTPException(
                status_code=400,
                detail="Không thể cập nhật time slot cho ngày trong quá khứ"
            )
        
        if new_date == today:
            slot_start = datetime.combine(new_date, new_start)
            if slot_start <= now:
                raise HTTPException(
                    status_code=400,
                    detail=f"Không thể cập nhật time slot trong quá khứ"
                )
                    
        # Tính toán thời gian mới
        final_date = data.slot_date if data.slot_date else datetime.fromisoformat(slot['slot_date']).date()
        final_start = data.start_time if data.start_time else time.fromisoformat(slot['start_time'])
        final_end = data.end_time if data.end_time else time.fromisoformat(slot['end_time'])
        
        # Check duration
        if final_start >= final_end:
            raise HTTPException(
                status_code=400,
                detail="Giờ kết thúc phải sau giờ bắt đầu"
            )
        
        duration = datetime.combine(datetime.min, final_end) - datetime.combine(datetime.min, final_start)
        if duration.total_seconds() < 900:
            raise HTTPException(
                status_code=400,
                detail="Thời gian slot phải ít nhất 15 phút"
            )
        
        # Check overlap với slots khác
        overlap_check = supabase.table("time_slots")\
            .select("id")\
            .eq("barber_id", slot['barber_id'])\
            .eq("slot_date", final_date.isoformat())\
            .neq("id", time_slot_id)\
            .or_(
                f"and(start_time.lte.{final_start},end_time.gt.{final_start}),"
                f"and(start_time.lt.{final_end},end_time.gte.{final_end}),"
                f"and(start_time.gte.{final_start},end_time.lte.{final_end})"
            )\
            .execute()
        
        if overlap_check.data:
            raise HTTPException(
                status_code=400,
                detail="Thời gian mới bị trùng với slot khác"
            )
        
        # Kiểm tra xem có booking nào đang sử dụng slot này không
        if data.is_available == False or data.start_time or data.end_time or data.slot_date:
            booking_check = supabase.table("bookings")\
                .select("id")\
                .eq("time_slot_id", time_slot_id)\
                .in_("status", ["confirmed", "pending"])\
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
            .select("id, is_available, slot_date, start_time, end_time, barber_id")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not current_slot.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        slot_data = current_slot.data[0]
        
        # Dùng để lấy trạng thái ngược lại
        new_status = not slot_data['is_available']

        # Nếu ĐÓNG slot → Check có booking không
        if not new_status:
            booking_check = supabase.table("bookings")\
                .select("id")\
                .eq("time_slot_id", time_slot_id)\
                .in_("status", ["confirmed"])\
                .execute()
            
            if booking_check.data:
                raise HTTPException(
                    status_code=400, 
                    detail="Không thể đóng slot đang có booking"
                )
        
        # Nếu MỞ slot → Check không cho mở slot trong quá khứ
        if new_status:
            slot_datetime = datetime.combine(
                datetime.fromisoformat(slot_data['slot_date']).date(),
                time.fromisoformat(slot_data['start_time'])
            )
            
            if slot_datetime <= datetime.now():
                raise HTTPException(
                    status_code=400,
                    detail="Không thể mở lại slot trong quá khứ"
                )
        
        # Update trạng thái
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
    try:
        # Lấy thông tin slot để check quá khứ
        slot_data = supabase.table("time_slots")\
            .select("slot_date, start_time")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not slot_data.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        # Không cho xóa slot trong quá khứ
        slot = slot_data.data[0]
        slot_datetime = datetime.combine(
            datetime.fromisoformat(slot['slot_date']).date(),
            time.fromisoformat(slot['start_time'])
        )
        
        if slot_datetime <= datetime.now():
            raise HTTPException(
                status_code=400,
                detail="Không thể xóa slot trong quá khứ"
            )
        
        # Kiểm tra có booking không
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