from datetime import datetime, time, timedelta
from app.database.supabase_client import supabase
from app.schemas.booking_schema import BookingCreate
from fastapi import HTTPException

# ==================== Create Booking ====================

def create_booking(data: BookingCreate):
    """Tạo booking mới với nhiều dịch vụ"""
    try:
        now = datetime.now()
        today = now.date()
        # Kiểm tra user có tồn tại không
        user_check = supabase.table("users").select("id").eq("id", data.user_id).execute()
        if not user_check.data:
            raise HTTPException(status_code=404, detail="User không tồn tại")
        
        # Kiểm tra time_slot có tồn tại và available không
        time_slot_check = supabase.table("time_slots")\
            .select("id, is_available, barber_id, start_time, end_time, slot_date")\
            .eq("id", data.time_slot_id)\
            .execute()
        
        if not time_slot_check.data:
            raise HTTPException(status_code=404, detail="Time slot không tồn tại")
        
        time_slot = time_slot_check.data[0]
        if not time_slot.get('is_available'):
            raise HTTPException(status_code=400, detail="Time slot đã được đặt")
        
        slot_datetime = datetime.combine(
            datetime.fromisoformat(time_slot['slot_date']).date(),
            time.fromisoformat(time_slot['start_time'])
        )
        
        # Không cho đặt lịch trong quá khứ
        if slot_datetime < now:
            raise HTTPException(
                status_code=400,
                detail=f"Không thể đặt lịch trong quá khứ. Time slot bắt đầu lúc {slot_datetime.strftime('%H:%M %d/%m/%Y')}"
            )
        
        # Yêu cầu đặt trước ít nhất 30p
        min_advance_booking = timedelta(minutes=30)
        if slot_datetime < now + min_advance_booking:
            raise HTTPException(
                status_code=400,
                detail=f"Vui lòng đặt lịch trước ít nhất 30 phút. Time slot bắt đầu lúc {slot_datetime.strftime('%H:%M %d/%m/%Y')}"
            )        
        
        barber_id = time_slot.get('barber_id')
        
        # Kiểm tra tất cả services có tồn tại và thuộc về barber không
        for service_id in data.service_ids:
            service_check = supabase.table("services")\
                .select("id, barber_id")\
                .eq("id", service_id)\
                .execute()
            
            if not service_check.data:
                raise HTTPException(status_code=404, detail=f"Dịch vụ {service_id} không tồn tại")
            
            if service_check.data[0].get('barber_id') != barber_id:
                raise HTTPException(status_code=400, detail=f"Dịch vụ {service_id} không thuộc barber này")
        # kiểm tra user đã booking chưa
        duplicate_booking = supabase.table("bookings")\
            .select("id")\
            .eq("user_id", data.user_id)\
            .eq("time_slot_id", data.time_slot_id)\
            .in_("status", ["confirmed", "pending"])\
            .execute()
        
        if duplicate_booking.data:
            raise HTTPException(
                status_code=400,
                detail="Bạn đã đặt lịch cho time slot này rồi"
            )
        # Tạo booking
        booking_data = {
            "user_id": data.user_id,
            "time_slot_id": data.time_slot_id,
            "total_duration_min": data.total_duration_min,
            "status": data.status,
            "total_price": data.total_price
        }
        
        response = supabase.table("bookings").insert(booking_data).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Tạo booking thất bại")
        
        booking = response.data[0]
        booking_id = booking['id']
        
        # Thêm các services vào booking_services
        booking_services = [
            {"booking_id": booking_id, "service_id": service_id}
            for service_id in data.service_ids
        ]
        
        supabase.table("booking_services").insert(booking_services).execute()
        
        # Cập nhật time_slot thành unavailable
        supabase.table("time_slots")\
            .update({"is_available": False})\
            .eq("id", data.time_slot_id)\
            .execute()
        
        return {
            "message": "Đặt lịch thành công",
            "booking": booking
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo booking thất bại: {str(e)}")


# ==================== Get Bookings ====================

def get_all_bookings():
    """Lấy danh sách tất cả bookings với thông tin đầy đủ"""
    try:
        # Lấy bookings
        response = supabase.table("bookings")\
            .select("*, users(full_name, email, phone), time_slots(*, barbers(id, name, address))")\
            .execute()
        
        bookings = response.data
        
        # Lấy services cho từng booking
        for booking in bookings:
            booking_services = supabase.table("booking_services")\
                .select("services(id, service_name, price, duration_min)")\
                .eq("booking_id", booking['id'])\
                .execute()
            
            booking['services'] = [bs['services'] for bs in booking_services.data]
        
        return bookings
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")
    
def get_booking_by_id(booking_id: int):
    """Lấy thông tin booking theo ID với đầy đủ thông tin"""
    try:
        print(f" DEBUG: Fetching booking ID: {booking_id}")
        
        # Test query đơn giản trước
        simple_response = supabase.table("bookings")\
            .select("*")\
            .eq("id", booking_id)\
            .execute()
        
        
        if not simple_response.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        booking = simple_response.data[0]
        
        # Lấy user info
        try:
            if booking.get('user_id'):
                user_response = supabase.table("users")\
                    .select("full_name, email, phone")\
                    .eq("id", booking['user_id'])\
                    .execute()
                booking['users'] = user_response.data[0] if user_response.data else None
        except Exception as user_error:
            booking['users'] = None
        
        # Lấy time_slot info
        try:
            if booking.get('time_slot_id'):
                slot_response = supabase.table("time_slots")\
                    .select("*")\
                    .eq("id", booking['time_slot_id'])\
                    .execute()
                
                if slot_response.data:
                    time_slot = slot_response.data[0]
                    
                    # Lấy barber info
                    if time_slot.get('barber_id'):
                        barber_response = supabase.table("barbers")\
                            .select("id, name, address, phone")\
                            .eq("id", time_slot['barber_id'])\
                            .execute()
                        time_slot['barbers'] = barber_response.data[0] if barber_response.data else None
                    
                    booking['time_slots'] = time_slot
                else:
                    booking['time_slots'] = None
        except Exception as slot_error:
            booking['time_slots'] = None
        
        # Lấy services
        try:
            booking_services = supabase.table("booking_services")\
                .select("service_id")\
                .eq("booking_id", booking_id)\
                .execute()
            
            
            services = []
            for bs in booking_services.data:
                service = supabase.table("services")\
                    .select("id, service_name, price, duration_min")\
                    .eq("id", bs['service_id'])\
                    .execute()
                if service.data:
                    services.append(service.data[0])
            
            booking['services'] = services
        except Exception as service_error:
            booking['services'] = []
        
        return booking
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}")
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")
    
def get_bookings_by_user(user_id: str):
    """Lấy danh sách bookings của 1 user"""
    try:
        response = supabase.table("bookings")\
            .select("*, time_slots(*, barbers(id, name, address))")\
            .eq("user_id", user_id)\
            .order("id", desc=True)\
            .execute()
        
        bookings = response.data
        
        # Lấy services cho từng booking
        for booking in bookings:
            booking_services = supabase.table("booking_services")\
                .select("services(id, service_name, price, duration_min)")\
                .eq("booking_id", booking['id'])\
                .execute()
            
            booking['services'] = [bs['services'] for bs in booking_services.data]
        
        return bookings
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_bookings_by_barber(barber_id: str):
    """Lấy danh sách bookings của 1 barber"""
    try:
        # Lấy bookings thông qua time_slots
        response = supabase.table("bookings")\
            .select("*, users(full_name, email, phone), time_slots!inner(*, barbers(id, name))")\
            .eq("time_slots.barber_id", barber_id)\
            .order("id", desc=True)\
            .execute()
        
        bookings = response.data
        
        # Lấy services cho từng booking
        for booking in bookings:
            booking_services = supabase.table("booking_services")\
                .select("services(id, service_name, price, duration_min)")\
                .eq("booking_id", booking['id'])\
                .execute()
            
            booking['services'] = [bs['services'] for bs in booking_services.data]
        
        return bookings
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


def get_bookings_by_status(status: str):
    """Lấy danh sách bookings theo status"""
    try:
        allowed_statuses = ['confirmed', 'completed', 'cancelled']
        if status not in allowed_statuses:
            raise HTTPException(status_code=400, detail=f"Status không hợp lệ. Phải là: {allowed_statuses}")
        
        response = supabase.table("bookings")\
            .select("*, users(full_name, email), time_slots(*, barbers(id, name))")\
            .eq("status", status)\
            .order("id", desc=True)\
            .execute()
        
        bookings = response.data
        
        # Lấy services cho từng booking
        for booking in bookings:
            booking_services = supabase.table("booking_services")\
                .select("services(id, service_name, price, duration_min)")\
                .eq("booking_id", booking['id'])\
                .execute()
            
            booking['services'] = [bs['services'] for bs in booking_services.data]
        
        return bookings
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


# ==================== Update Booking ====================
def update_booking_status(booking_id: int, status: str):
    """Cập nhật status của booking"""
    try:
        allowed_statuses = ['confirmed', 'completed', 'cancelled', 'pending']
        if status not in allowed_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Status không hợp lệ. Phải là: {allowed_statuses}"
            )
        
        # Lấy booking hiện tại
        current_booking = supabase.table("bookings")\
            .select("id, status, time_slot_id")\
            .eq("id", booking_id)\
            .execute()
        
        if not current_booking.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        old_status = current_booking.data[0]['status']
        time_slot_id = current_booking.data[0]['time_slot_id']
        
        #  Lấy thông tin time slot
        current_slot = supabase.table("time_slots")\
            .select("slot_date, start_time, end_time")\
            .eq("id", time_slot_id)\
            .execute()
        
        if not current_slot.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy time slot")
        
        slot = current_slot.data[0]
        
        #  Kiểm tra logic chuyển trạng thái
        
        # 1. Không cho completed nếu chưa đến giờ (trừ 15 phút)
        if status == 'completed':
            slot_start = datetime.combine(
                datetime.fromisoformat(slot['slot_date']).date(),
                time.fromisoformat(slot['start_time'])
            )
            slot_end = datetime.combine(
                datetime.fromisoformat(slot['slot_date']).date(),
                time.fromisoformat(slot['end_time'])
            )
            now = datetime.now()
            
            # Chỉ cho completed từ 15 phút trước start_time đến end_time
            if now < slot_start - timedelta(minutes=15):
                raise HTTPException(
                    status_code=400,
                    detail=f"Chỉ có thể hoàn thành booking từ 15 phút trước giờ bắt đầu ({slot_start.strftime('%H:%M %d/%m')})"
                )
        
        # 2. Không cho cancel nếu đã completed
        if status == 'cancelled' and old_status == 'completed':
            raise HTTPException(
                status_code=400,
                detail="Không thể hủy booking đã hoàn thành"
            )
        
        # 3. Không cho cancel nếu quá gần giờ 
        if status == 'cancelled':
            slot_start = datetime.combine(
                datetime.fromisoformat(slot['slot_date']).date(),
                time.fromisoformat(slot['start_time'])
            )
            
            if datetime.now() > slot_start - timedelta(hours=1):
                raise HTTPException(
                    status_code=400,
                    detail="Không thể hủy booking trong vòng 1 giờ trước giờ hẹn"
                )
        
        # 4. Không cho thay đổi trạng thái trong quá khứ
        slot_end = datetime.combine(
            datetime.fromisoformat(slot['slot_date']).date(),
            time.fromisoformat(slot['end_time'])
        )
        
        if datetime.now() > slot_end + timedelta(hours=24):
            raise HTTPException(
                status_code=400,
                detail="Không thể thay đổi trạng thái booking đã quá 24 giờ"
            )
        
        #  Cập nhật status
        response = supabase.table("bookings")\
            .update({"status": status})\
            .eq("id", booking_id)\
            .execute()
        
        #  Nếu hủy booking, cập nhật lại time_slot thành available
        if status == 'cancelled' and old_status != 'cancelled':
            supabase.table("time_slots")\
                .update({"is_available": True})\
                .eq("id", time_slot_id)\
                .execute()
        
        return {
            "message": f"Cập nhật trạng thái thành '{status}' thành công",
            "booking": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cập nhật thất bại: {str(e)}"
        )

def cancel_booking(booking_id: int, reason: str = None):
    """Hủy booking với lý do"""
    try:
        #  Check booking status
        booking = supabase.table("bookings")\
            .select("status, time_slot_id, time_slots(slot_date, start_time)")\
            .eq("id", booking_id)\
            .execute()
        
        if not booking.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        current_status = booking.data[0]['status']
        
        #  Không cho cancel nếu đã completed
        if current_status == 'completed':
            raise HTTPException(
                status_code=400,
                detail="Không thể hủy booking đã hoàn thành"
            )
        
        #  Không cho cancel nếu đã cancelled
        if current_status == 'cancelled':
            raise HTTPException(
                status_code=400,
                detail="Booking đã bị hủy trước đó"
            )
        
        #  Check thời gian cancel
        time_slot = booking.data[0]['time_slots']
        slot_start = datetime.combine(
            datetime.fromisoformat(time_slot['slot_date']).date(),
            time.fromisoformat(time_slot['start_time'])
        )
        
        # Phải cancel trước ít nhất 1 giờ
        if datetime.now() > slot_start - timedelta(hours=1):
            raise HTTPException(
                status_code=400,
                detail="Phải hủy lịch trước ít nhất 1 giờ"
            )
        
        #  Update với reason
        update_data = {
            "status": "cancelled",
            "cancellation_reason": reason,
            "cancelled_at": datetime.now().isoformat()
        }
        
        response = supabase.table("bookings")\
            .update(update_data)\
            .eq("id", booking_id)\
            .execute()
        
        #  Mở lại time slot
        supabase.table("time_slots")\
            .update({"is_available": True})\
            .eq("id", booking.data[0]['time_slot_id'])\
            .execute()
        
        return {
            "message": "Hủy booking thành công",
            "booking": response.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hủy booking thất bại: {str(e)}")
    
def boom_booking(booking_id: int):
    """
    Owner đánh dấu khách không đến
    - Tự động ghi lý do: no_show
    - Tự động ghi thời gian
    - Chỉ được gọi sau kết thúc
    """
    try:        
        # Lấy booking
        booking = supabase.table("bookings")\
            .select("*, time_slots(slot_date, end_time, barber_id)")\
            .eq("id", booking_id)\
            .execute()
        
        if not booking.data:
            raise HTTPException(status_code=404, detail="Không tìm thấy booking")
        
        booking_data = booking.data[0]
        time_slot = booking_data['time_slots']
        
        #  Kiểm tra status
        if booking_data['status'] not in ['confirmed']:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ có thể đánh dấu không đến cho booking đang confirmed"
            )
        
        #  Kiểm tra thời gian
        slot_end = datetime.combine(
            datetime.fromisoformat(time_slot['slot_date']).date(),
            time.fromisoformat(time_slot['end_time'])
        )
        
        now = datetime.now()
        
        if now < slot_end:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ có thể đánh dấu không đến sau  ({slot_end.strftime('%H:%M %d/%m')} giờ hẹn)"
            )
        
        # Update booking - TỰ ĐỘNG GHI LÝ DO VÀ THỜI GIAN
        cancelled_at = datetime.now()
        
        response = supabase.table("bookings")\
            .update({
                "status": "cancelled",
                "cancellation_reason": "Khách hàng không đến",
                "cancelled_at": cancelled_at.isoformat()
            })\
            .eq("id", booking_id)\
            .execute()
                
        return {
            "message": "Đã đánh dấu khách không đến",
            "booking": response.data[0]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Thao tác thất bại: {str(e)}")