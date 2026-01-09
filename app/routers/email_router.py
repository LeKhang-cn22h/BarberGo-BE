from fastapi import APIRouter, HTTPException, BackgroundTasks
from services.email_service import email_service
from schemas.email_schema import EmailRequest,BookingEmailRequest
router = APIRouter(prefix="/api/email", tags=["Email"])

@router.post("/send-simple")
async def send_simple_email(
    email_data: EmailRequest,
    background_tasks: BackgroundTasks
):
    """Gửi email đơn giản"""
    background_tasks.add_task(
        email_service.send_simple_email,
        email_data.recipients,
        email_data.subject,
        email_data.body
    )
    return {"message": "Email is being sent in background"}

@router.post("/send-booking-confirmation")
async def send_booking_confirmation(
    booking_data: BookingEmailRequest,
    background_tasks: BackgroundTasks
):
    """Gửi email xác nhận đặt lịch"""
    template_data = {
        "customer_name": booking_data.customer_name,
        "service_name": booking_data.service_name,
        "booking_date": booking_data.booking_date,
        "booking_time": booking_data.booking_time,
        "barber_name": booking_data.barber_name,
        "shop_address": booking_data.shop_address,
        "booking_code": booking_data.booking_code
    }
    
    background_tasks.add_task(
        email_service.send_template_email,
        [booking_data.recipient],
        "Xác nhận đặt lịch - BarberGo",
        "booking_confirmation.html",
        template_data
    )
    return {"message": "Booking confirmation email is being sent"}