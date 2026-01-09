from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.email_service import email_service
from app.schemas.email_schema import EmailRequest, OwnerCredentialsEmailRequest

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

@router.post("/send-owner-credentials")
async def send_owner_credentials(
    credentials_data: OwnerCredentialsEmailRequest,
    background_tasks: BackgroundTasks
):
    """Gửi email thông tin đăng nhập Owner"""
    
    # Tạo nội dung email đơn giản
    email_body = f"""
Thông tin đăng nhập tài khoản Owner - BarberGo

Email: {credentials_data.email}
Mật khẩu: {credentials_data.password}

Vui lòng đổi mật khẩu sau khi đăng nhập lần đầu.
    """
    
    background_tasks.add_task(
        email_service.send_simple_email,
        [credentials_data.recipient],
        "Thông tin tài khoản Owner - BarberGo",
        email_body
    )
    return {"message": "Owner credentials email is being sent"}