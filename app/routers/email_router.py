from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.email_service import EmailService
from app.schemas.email_schema import EmailRequest
router = APIRouter(prefix="", tags=["Email"])

@router.post("/send-simple")
async def send_simple_email(
    email_data: EmailRequest,
    background_tasks: BackgroundTasks
):
    """Gửi email đơn giản"""
    background_tasks.add_task(
        EmailService.send_simple_email,
        email_data.recipients,
        email_data.subject,
        email_data.body
    )
    return {"message": "Email is being sent in background"}