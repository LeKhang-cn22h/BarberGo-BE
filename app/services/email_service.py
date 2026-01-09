from fastapi_mail import FastMail, MessageSchema, MessageType
from app.config.email_config import conf
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.fm = FastMail(conf)

    async def send_simple_email(
        self,
        recipients: List[str],
        subject: str,
        body: str
    ):
        try:
            message = MessageSchema(
                subject=subject,
                recipients=recipients,
                body=body,
                subtype=MessageType.plain
            )
            await self.fm.send_message(message)
            return {"message": "Email sent successfully"}
        except Exception as e:
            logger.error(f"Send email failed: {e}")
            raise

email_service = EmailService()
