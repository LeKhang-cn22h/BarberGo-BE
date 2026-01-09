from fastapi_mail import FastMail, MessageSchema, MessageType
from config.email_config import conf
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
        """Gửi email text đơn giản"""
        try:
            message = MessageSchema(
                subject=subject,
                recipients=recipients,
                body=body,
                subtype=MessageType.plain
            )
            await self.fm.send_message(message)
            logger.info(f"Email sent successfully to {recipients}")
            return {"message": "Email sent successfully"}
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise
    
    async def send_html_email(
        self,
        recipients: List[str],
        subject: str,
        html_content: str
    ):
        """Gửi email HTML"""
        try:
            message = MessageSchema(
                subject=subject,
                recipients=recipients,
                body=html_content,
                subtype=MessageType.html
            )
            await self.fm.send_message(message)
            return {"message": "Email sent successfully"}
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise
    
    async def send_template_email(
        self,
        recipients: List[str],
        subject: str,
        template_name: str,
        template_data: dict
    ):
        """Gửi email với template"""
        try:
            message = MessageSchema(
                subject=subject,
                recipients=recipients,
                template_body=template_data,
                subtype=MessageType.html
            )
            await self.fm.send_message(message, template_name=template_name)
            return {"message": "Email sent successfully"}
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise

email_service = EmailService()