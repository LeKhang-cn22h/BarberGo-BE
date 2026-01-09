from pydantic import BaseModel, EmailStr
from typing import List

class EmailRequest(BaseModel):
    recipients: List[EmailStr]
    subject: str
    body: str

class BookingEmailRequest(BaseModel):
    recipient: EmailStr
    customer_name: str
    service_name: str
    booking_date: str
    booking_time: str
    barber_name: str
    shop_address: str
    booking_code: str