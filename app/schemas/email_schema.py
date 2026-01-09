from pydantic import BaseModel, EmailStr
from typing import List

class EmailRequest(BaseModel):
    recipients: List[EmailStr]
    subject: str
    body: str
