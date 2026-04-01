from pydantic import BaseModel, EmailStr
from typing import Optional

# Dùng cho route đăng ký user
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None

# Dùng cho trả về thông tin user
class User(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    role:str = "user"

# Dùng cho route đăng nhập user
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Dùng cho route cập nhật thông tin user
class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None

# Yêu cầu gửi lại email xác nhận
class ResendConfirmationRequest(BaseModel):
    email: EmailStr

# Dùng cho forgot password
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# Dùng cho reset password
class ResetPasswordRequest(BaseModel):
    email: EmailStr
    token: str
    new_password: str

class CreateOwnerRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None

class GGLoginRequest(BaseModel):
    id_token:str

class RefreshTokenRequest(BaseModel):
    refresh_token: str