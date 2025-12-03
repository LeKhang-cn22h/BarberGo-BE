from fastapi import APIRouter, Depends
from app.schemas.user_schema import (
    RegisterRequest,
    UserUpdate,
    UserLogin,
    ResendConfirmationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    CreateOwnerRequest
)
from app.services import user_service
from app.dependencies.current_user import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

# ==================== Auth Routes ====================

@router.post("/register")
def register(data: RegisterRequest):
    """
    Đăng ký tài khoản mới
    - Supabase tự động gửi email xác nhận
    - User cần xác nhận email trước khi đăng nhập
    """
    return user_service.register_user(data)
@router.post("/create-owner")
def create_owner(data: CreateOwnerRequest):
    """
    Tạo tài khoản owner (admin)
    - Email tự động confirmed
    - Có thể đăng nhập ngay
    - Role = owner
    """
    return user_service.create_owner_account(data)

@router.post("/resend-confirmation")
def resend_confirmation(data: ResendConfirmationRequest):
    """
    Gửi lại email xác nhận
    - Dùng khi user không nhận được email đăng ký
    """
    return user_service.resend_confirmation_email(data)


@router.post("/login")
def login(data: UserLogin):
    """
    Đăng nhập
    - Yêu cầu email đã được xác nhận
    """
    return user_service.login_user(data)


@router.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest):
    """
    Quên mật khẩu - gửi email reset
    """
    return user_service.forgot_password(data)


@router.post("/reset-password")
def reset_password(data: ResetPasswordRequest):
    """
    Đặt lại mật khẩu với token từ email
    """
    return user_service.reset_password(data)


# ==================== User CRUD ====================

@router.get("/")
def list_users():
    """Lấy danh sách tất cả users"""
    return user_service.get_all_users()


@router.get("/{user_id}")
def get_user(user_id: str):
    """Lấy thông tin user theo ID"""
    return user_service.get_user_by_id(user_id)


@router.put("/{user_id}", dependencies=[Depends(get_current_user)])
def update_user(user_id: str, data: UserUpdate):
    """Cập nhật thông tin user"""
    return user_service.update_user(user_id, data)


@router.delete("/{user_id}", dependencies=[Depends(get_current_user)])
def delete_user(user_id: str):
    """Xóa user"""
    return user_service.delete_user(user_id)
