from fastapi import APIRouter, Depends,Request
from app.schemas.user_schema import (
    RegisterRequest,
    UserUpdate,
    UserLogin,
    ResendConfirmationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    CreateOwnerRequest,
    GGLoginRequest,
    RefreshTokenRequest
)
from app.api.dependencies import (
    require_admin, 
    verify_self_or_admin, 
    prevent_self_action,
    require_system,
)
from app.services import user_service
from app.services import auth_service
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/users", tags=["Users"])

# ==================== Auth Routes ====================

@router.post("/register")
@limiter.limit("10/minute")
def register(request:Request, data: RegisterRequest):
    """
    Đăng ký tài khoản mới
    - Supabase tự động gửi email xác nhận
    - User cần xác nhận email trước khi đăng nhập
    """
    return user_service.register_user(data)
@router.post("/create-owner")
@limiter.limit("10/minute")
def create_owner(request:Request, data: CreateOwnerRequest):
    """
    Tạo tài khoản owner (admin)
    - Email tự động confirmed
    - Có thể đăng nhập ngay
    - Role = owner
    """
    return user_service.create_owner_account(data)

 # gui id kiem tra can dang ky hay dang nhap
@router.post("/google")
@limiter.limit("10/minute")
def login_or_logup_gg(request:Request,data: GGLoginRequest):
    return user_service.gg_login_or_logup(data.id_token)

@router.post("/resend-confirmation")
@limiter.limit("10/minute")
def resend_confirmation(request:Request,data: ResendConfirmationRequest):
    """
    Gửi lại email xác nhận
    - Dùng khi user không nhận được email đăng ký
    """
    return user_service.resend_confirmation_email(data)


@router.post("/login")
@limiter.limit("10/minute")
def login(request:Request,data: UserLogin):
    """
    Đăng nhập
    - Yêu cầu email đã được xác nhận
    """
    return user_service.login_user(data)


@router.post("/forgot-password")
@limiter.limit("10/minute")
def forgot_password(request:Request, data: ForgotPasswordRequest):
    """
    Quên mật khẩu - gửi email reset
    """
    return user_service.forgot_password(data)


@router.post("/reset-password")
@limiter.limit("10/minute")
def reset_password(request:Request,data: ResetPasswordRequest):
    """
    Đặt lại mật khẩu với token từ email
    """
    return user_service.reset_password(data)

@router.post("/auth/refresh")
def refresh_token(data:RefreshTokenRequest):
    return auth_service.refresh_access_token(data.refresh_token)
# ==================== User CRUD ====================

@router.get("/")
@limiter.limit("120/minute")
def list_users(request:Request, current_user=Depends(require_admin)):
    """Lấy danh sách tất cả users"""
    return user_service.get_all_users()


@router.get("/{user_id}")
@limiter.limit("120/minute")

def get_user(request:Request,user_id: str, current_user=Depends(verify_self_or_admin)):
    """Lấy thông tin user theo ID"""
    return user_service.get_user_by_id(user_id)


@router.put("/{user_id}")
@limiter.limit("30/minute")
def update_user(request:Request,user_id: str, data: UserUpdate, current_user=Depends(verify_self_or_admin)):
    """Cập nhật thông tin user"""
    return user_service.update_user(user_id, data)


@router.delete("/{user_id}",)
@limiter.limit("10/minute")

def delete_user(request:Request,user_id: str, current_user=Depends(require_system)):
    # ngăn admin xóa chỉnh mình
    prevent_self_action(user_id,current_user,action="xóa")
    """Xóa user"""
    return user_service.delete_user(user_id)

