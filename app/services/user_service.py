from app.database.supabase_client import supabase
from app.schemas.user_schema import (
    RegisterRequest,
    UserUpdate,
    UserLogin,
    ResendConfirmationRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    
)
from fastapi import HTTPException
from datetime import datetime

# ==================== Register & Email Confirmation ====================

def register_user(data: RegisterRequest):
    """Đăng ký user - Supabase tự động gửi email xác nhận"""
    try:
        # Kiểm tra email đã tồn tại chưa
        existing_user = supabase.table("users").select("email").eq("email", data.email).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Email đã được đăng ký")
        
        # Đăng ký user - Supabase tự động gửi email verification
        auth_response = supabase.auth.sign_up({
            "email": data.email,
            "password": data.password,
            "options": {
                "email_redirect_to": "myapp://email-confirmed"  # Deep link cho mobile
            }
        })

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Đăng ký thất bại")

        user_id = auth_response.user.id

        # Lưu thông tin vào table users
        user_data = {
            "id": user_id,
            "email": data.email,
            "full_name": data.full_name,
            "phone": data.phone,
            "avatar_url": None,
            "created_at": datetime.now().isoformat()
        }
        supabase.table("users").insert(user_data).execute()

        return {
            "message": "Đăng ký thành công! Vui lòng kiểm tra email để xác nhận tài khoản.",
            "user": {
                "id": user_id,
                "email": data.email,
                "full_name": data.full_name,
                "phone": data.phone,
                "email_confirmed": auth_response.user.email_confirmed_at is not None
            },
            "note": "Tài khoản chưa được kích hoạt. Hãy xác nhận email để đăng nhập."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Đăng ký thất bại: {str(e)}")


def resend_confirmation_email(data: ResendConfirmationRequest):
    """Gửi lại email xác nhận"""
    try:
        response = supabase.auth.resend({
            "type": "signup",
            "email": data.email,
            "options": {
                "email_redirect_to": "myapp://email-confirmed"
            }
        })
        return {
            "message": "Email xác nhận đã được gửi lại",
            "email": data.email
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể gửi email: {str(e)}")


# ==================== Login ====================

def login_user(data: UserLogin):
    """Đăng nhập user"""
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": data.email,
            "password": data.password
        })
        
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Email hoặc mật khẩu không đúng")
        
        # Kiểm tra email đã xác nhận chưa
        if not auth_response.user.email_confirmed_at:
            raise HTTPException(
                status_code=403, 
                detail="Email chưa được xác nhận. Vui lòng kiểm tra email và xác nhận tài khoản."
            )
        
        # Lấy thông tin user từ table users
        user_response = supabase.table("users").select("*").eq("id", auth_response.user.id).execute()
        user_data = user_response.data[0] if user_response.data else {}
        
        return {
            "message": "Đăng nhập thành công",
            "user": {
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "full_name": user_data.get("full_name"),
                "phone": user_data.get("phone"),
                "avatar_url": user_data.get("avatar_url"),
                "email_confirmed": True
            },
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Email hoặc mật khẩu không đúng")


# ==================== CRUD Operations ====================

def get_all_users():
    """Lấy danh sách tất cả users"""
    response = supabase.table("users").select("*").execute()
    return response.data


def get_user_by_id(user_id: str):
    """Lấy thông tin user theo ID"""
    response = supabase.table("users").select("*").eq("id", user_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    return response.data[0]


def update_user(user_id: str, data: UserUpdate):
    """Cập nhật thông tin user"""
    # Chỉ update các field không None
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")
    
    response = supabase.table("users").update(update_data).eq("id", user_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    
    return {
        "message": "Cập nhật thành công",
        "user": response.data[0]
    }


def delete_user(user_id: str):
    """Xóa user"""
    # Xóa từ table users
    response = supabase.table("users").update({"status":False}).eq("id", user_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Không tìm thấy user")
    
    return {"message": "Xóa user thành công"}


# ==================== Password Reset ====================

def forgot_password(data: ForgotPasswordRequest):
    """Gửi email reset password"""
    try:
        response = supabase.auth.reset_password_for_email(
            data.email,
            options={
                "redirect_to": "myapp://reset-password"  # Deep link cho mobile
            }
        )
        return {
            "message": "Email đặt lại mật khẩu đã được gửi",
            "email": data.email
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể gửi email: {str(e)}")


def reset_password(data: ResetPasswordRequest):
    """Đặt lại mật khẩu với token từ email"""
    try:
        # Verify token và update password
        response = supabase.auth.verify_otp({
            "email": data.email,
            "token": data.token,
            "type": "recovery"
        })
        
        if response.user:
            # Update password
            supabase.auth.update_user({
                "password": data.new_password
            })
            return {"message": "Đặt lại mật khẩu thành công"}
        else:
            raise HTTPException(status_code=400, detail="Token không hợp lệ")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Đặt lại mật khẩu thất bại: {str(e)}")

# ==================== Admin - Create Owner ====================

def create_owner_account(data):
    """Tạo tài khoản owner - Email tự động confirmed"""
    try:
        # Kiểm tra email đã tồn tại chưa
        existing_user = supabase.table("users").select("email").eq("email", data.email).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Email đã được đăng ký")
        
        # Tạo user với admin API - TỰ ĐỘNG CONFIRM EMAIL
        auth_response = supabase.auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True,  # ← TỰ ĐỘNG CONFIRM
            "user_metadata": {
                "full_name": data.full_name,
                "role": "owner"
            }
        })

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Tạo tài khoản thất bại")

        user_id = auth_response.user.id

        # Lưu thông tin vào table users với role = owner
        user_data = {
            "id": user_id,
            "email": data.email,
            "full_name": data.full_name,
            "phone": data.phone,
            "role": "owner",
            "avatar_url": None,
            "created_at": datetime.now().isoformat()
        }
        supabase.table("users").insert(user_data).execute()

        return {
            "message": "Tạo tài khoản owner thành công. Có thể đăng nhập ngay không cần confirm email.",
            "user": {
                "id": user_id,
                "email": data.email,
                "full_name": data.full_name,
                "phone": data.phone,
                "role": "owner",
                "email_confirmed": True
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tạo tài khoản owner thất bại: {str(e)}")