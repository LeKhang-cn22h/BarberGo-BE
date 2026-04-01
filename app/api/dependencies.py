from fastapi import Depends, HTTPException, status
from app.dependencies.current_user import get_current_user
from app.database.supabase_client import supabase

#  Role-Based Access Control 
async def require_owner(user = Depends(get_current_user)):
    """
    Yêu cầu user phải có role = owner
    Sử dụng cho các chức năng chỉ chủ sở hữu mới được phép
    """
    # Lấy thông tin user từ database
    db_user = supabase.table("users").select("*").eq("id", user.id).execute()
    
    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông tin user"
        )
    
    user_data = db_user.data[0]
    
    # Kiểm tra role
    if user_data.get("role") != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_permissions",
                "message": "Bạn không có quyền truy cập. Chỉ owner mới được phép."
            }
        )
    
    return user_data


async def require_admin(user = Depends(get_current_user)):
    """
    Yêu cầu user phải có role = admin hoặc owner
    Sử dụng cho các chức năng quản trị
    """
    # Lấy thông tin user từ database
    db_user = supabase.table("users").select("*").eq("id", user.id).execute()
    
    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông tin user"
        )
    
    user_data = db_user.data[0]
    
    # Kiểm tra role
    allowed_roles = ["admin", "owner"]
    if user_data.get("role") not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_permissions",
                "message": "Bạn không có quyền truy cập. Chỉ admin/owner mới được phép."
            }
        )
    
    return user_data

async def require_owner_of_barber(
    barber_id: str,
    current_user = Depends(get_current_user)
):
    # Admin được xem tất cả
    if current_user.role == "admin":
        return current_user

    barber = supabase.table("barbers") \
        .select("id, user_id") \
        .eq("id", barber_id) \
        .single() \
        .execute()

    if not barber.data:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy barber"
        )

    if barber.data["user_id"] != current_user.id:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "insufficient_permissions",
                "message": "Bạn không có quyền xem booking của barber này"
            }
        )

    return current_user

async def require_system(user = Depends(get_current_user)):
    """
    Yêu cầu user phải có role = admin
    Sử dụng cho các chức năng chỉ admin
    """
    # Lấy thông tin user từ database
    db_user = supabase.table("users").select("*").eq("id", user.id).execute()
    
    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông tin user"
        )
    
    user_data = db_user.data[0]
    
    # Kiểm tra role
    allowed_roles = [ "admin"]
    if user_data.get("role") not in allowed_roles:
      raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "insufficient_permissions",
                "message": "Bạn không có quyền truy cập."
            }
        )
    
    return user_data


#  Get User with Database Info 
async def get_current_user_with_db(user = Depends(get_current_user)):
    """
    Lấy thông tin user từ Auth và kết hợp với database
    Trả về full user data bao gồm role, status, etc.
    """
    # Lấy thông tin user từ database
    db_user = supabase.table("users").select("*").eq("id", user.id).execute()
    
    if not db_user.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy thông tin user trong database"
        )
    
    user_data = db_user.data[0]
    
    # Kiểm tra tài khoản có bị vô hiệu hóa không
    if user_data.get("status") == False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "account_disabled",
                "message": "Tài khoản của bạn đã bị vô hiệu hóa"
            }
        )
    
    return user_data


#  Resource Ownership Verification 
def verify_resource_owner(resource_user_id: str, current_user: dict):
    """
    Kiểm tra user có phải là owner của resource không
    - Owner/Admin có thể bỏ qua kiểm tra
    - User thường chỉ được truy cập resource của mình
    
    Usage:
        current_user = await get_current_user_with_db(user)
        verify_resource_owner(post.user_id, current_user)
    """
    # Owner và admin có thể truy cập mọi resource
    if current_user.get("role") in ["owner", "admin"]:
        return True
    
    # User thường chỉ được truy cập resource của mình
    if current_user["id"] != resource_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "access_denied",
                "message": "Bạn không có quyền truy cập resource này"
            }
        )
    
    return True


#  Custom Permission Check 
def require_permission(allowed_roles: list):
    """
    Factory function để tạo custom permission dependency
    
    Usage:
        @router.get("/special")
        async def special_route(
            user = Depends(require_permission(["admin", "owner"]))
        ):
            return {"message": "Success"}
    """
    async def permission_dependency(user = Depends(get_current_user)):
        # Lấy thông tin user từ database
        db_user = supabase.table("users").select("*").eq("id", user.id).execute()
        
        if not db_user.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy thông tin user"
            )
        
        user_data = db_user.data[0]
        
        # Kiểm tra role
        if user_data.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_permissions",
                    "message": f"Bạn không có quyền truy cập. Cần role: {', '.join(allowed_roles)}"
                }
            )
        
        return user_data
    
    return permission_dependency


#  Self or Admin Access 
async def verify_self_or_admin(
    user_id: str,
    current_user = Depends(get_current_user_with_db)
):
    """
    Kiểm tra user có phải đang truy cập thông tin của chính mình
    hoặc là admin/owner
    
    Usage:
        @router.get("/users/{user_id}")
        async def get_user(
            user_id: str,
            verified_user = Depends(verify_self_or_admin)
        ):
            return get_user_by_id(user_id)
    """
    # Admin/Owner được phép truy cập mọi user
    if current_user.get("role") in ["admin", "owner"]:
        return current_user
    
    # User thường chỉ được truy cập thông tin của mình
    if current_user["id"] != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "access_denied",
                "message": "Bạn chỉ có thể truy cập thông tin của chính mình"
            }
        )
    
    return current_user


#  Optional Authentication 
async def get_current_user_optional(authorization: str = None):
    """
    Lấy thông tin user nếu có token, không bắt buộc
    Trả về None nếu không có token hoặc token không hợp lệ
    
    Usage:
        @router.get("/public-products")
        async def list_products(
            current_user = Depends(get_current_user_optional)
        ):
            if current_user:
                # Hiển thị giá ưu đãi cho user đã đăng nhập
                return get_products_with_discount()
            return get_products()
    """
    if not authorization:
        return None
    
    try:
        user = get_current_user(authorization)
        db_user = supabase.table("users").select("*").eq("id", user.id).execute()
        return db_user.data[0] if db_user.data else None
    except:
        return None


#  Check Account Status 
def check_account_active(user_data: dict):
    """
    Kiểm tra tài khoản có đang hoạt động không
    
    Usage:
        current_user = await get_current_user_with_db(user)
        check_account_active(current_user)
    """
    if user_data.get("status") == False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "account_disabled",
                "message": "Tài khoản của bạn đã bị vô hiệu hóa. Vui lòng liên hệ admin."
            }
        )
    return True


#  Prevent Self Action 
def prevent_self_action(target_user_id: str, current_user: dict, action: str = "thao tác"):
    """
    Ngăn user thực hiện action lên chính mình
    
    Usage:
        prevent_self_action(user_id, current_user, "xóa")
        prevent_self_action(user_id, current_user, "ban")
    """
    if current_user["id"] == target_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "self_action_not_allowed",
                "message": f"Bạn không thể {action} chính mình"
            }
        )
    return True