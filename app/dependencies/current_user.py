from fastapi import Depends, HTTPException, Header, status
from app.database.supabase_client import supabase
from supabase_auth.errors import AuthApiError

def get_current_user(authorization: str = Header(...)):
    """Validate access token và return user"""
    
    # Kiểm tra format Bearer token
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    token = authorization.split(" ")[1]
    
    try:
        # Validate token với Supabase
        user = supabase.auth.get_user(token).user
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return user
        
    except AuthApiError as e:
        error_msg = str(e).lower()
        
        # Token expired - FE sẽ catch và tự động refresh
        if "expired" in error_msg or "invalid claims" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "token_expired",
                    "message": "Access token đã hết hạn"
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Token invalid (sai format, signature không đúng)
        if "invalid" in error_msg or "unable to parse" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "token_invalid",
                    "message": "Access token không hợp lệ"
                }
            )
        
        # Forbidden (user bị ban, permission denied...)
        if "forbidden" in error_msg or "403" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "access_forbidden",
                    "message": "Không có quyền truy cập"
                }
            )
        
        # Lỗi auth khác
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "auth_error",
                "message": f"Lỗi xác thực: {str(e)}"
            }
        )
        
    except Exception as e:
        # Lỗi server không mong muốn
        print(f"Unexpected error in get_current_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi server khi xác thực"
        )