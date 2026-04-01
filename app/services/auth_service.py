from fastapi import HTTPException, status
from app.database.supabase_client import supabase

def refresh_access_token(refresh_token: str) -> dict:
    """
    Làm mới access token bằng refresh token
    
    Args:
        refresh_token: Refresh token từ client
        
    Returns:
        dict: Chứa access_token, refresh_token mới và thông tin token
        
    Raises:
        HTTPException: Nếu refresh token không hợp lệ hoặc đã hết hạn
    """
    try:
        # Gọi Supabase để refresh session
        response = supabase.auth.refresh_session(refresh_token)

        # Kiểm tra response
        if not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )

        # Trả về token mới
        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "token_type": "bearer",
            "expires_in": response.session.expires_in or 3600  # Default 1 hour
        }

    except HTTPException:
        # Re-raise HTTPException từ trên
        raise
        
    except Exception as e:
        # Log lỗi (nếu có logger)
        print(f" Refresh token error: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to refresh token. Please login again."
        )