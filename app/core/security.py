# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
# from jose import JWTError, jwt

# # --- QUAN TRỌNG: Import đúng cái class User bạn vừa gửi ---
# # Giả sử file chứa code bạn gửi nằm ở app/schemas/user_schemas.py
# from app.schemas.user_schema import User 

# SECRET_KEY = "chuoi-bi-mat-cua-ban" 
# ALGORITHM = "HS256"

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Không thể xác thực người dùng",
#         headers={"WWW-Authenticate": "Bearer"},
#     )

#     try:
#         # 1. Giải mã token
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
#         # 2. Lấy thông tin từ token
#         # Token thường chứa: sub (id hoặc email), role, v.v.
#         user_id: str = payload.get("sub") 
#         email: str = payload.get("email") 
#         role: str = payload.get("role")
        
#         if user_id is None or email is None:
#             raise credentials_exception
            
#         # 3. Tạo object User từ thông tin trong Token
#         # Ở đây mình dùng class User của bạn để hứng dữ liệu
#         # Lưu ý: Token phải chứa đủ thông tin này, hoặc bạn phải query DB ở đây
#         user_data = User(
#             id=user_id,
#             email=email,
#             role=role if role else "user", # Mặc định là user thường
#             full_name=payload.get("full_name"), # Nếu token có lưu tên
#             phone=payload.get("phone")
#         )
        
#         return user_data

#     except JWTError:
#         raise credentials_exception
#     except Exception as e:
#         # Bắt lỗi nếu dữ liệu không khớp với Pydantic model
#         print(f"Lỗi validate user: {e}") 
#         raise credentials_exception