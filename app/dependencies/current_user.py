from fastapi import Depends, HTTPException, Header
from app.database.supabase_client import supabase

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    token = authorization.split(" ")[1]
    user = supabase.auth.get_user(token).user
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user
