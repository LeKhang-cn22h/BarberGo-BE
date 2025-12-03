from fastapi import APIRouter, HTTPException
from app.database.supabase_client import supabase
from app.schemas.user_schema import UserCreate

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/")
def create_user(data: UserCreate):
    response = supabase.table("users").insert(data.dict()).execute()
    return response.data

@router.get("/")
def get_all_users():
    response = supabase.table("users").select("*").execute()
    return response.data

@router.get("/{user_id}")
def get_user(user_id: int):
    response = supabase.table("users").select("*").eq("id", user_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    return response.data[0]

@router.put("/{user_id}")
def update_user(user_id: int, data: UserCreate):
    response = supabase.table("users").update(data.dict()).eq("id", user_id).execute()
    return response.data

@router.delete("/{user_id}")
def delete_user(user_id: int):
    response = supabase.table("users").delete().eq("id", user_id).execute()
    return {"message": "User deleted"}
