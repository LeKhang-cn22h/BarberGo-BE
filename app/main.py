from fastapi import FastAPI
from app.api.acneAPI import router as acne_router
from app.routers.user_router import router as user_router

app = FastAPI(
    title="Acne Detection API and Supabase FastAPI",
    description="API phát hiện mụn sử dụng YOLOv8 và MediaPipe Face Mesh và quản lý người dùng với Supabase",
    version="1.0.0"
)

app.include_router(acne_router)

@app.get("/")
async def root():
    return {
        "message": "Acne Detection API, and Supabase FastAPI is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }