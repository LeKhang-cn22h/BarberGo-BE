from fastapi import FastAPI
from app.api.acneAPI import router as acne_router
from app.routers.user_router import router as user_router
from app.routers.barbers_router import router as barbers_router
from app.routers.service_router import router as service_router
from app.routers.booking_router import router as booking_router
from app.routers.ratings_router import router as ratings_router
from app.routers.Hairstyle_router import router as Hairstyle_router
app = FastAPI(
    title="Acne Detection API and Supabase FastAPI",
    description="API phát hiện mụn sử dụng YOLOv8 và MediaPipe Face Mesh và quản lý người dùng với Supabase",
    version="1.0.0"
)

app.include_router(acne_router)
app.include_router(user_router)
app.include_router(barbers_router)
app.include_router(service_router)
app.include_router(booking_router)
app.include_router(ratings_router)
app.include_router(Hairstyle_router)

@app.get("/")
async def root():
    return {
        "message": "Acne Detection API, and Supabase FastAPI is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }