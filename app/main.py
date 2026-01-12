from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api.acneAPI import router as acne_router
from app.routers.user_router import router as user_router
from app.routers.barbers_router import router as barbers_router
from app.routers.service_router import router as service_router
from app.routers.booking_router import router as booking_router
from app.routers.ratings_router import router as ratings_router
from app.routers.Hairstyle_router import router as Hairstyle_router
from app.routers.appointment_router import router as appointment_router
from app.routers.time_slot_router import router as time_slot_router
from app.routers.rag_router import router as rag_router
from fastapi.middleware.cors import CORSMiddleware
from app.routers.email_router import router as email_router

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Acne Detection API and Supabase FastAPI",
    description="API phát hiện mụn sử dụng YOLOv8 và MediaPipe Face Mesh và quản lý người dùng với Supabase",
    version="1.0.0"
)
# Add limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vue dev server
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
        "http://localhost:3000",      # Backup
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],              # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],              # Authorization, Content-Type, etc.
)
app.include_router(acne_router)
app.include_router(user_router)
app.include_router(barbers_router)
app.include_router(service_router)
app.include_router(booking_router)
app.include_router(ratings_router)
app.include_router(appointment_router)
app.include_router(time_slot_router)
app.include_router(rag_router)

app.include_router(Hairstyle_router)
app.include_router(email_router)


@app.get("/")
async def root():
    return {
        "message": "Acne Detection API, and Supabase FastAPI is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }