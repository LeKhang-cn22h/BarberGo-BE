from fastapi import FastAPI
from app.api.acneAPI import router as acne_router

app = FastAPI(
    title="Acne Detection API",
    description="API phát hiện mụn sử dụng YOLOv8 và MediaPipe Face Mesh",
    version="1.0.0"
)

app.include_router(acne_router)

@app.get("/")
async def root():
    return {
        "message": "Acne Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }