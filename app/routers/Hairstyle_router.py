from fastapi import APIRouter, UploadFile, File, HTTPException, Query,Request, Depends
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.dependencies.current_user import get_current_user

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(
    prefix="/api/v1/hairstyle",
    tags=["Hair Style"]
)

logger = logging.getLogger(__name__)


@router.post("/generate" )
@limiter.limit("10/minute")

async def generate_hairstyle(
    request:Request,
    file: UploadFile = File(..., description="Input face image"),
    style: str = Query(..., description="Hair style ID"),
    seed: int | None = Query(None),
    steps: int = Query(30, ge=10, le=50),
    denoising_strength: float = Query(0.35, ge=0.1, le=0.8),

):
    """
    Gửi 1 ảnh → trả về 1 ảnh đã đổi kiểu tóc
    """
    try:
        # Import service
        from app.services.Hairstyle_service import get_hair_generator
        from app.config.hair_config import HairStylePrompts

        # Check style hợp lệ
        if style not in HairStylePrompts.HAIR_STYLES:
            raise HTTPException(status_code=400, detail="Invalid style name")

        # Đọc ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        # Load model
        generator = get_hair_generator()

        # Generate
        result = generator.generate_single_style(
            image=image,
            style_name=style,
            seed=seed,
            num_steps=steps,
            denoising_strength=denoising_strength
        )

        # Trả ảnh
        img_bytes = io.BytesIO()
        result["result"].save(img_bytes, format="PNG", quality=95)
        img_bytes.seek(0)

        return StreamingResponse(
            img_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=hairstyle.png"
            }
        )

    except Exception as e:
        logger.error(f"Generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/styles")
async def get_available_styles():
    """Get list of all available hair styles"""
    try:
        # Import config
        from app.config.hair_config import HairStylePrompts

        styles = []
        for style_id in HairStylePrompts.get_style_list():
            style_info = HairStylePrompts.get_style_info(style_id)
            styles.append({
                "id": style_info["id"],
                "name": style_info["name"],
                "description": style_info["prompt"][:100] + "...",
                "gender": style_info.get("gender", "unisex"),
                "category": style_info.get("category", "general"),
            })

        return {
            "total": len(styles),
            "styles": styles
        }

    except Exception as e:
        logger.error(f"Error getting styles: {e}")
        raise HTTPException(status_code=500, detail=str(e))