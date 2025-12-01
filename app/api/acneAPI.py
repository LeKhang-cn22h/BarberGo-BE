from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.acne_detection import AcneDetectionService
from app.services.advice_generator import AdviceGenerator
from app.schemas.acne import AcneDetectionResponse
import cv2
import numpy as np

router = APIRouter(prefix="/acne", tags=["acne"])

# Khởi tạo services
acne_service = AcneDetectionService()
advice_generator = AdviceGenerator()


def read_image(file: UploadFile):
    """Đọc ảnh từ upload file"""
    content = file.file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


@router.post("/detect", response_model=AcneDetectionResponse)
async def detect_acne(
        left_image: UploadFile = File(..., description="Ảnh mặt trái"),
        front_image: UploadFile = File(..., description="Ảnh mặt chính diện"),
        right_image: UploadFile = File(..., description="Ảnh mặt phải")
):
    """
    API phát hiện mụn từ 3 góc chụp
    """
    try:
        # Đọc 3 ảnh
        images = {
            'left': read_image(left_image),
            'front': read_image(front_image),
            'right': read_image(right_image)
        }

        all_results = {}

        # Xử lý từng ảnh
        for position, img in images.items():
            result = acne_service.process_single_image(img)

            if not result:
                raise HTTPException(
                    status_code=400,
                    detail=f"Không phát hiện khuôn mặt trong ảnh {position}"
                )

            all_results[position] = result

        # Tổng hợp kết quả
        summary = acne_service.aggregate_results(all_results)

        # Tạo lời khuyên
        advice = advice_generator.generate_advice(summary)

        # Tính tổng mụn
        total_acne = sum(data['total'] for data in summary.values())

        return {
            "success": True,
            "data": {
                "summary": summary,
                "total_acne": total_acne,
                "advice": advice,
                "detailed_results": all_results
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
