from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import traceback

from app.services.acne_detection import AcneDetectionService
from app.services.advice_generator import AdviceGenerator

router = APIRouter(
    prefix="/acne",
    tags=["Acne Detection"]
)

# Khởi tạo services với error handling tốt hơn
acne_service = None
advice_generator = None

print("🔧 Initializing Acne Detection Service...")
try:
    acne_service = AcneDetectionService()
    print("Acne Detection Service initialized")
except Exception as e:
    print(f" Failed to initialize AcneDetectionService: {e}")
    traceback.print_exc()

print("🔧 Initializing Advice Generator...")
try:
    advice_generator = AdviceGenerator()
    print("Advice Generator initialized")
except Exception as e:
    print(f" Failed to initialize AdviceGenerator: {e}")
    traceback.print_exc()
    # Nếu không có Gemini API key, vẫn tiếp tục nhưng không có advice


def read_image_file(file_bytes: bytes) -> np.ndarray:
    """Đọc file ảnh thành numpy array (BGR)"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Cannot read image: {str(e)}")


@router.post("/detect")
async def detect_acne(image: UploadFile = File(...)):
    """
    Phát hiện mụn từ 1 ảnh chính diện (Binary Detection)
    """
    try:
        # Check service availability
        if acne_service is None:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Acne Detection Service chưa được khởi tạo. Vui lòng khởi động lại server."
                }
            )

        print("\n" + "=" * 60)
        print(" Received acne detection request (Binary Classification)")

        # Đọc ảnh
        print(" Reading image...")
        img = read_image_file(await image.read())
        print(f"Image loaded: {img.shape}")

        # Process ảnh
        print("Processing image with CNN model...")
        results = acne_service.process_image(img)

        # Kiểm tra nếu không detect được mặt
        if not results:
            print("No face detected in image")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Không phát hiện được khuôn mặt trong ảnh. Vui lòng chụp rõ hơn."
                }
            )

        # Tạo summary
        print("\n Creating summary...")
        summary_data = acne_service.get_summary(results)

        acne_regions = summary_data['acne_regions']
        clear_regions = summary_data['clear_regions']
        overall_severity = summary_data['overall_severity']
        severity_count = summary_data['severity_count']

        print(f"Detection complete!")
        print(f"   Total regions analyzed: {summary_data['total_regions']}")
        print(f"   Regions with acne: {acne_regions}")
        print(f"   Clear regions: {clear_regions}")
        print(f"   Overall severity: {overall_severity}")
        print(f"   Severity distribution: {severity_count}")

        # In ra chi tiết từng vùng
        print("\n REGION DETAILS:")
        for region, data in results.items():
            has_acne = data['has_acne']
            confidence = data['confidence']
            severity = data['severity']

            status = " CÓ MỤN" if has_acne else " SẠCH"
            print(f"   {status} {region}: {severity} (conf: {confidence:.3f})")

        # Tạo lời khuyên (với fallback nếu advice_generator không có)
        advice = []
        overall_summary = {}

        if advice_generator is not None:
            try:
                print("\n Generating personalized advice...")
                advice = advice_generator.generate_advice(results)
                print(f"Generated {len(advice)} advice items")

                # Tạo overall summary
                overall_summary = advice_generator.get_overall_summary(advice, summary_data)
                print(f"\n Overall Assessment:")
                print(f"   Severity: {overall_summary['severity']}")
                print(f"   Recommendation: {overall_summary['recommendation']}")
                print(f"   Need doctor: {overall_summary['need_doctor']}")

            except Exception as e:
                print(f"Failed to generate advice: {e}")
                traceback.print_exc()
                # Fallback: basic advice
                advice = [{
                    'zone': 'Tổng quan',
                    'message': 'Không thể tạo lời khuyên chi tiết. Vui lòng tham khảo bác sĩ da liễu.'
                }]
                overall_summary = {
                    'severity': overall_severity,
                    'recommendation': 'Tham khảo bác sĩ da liễu để được tư vấn cụ thể.',
                    'need_doctor': overall_severity in ['moderate', 'severe'],
                    'affected_regions': acne_regions
                }
        else:
            print("Advice Generator not available, using basic advice")
            advice = [{
                'zone': 'Tổng quan',
                'message': 'Advice Generator chưa được khởi tạo. Vui lòng kiểm tra GEMINI_API_KEY.'
            }]
            overall_summary = {
                'severity': overall_severity,
                'recommendation': 'Service chưa sẵn sàng. Vui lòng kiểm tra cấu hình.',
                'need_doctor': overall_severity in ['moderate', 'severe'],
                'affected_regions': acne_regions
            }

        print("=" * 60 + "\n")

        # Response
        return JSONResponse({
            "success": True,
            "data": {
                "regions": results,
                "summary": {
                    "total_regions": summary_data['total_regions'],
                    "acne_regions": acne_regions,
                    "clear_regions": clear_regions,
                    "overall_severity": overall_severity,
                    "average_confidence": summary_data['average_confidence'],
                    "severity_count": severity_count
                },
                "advice": advice,
                "overall": overall_summary
            }
        })

    except ValueError as e:
        print(f" ValueError: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Lỗi xử lý ảnh: {str(e)}"
            }
        )

    except Exception as e:
        print("\n" + "=" * 60)
        print(" UNEXPECTED ERROR:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60 + "\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Lỗi hệ thống: {str(e)}",
                "error_type": type(e).__name__
            }
        )


@router.get("/health")
async def health_check():
    """Kiểm tra service có hoạt động không"""
    return {
        "status": "healthy" if (acne_service and advice_generator) else "degraded",
        "services": {
            "acne_detection": acne_service is not None,
            "advice_generator": advice_generator is not None
        }
    }