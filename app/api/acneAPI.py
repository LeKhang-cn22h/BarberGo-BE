# app/api/acneAPI.py
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

# Khởi tạo services
print("🔧 Initializing Acne Detection Service...")
try:
    acne_service = AcneDetectionService()
    advice_generator = AdviceGenerator()
    print("✅ Acne Detection Service initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize service: {e}")
    traceback.print_exc()


def read_image_file(file_bytes: bytes) -> np.ndarray:
    """Đọc file ảnh thành numpy array (BGR)"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Cannot read image: {str(e)}")


@router.post("/detect")
async def detect_acne(
        image: UploadFile = File(...)
):
    """
    Phát hiện mụn từ 1 ảnh chính diện (Binary Detection)

    Args:
        image: Ảnh khuôn mặt chính diện

    Returns:
        {
            "success": true,
            "data": {
                "regions": {
                    "forehead": {
                        "has_acne": bool,
                        "confidence": float,
                        "severity": str        # none/mild/moderate/severe
                    },
                    ...
                },
                "summary": {
                    "total_regions": int,
                    "acne_regions": int,       # Số vùng có mụn
                    "clear_regions": int,      # Số vùng sạch
                    "overall_severity": str,   # Mức độ chung
                    "average_confidence": float,
                    "severity_count": {
                        "none": 5,
                        "mild": 2,
                        "moderate": 1,
                        "severe": 0
                    }
                },
                "advice": [...],
                "overall": {
                    "severity": str,
                    "recommendation": str,
                    "need_doctor": bool
                }
            }
        }
    """
    try:
        print("\n" + "=" * 60)
        print("📸 Received acne detection request (Binary Classification)")

        # ✅ Đọc ảnh
        print("📖 Reading image...")
        img = read_image_file(await image.read())
        print(f"✅ Image loaded: {img.shape}")

        # ✅ Process ảnh
        print("🔍 Processing image with CNN model...")
        results = acne_service.process_image(img)

        # Kiểm tra nếu không detect được mặt
        if not results:
            print("⚠️  No face detected in image")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Không phát hiện được khuôn mặt trong ảnh. Vui lòng chụp rõ hơn."
                }
            )

        # ✅ Tạo summary (tổng hợp)
        print("\n📊 Creating summary...")
        summary_data = acne_service.get_summary(results)

        acne_regions = summary_data['acne_regions']
        clear_regions = summary_data['clear_regions']
        overall_severity = summary_data['overall_severity']
        severity_count = summary_data['severity_count']

        print(f"✅ Detection complete!")
        print(f"   Total regions analyzed: {summary_data['total_regions']}")
        print(f"   Regions with acne: {acne_regions}")
        print(f"   Clear regions: {clear_regions}")
        print(f"   Overall severity: {overall_severity}")
        print(f"   Severity distribution: {severity_count}")

        # ✅ In ra chi tiết từng vùng
        print("\n📋 REGION DETAILS:")
        for region, data in results.items():
            has_acne = data['has_acne']
            confidence = data['confidence']
            severity = data['severity']

            status = "🔴 CÓ MỤN" if has_acne else "🟢 SẠCH"
            print(f"   {status} {region}: {severity} (conf: {confidence:.3f})")

        # ✅ Tạo lời khuyên (cần update AdviceGenerator cho binary)
        print("\n💡 Generating personalized advice...")
        advice = advice_generator.generate_advice(results)
        print(f"✅ Generated {len(advice)} advice items")

        # ✅ Tạo overall summary
        overall_summary = advice_generator.get_overall_summary(advice, summary_data)
        print(f"\n📈 Overall Assessment:")
        print(f"   Severity: {overall_summary['severity']}")
        print(f"   Recommendation: {overall_summary['recommendation']}")
        print(f"   Need doctor: {overall_summary['need_doctor']}")

        print("=" * 60 + "\n")

        # ✅ Response (format cho binary classification)
        return JSONResponse({
            "success": True,
            "data": {
                "regions": results,  # Chi tiết từng vùng
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
        print(f"❌ ValueError: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Lỗi xử lý ảnh: {str(e)}"
            }
        )

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ UNEXPECTED ERROR:")
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
    try:
        test_result = acne_service is not None
        return {
            "status": "healthy" if test_result else "unhealthy",
            "service": "acne_detection",
            "model_loaded": test_result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# # app/api/acneAPI.py
# from fastapi import APIRouter, File, UploadFile
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# from PIL import Image
# import io
# import traceback
#
# from app.services.acne_detection import AcneDetectionService
# from app.services.advice_generator import AdviceGenerator
#
# router = APIRouter(
#     prefix="/acne",
#     tags=["Acne Detection"]
# )
#
# # Khởi tạo services
# print("🔧 Initializing Acne Detection Service...")
# try:
#     acne_service = AcneDetectionService()
#     advice_generator = AdviceGenerator()
#     print("✅ Acne Detection Service initialized successfully")
# except Exception as e:
#     print(f"❌ Failed to initialize service: {e}")
#     traceback.print_exc()
#
#
# def read_image_file(file_bytes: bytes) -> np.ndarray:
#     """Đọc file ảnh thành numpy array (BGR)"""
#     try:
#         image = Image.open(io.BytesIO(file_bytes))
#         return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     except Exception as e:
#         raise ValueError(f"Cannot read image: {str(e)}")
#
#
# @router.post("/detect")
# async def detect_acne(
#         image: UploadFile = File(...)  # ✅ CHỈ 1 ẢNH
# ):
#     """
#     Phát hiện và phân loại mụn từ 1 ảnh chính diện
#
#     Args:
#         image: Ảnh khuôn mặt chính diện
#
#     Returns:
#         {
#             "success": true,
#             "data": {
#                 "results": {
#                     "forehead": {
#                         "acne_type": str,      # 'pustules', 'blackheads', ...
#                         "confidence": float,
#                         "top_3": [...]         # Optional
#                     },
#                     ...
#                 },
#                 "summary": {
#                     "total_regions": int,
#                     "acne_count": {            # Số lượng từng loại mụn
#                         "pustules": 2,
#                         "blackheads": 1,
#                         ...
#                     },
#                     "most_common": str         # Loại mụn phổ biến nhất
#                 },
#                 "advice": [...]
#             }
#         }
#     """
#     try:
#         print("\n" + "=" * 60)
#         print("📸 Received acne detection request (1 image)")
#
#         # ✅ Đọc ảnh
#         print("📖 Reading image...")
#         img = read_image_file(await image.read())
#         print(f"✅ Image loaded: {img.shape}")
#
#         # ✅ Process ảnh
#         print("🔍 Processing image with CNN model...")
#         results = acne_service.process_image(img)
#
#         # Kiểm tra nếu không detect được mặt
#         if not results:
#             print("⚠️  No face detected in image")
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "success": False,
#                     "error": "Không phát hiện được khuôn mặt trong ảnh. Vui lòng chụp rõ hơn."
#                 }
#             )
#
#         # ✅ Tạo summary (tổng hợp)
#         print("\n📊 Creating summary...")
#         summary_data = acne_service.get_summary(results)
#
#         total_acne_zones = summary_data['total_regions']
#         acne_count = summary_data['acne_count']
#         most_common = summary_data['most_common']
#
#         print(f"✅ Detection complete!")
#         print(f"   Total regions analyzed: {len(results)}")
#         print(f"   Regions with acne: {total_acne_zones}")
#         print(f"   Acne distribution: {acne_count}")
#         print(f"   Most common type: {most_common}")
#
#         # ✅ In ra chi tiết từng vùng
#         print("\n📋 REGION DETAILS:")
#         for region, data in results.items():
#             acne_type = data.get('acne_type', 'none')
#             confidence = data.get('confidence', 0.0)
#
#             if acne_type != 'none':
#                 print(f"   ✓ {region}: {acne_type} (conf: {confidence:.3f})")
#             else:
#                 print(f"   ✗ {region}: No acne (conf: {confidence:.3f})")
#
#         # ✅ Tạo lời khuyên
#         print("\n💡 Generating personalized advice...")
#         advice = advice_generator.generate_advice(results)
#         print(f"✅ Generated {len(advice)} advice items")
#
#         # ✅ Tạo overall summary
#         overall_summary = advice_generator.get_overall_summary(advice)
#         print(f"\n📈 Overall Assessment:")
#         print(f"   Severity: {overall_summary['overall_severity']}")
#         print(f"   Recommendation: {overall_summary['recommendation']}")
#         print(f"   Need doctor: {overall_summary['need_doctor']}")
#
#         print("=" * 60 + "\n")
#
#         # ✅ Response (format mới)
#         return JSONResponse({
#             "success": True,
#             "data": {
#                 "results": results,  # ← Đổi tên: summary → results
#                 "summary": {
#                     "total_regions": len(results),
#                     "acne_count": acne_count,
#                     "most_common": most_common
#                 },
#                 "advice": advice,
#                 "overall": overall_summary  # ← Thêm overall assessment
#             }
#         })
#
#     except ValueError as e:
#         print(f"❌ ValueError: {str(e)}")
#         return JSONResponse(
#             status_code=400,
#             content={
#                 "success": False,
#                 "error": f"Lỗi xử lý ảnh: {str(e)}"
#             }
#         )
#
#     except Exception as e:
#         print("\n" + "=" * 60)
#         print("❌ UNEXPECTED ERROR:")
#         print(f"Error type: {type(e).__name__}")
#         print(f"Error message: {str(e)}")
#         print("\nFull traceback:")
#         traceback.print_exc()
#         print("=" * 60 + "\n")
#
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "error": f"Lỗi hệ thống: {str(e)}",
#                 "error_type": type(e).__name__
#             }
#         )
#
#
# @router.get("/health")
# async def health_check():
#     """Kiểm tra service có hoạt động không"""
#     try:
#         test_result = acne_service is not None
#         return {
#             "status": "healthy" if test_result else "unhealthy",
#             "service": "acne_detection",
#             "model_loaded": test_result
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }