# #api/hairStyle_api.py
#
# from pathlib import Path
# import os
# import torch
#
# from fastapi import APIRouter
# class HairStyleConfig:
#
#     # ==================== DIRECTORIES ====================
#     BASE_DIR = Path(__file__).parent.parent.parent
#     MODELS_DIR = BASE_DIR / "models" / "hairstyle"
#     CACHE_DIR = BASE_DIR / "cache" / "hairstyle"
#     OUTPUT_DIR = BASE_DIR / "outputs" / "hairstyle"
#     TEMP_DIR = BASE_DIR / "temp" / "hairstyle"
#
#     # ==================== MODEL IDS ====================
#     # Stable Diffusion 1.5
#     SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
#
#     # ControlNet
#     CONTROLNET_FACEID = "lllyasviel/control_v11p_sd15_canny"
#     # Alternative ControlNet models:
#     # - "h94/IP-Adapter-FaceID"
#     # - "CrucibleAI/ControlNetMediaPipeFace"
#
#     # InsightFace model for face detection
#     FACEID_MODEL = "buffalo_l"
#
#     # ==================== DEVICE SETTINGS ====================
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     USE_XFORMERS = True  # Memory efficient attention
#     USE_FLOAT16 = True  # FP16 for faster inference on GPU
#
#     # ==================== GENERATION SETTINGS ====================
#     NUM_INFERENCE_STEPS = 25  # 20-30 for speed, 30-50 for quality
#     GUIDANCE_SCALE = 7.5  # 7-8 standard, 8-10 for more detail
#     NUM_IMAGES_PER_STYLE = 1  # Images to generate per style
#     IMAGE_SIZE = (512, 512)  # SD1.5 standard resolution
#
#     # ==================== FACEID SETTINGS ====================
#     FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection
#     FACE_EMBEDDING_SIZE = 512  # Face embedding dimension
#     CONTROLNET_CONDITIONING_SCALE = 0.8  # Strength of face preservation (0.5-1.0)
#
#     # ==================== API SETTINGS ====================
#     MAX_QUEUE_SIZE = 10  # Maximum async job queue size
#     TIMEOUT_SECONDS = 180  # Request timeout (3 minutes)
#     ENABLE_SAFETY_CHECKER = True  # Enable NSFW content filter
#     MAX_BATCH_SIZE = 12  # Maximum images per batch request
#
#     # ==================== QUALITY PROMPTS ====================
#     # Universal quality enhancers (applied to all styles)
#     QUALITY_POSITIVE = (
#         "high quality, professional photo, 8k uhd, realistic, "
#         "detailed hair texture, natural lighting, sharp focus, "
#         "photograph, hyperrealistic"
#     )
#
#     QUALITY_NEGATIVE = (
#         "low quality, blurry, distorted, deformed, ugly, "
#         "bad anatomy, disfigured, mutation, extra heads, "
#         "cartoon, anime, drawing, painting, illustration, "
#         "worst quality, low res, jpeg artifacts"
#     )
#
#     # ==================== METHODS ====================
#     @classmethod
#     def setup_directories(cls):
#         """Create necessary directories for models, cache, outputs"""
#         for dir_path in [cls.MODELS_DIR, cls.CACHE_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
#             dir_path.mkdir(exist_ok=True, parents=True)
#
#     @classmethod
#     def get_device_info(cls) -> dict:
#         """Get device information"""
#         info = {
#             "device": cls.DEVICE,
#             "cuda_available": torch.cuda.is_available(),
#         }
#
#         if torch.cuda.is_available():
#             info.update({
#                 "gpu_name": torch.cuda.get_device_name(0),
#                 "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
#                 "cuda_version": torch.version.cuda,
#             })
#
#         return info
#
#     @classmethod
#     def validate_settings(cls) -> bool:
#         """Validate configuration settings"""
#         errors = []
#
#         # Check VRAM if using GPU
#         if cls.DEVICE == "cuda":
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
#             if gpu_memory < 6:
#                 errors.append(f"GPU memory ({gpu_memory:.2f}GB) is less than recommended 6GB")
#
#         # Check image size
#         if cls.IMAGE_SIZE[0] > 768 or cls.IMAGE_SIZE[1] > 768:
#             errors.append(f"Image size {cls.IMAGE_SIZE} may require more VRAM. Recommended: (512, 512)")
#
#         # Check steps
#         if cls.NUM_INFERENCE_STEPS < 10:
#             errors.append("NUM_INFERENCE_STEPS too low. Minimum: 10")
#
#         if errors:
#             print("⚠️ Configuration warnings:")
#             for error in errors:
#                 print(f"  - {error}")
#             return False
#
#         return True
#
# hairstyle_router = APIRouter()
# @hairstyle_router.get("/hair/device-info")
# async def get_device_info():
#     """Kiểm tra GPU máy chủ trước khi chạy AI"""
#     return HairStyleConfig.get_device_info()
#
#
# @hairstyle_router.get("/hair/test")
# async def test_hair_api():
#     return {"status": "Hair API OK"}
# # Initialize directories on import
# HairStyleConfig.setup_directories()