"""
Hair Style Generation Service - SD1.5 + Inpainting với MediaPipe
FAST VERSION: Chạy nhanh, ít VRAM, dễ debug
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
import time
import mediapipe as mp
import os
from pathlib import Path

from diffusers import (
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

logger = logging.getLogger(__name__)


class HairStyleConfig:
    """Config cho SD1.5 + Inpainting"""
    # Model SD1.5 Inpainting
    SD15_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"

    # Hoặc các model alternatives:
    # "stabilityai/stable-diffusion-2-inpainting"
    # "digiplay/AbsoluteReality_v1.8.1"  # Realistic
    # "SG161222/Realistic_Vision_V5.1_noVAE"  # Photorealistic

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_XFORMERS = True if DEVICE == "cuda" else False

    # SD1.5 dùng 512x512
    IMAGE_SIZE = (512, 512)

    # Generation settings
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    DENOISING_STRENGTH = 0.75  # Cao vì mask nhỏ

    # Mask settings
    MASK_DILATE = 10
    FACE_PROTECTION_PADDING = 0.25

    # Debug
    DEBUG_MODE = True
    DEBUG_DIR = "./debug_masks"


class FaceDetectorMediaPipe:
    """Face detection với MediaPipe - Nhanh và chính xác"""

    def __init__(self, config: HairStyleConfig):
        self.config = config
        logger.info("Initializing MediaPipe Face Detection...")

        # Khởi tạo MediaPipe
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 1=full-range detection
            min_detection_confidence=0.5
        )

        logger.info("✓ MediaPipe Face Detection initialized!")

    def get_face_info(self, image: np.ndarray) -> Optional[Dict]:
        """Get face bounding box"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Detect face
            results = self.face_detection.process(rgb_image)

            if results.detections:
                # Lấy face có confidence cao nhất
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                # Convert relative to absolute coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                bbox_abs = [
                    max(0, x),
                    max(0, y),
                    min(w, x + width),
                    min(h, y + height)
                ]

                return {
                    'bbox': bbox_abs,
                    'confidence': detection.score[0],
                    'width': width,
                    'height': height
                }

            return None

        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return None


    def get_forehead_position(self, image: np.ndarray, face_info: Dict) -> int:
        """Lấy đường chân tóc CHUẨN NHẤT bằng MediaPipe Face Mesh"""
        h, w = image.shape[:2]

        # Khởi tạo MediaPipe Face Mesh nếu chưa có
        if not hasattr(self, 'face_mesh'):
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process với Face Mesh
            results = self.face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Lấy các landmarks quan trọng cho chân tóc
                # Landmark indices cho forehead/hairline:
                # 10: giữa trán trên
                # 67, 69, 109: trán trái
                # 298, 299, 300: trán phải
                # 151: điểm giữa chân tóc

                forehead_points = []

                # Điểm giữa chân tóc (landmark 151)
                if len(face_landmarks.landmark) > 151:
                    lm = face_landmarks.landmark[151]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    forehead_points.append((px, py))
                    print(f"🔍 Landmark 151 (mid forehead): y={py}")

                # Điểm trên trán (landmark 10)
                if len(face_landmarks.landmark) > 10:
                    lm = face_landmarks.landmark[10]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    forehead_points.append((px, py))
                    print(f"🔍 Landmark 10 (forehead top): y={py}")

                # Lấy điểm thấp nhất (gần lông mày nhất)
                if forehead_points:
                    # Lấy tọa độ y của tất cả điểm trán
                    y_values = [p[1] for p in forehead_points]

                    # Chọn điểm THẤP NHẤT (gần lông mày nhất) làm hairline
                    hairline_y = min(y_values)

                    # Điều chỉnh: lùi lên 5% face height để chắc chắn
                    x1, y1, x2, y2 = face_info['bbox']
                    face_height = y2 - y1
                    hairline_y = max(0, hairline_y - int(face_height * 0.05))

                    print(f"✅ Calculated hairline_y: {hairline_y}")

                    # Vẽ debug
                    if self.config.DEBUG_MODE:
                        debug_img = image.copy()
                        # Vẽ landmarks
                        for px, py in forehead_points:
                            cv2.circle(debug_img, (px, py), 5, (0, 255, 0), -1)
                        # Vẽ hairline
                        cv2.line(debug_img, (0, hairline_y), (w, hairline_y), (0, 0, 255), 3)
                        cv2.imwrite(f'{self.config.DEBUG_DIR}/hairline_landmarks.png', debug_img)

                    return hairline_y

            # Fallback: dùng logic cũ nếu không detect được landmarks
            print("⚠️ Cannot detect landmarks, using fallback method")
            x1, y1, x2, y2 = face_info['bbox']
            face_height = y2 - y1
            hairline_y = int(y1 + face_height * 0.33)
            return hairline_y

        except Exception as e:
            print(f"❌ Error in hairline detection: {e}")
            # Fallback
            x1, y1, x2, y2 = face_info['bbox']
            face_height = y2 - y1
            return int(y1 + face_height * 0.33)

    def create_face_protection_mask(
            self,
            image: np.ndarray,
            face_info: Dict
    ) -> np.ndarray:
        """Tạo mask bảo vệ mặt CHÍNH XÁC (phía DƯỚI hairline)"""
        h, w = image.shape[:2]

        # 1. Khởi tạo mask TRẮNG toàn bộ (thay đổi toàn bộ)
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # 2. Lấy hairline chính xác
        hairline_y = self.get_forehead_position(image, face_info)

        # 3. Lấy landmarks mặt để tạo mask chính xác
        try:
            if hasattr(self, 'face_mesh'):
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    # Tạo mask mặt từ landmarks
                    face_points = []

                    # Lấy các landmarks xung quanh mặt (trừ tóc)
                    # Jawline, cheeks, chin, forehead (phần dưới hairline)
                    jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132]
                    cheek_indices = [116, 117, 118, 119, 100, 47, 126, 209, 49]

                    for idx in jaw_indices + cheek_indices:
                        if len(face_landmarks.landmark) > idx:
                            lm = face_landmarks.landmark[idx]
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            if py > hairline_y:  # Chỉ lấy points dưới hairline
                                face_points.append((px, py))

                    if len(face_points) > 3:
                        # Tạo convex hull từ các points
                        points_array = np.array(face_points, dtype=np.int32)
                        hull = cv2.convexHull(points_array)

                        # Vẽ convex hull = ĐEN (0) trên mask tạm
                        face_region = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(face_region, hull, 255)

                        # Trừ vùng mặt ra khỏi mask tổng
                        mask = cv2.subtract(mask, face_region)

                        print(f"✅ Created precise face mask from {len(face_points)} landmarks")

                        # Debug
                        if self.config.DEBUG_MODE:
                            cv2.imwrite(f'{self.config.DEBUG_DIR}/face_landmarks_mask.png', mask)

                        return mask
        except Exception as e:
            print(f"⚠️ Landmark-based face mask failed: {e}")

        # 4. Fallback: dùng phương pháp cũ
        print("⚠️ Using fallback face protection method")

        x1, y1, x2, y2 = face_info['bbox']
        face_width = x2 - x1
        face_height = y2 - y1

        # Tạo mask tạm cho vùng mặt
        face_region = np.zeros((h, w), dtype=np.uint8)

        # Vẽ hình oval bao quanh mặt (từ hairline_y xuống)
        center_x = (x1 + x2) // 2
        center_y = (hairline_y + y2) // 2
        axes_x = int(face_width * 0.6)
        axes_y = int((y2 - hairline_y) * 0.7)

        cv2.ellipse(
            face_region,
            (center_x, center_y),
            (axes_x, axes_y),
            0, 0, 360,
            255,
            -1
        )

        # Trừ vùng mặt
        mask = cv2.subtract(mask, face_region)

        # Làm mềm
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask
# class FaceDetectorMediaPipe:
#     """Face detection với MediaPipe - Nhanh và chính xác"""
#
#     def __init__(self, config: HairStyleConfig):
#         self.config = config
#         logger.info("Initializing MediaPipe Face Detection...")
#
#         # Khởi tạo MediaPipe
#         mp_face_detection = mp.solutions.face_detection
#         self.face_detection = mp_face_detection.FaceDetection(
#             model_selection=1,  # 1=full-range detection
#             min_detection_confidence=0.5
#         )
#
#         logger.info("✓ MediaPipe Face Detection initialized!")
#
#     def get_face_info(self, image: np.ndarray) -> Optional[Dict]:
#         """Get face bounding box"""
#         try:
#             # Convert BGR to RGB
#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             h, w = image.shape[:2]
#
#             # Detect face
#             results = self.face_detection.process(rgb_image)
#
#             if results.detections:
#                 # Lấy face có confidence cao nhất
#                 detection = results.detections[0]
#                 bbox = detection.location_data.relative_bounding_box
#
#                 # Convert relative to absolute coordinates
#                 x = int(bbox.xmin * w)
#                 y = int(bbox.ymin * h)
#                 width = int(bbox.width * w)
#                 height = int(bbox.height * h)
#
#                 bbox_abs = [
#                     max(0, x),
#                     max(0, y),
#                     min(w, x + width),
#                     min(h, y + height)
#                 ]
#
#                 return {
#                     'bbox': bbox_abs,
#                     'confidence': detection.score[0],
#                     'width': width,
#                     'height': height
#                 }
#
#             return None
#
#         except Exception as e:
#             logger.error(f"MediaPipe detection error: {e}")
#             return None
#
#     def get_forehead_position(self, image: np.ndarray, face_info: Dict) -> int:
#         """Lấy vị trí CHÂN MÀY (thay vì chân tóc)"""
#         h, w = image.shape[:2]
#
#         # Khởi tạo MediaPipe Face Mesh nếu chưa có
#         if not hasattr(self, 'face_mesh'):
#             import mediapipe as mp
#             mp_face_mesh = mp.solutions.face_mesh
#             self.face_mesh = mp_face_mesh.FaceMesh(
#                 static_image_mode=True,
#                 max_num_faces=1,
#                 refine_landmarks=True,
#                 min_detection_confidence=0.5
#             )
#
#         try:
#             # Convert BGR to RGB
#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Process với Face Mesh
#             results = self.face_mesh.process(rgb_image)
#
#             if results.multi_face_landmarks:
#                 face_landmarks = results.multi_face_landmarks[0]
#
#                 # Lấy các landmarks cho CHÂN MÀY (không phải chân tóc)
#                 # Landmark indices cho chân mày:
#                 # Chân mày phải: 55, 65, 52, 53, 46 (phần dưới)
#                 # Chân mày trái: 285, 295, 282, 283, 276 (phần dưới)
#
#                 eyebrow_points = []
#
#                 # Chân mày phải - phần dưới cùng
#                 right_eyebrow_indices = [55, 65, 52, 53, 46]
#                 for idx in right_eyebrow_indices:
#                     if len(face_landmarks.landmark) > idx:
#                         lm = face_landmarks.landmark[idx]
#                         px = int(lm.x * w)
#                         py = int(lm.y * h)
#                         eyebrow_points.append((px, py))
#                         print(f"🔍 Right eyebrow landmark {idx}: y={py}")
#
#                 # Chân mày trái - phần dưới cùng
#                 left_eyebrow_indices = [285, 295, 282, 283, 276]
#                 for idx in left_eyebrow_indices:
#                     if len(face_landmarks.landmark) > idx:
#                         lm = face_landmarks.landmark[idx]
#                         px = int(lm.x * w)
#                         py = int(lm.y * h)
#                         eyebrow_points.append((px, py))
#                         print(f"🔍 Left eyebrow landmark {idx}: y={py}")
#
#                 if eyebrow_points:
#                     # Lấy tọa độ y của tất cả điểm chân mày
#                     y_values = [p[1] for p in eyebrow_points]
#
#                     # Chọn điểm CAO NHẤT (y nhỏ nhất) làm đường chân mày
#                     # Nhưng thực tế chúng ta muốn lấy phần DƯỚI CÙNG của chân mày (y lớn nhất)
#                     eyebrow_bottom_y = max(y_values)
#
#                     # Thêm margin an toàn: xuống thêm 5% face height để chắc chắn bảo vệ toàn bộ mặt
#                     x1, y1, x2, y2 = face_info['bbox']
#                     face_height = y2 - y1
#                     eyebrow_line_y = min(h - 1, eyebrow_bottom_y + int(face_height * 0.05))
#
#                     print(f"✅ Calculated eyebrow_line_y: {eyebrow_line_y} (from {len(eyebrow_points)} eyebrow points)")
#
#                     # Vẽ debug
#                     if self.config.DEBUG_MODE:
#                         debug_img = image.copy()
#                         # Vẽ landmarks chân mày
#                         for px, py in eyebrow_points:
#                             cv2.circle(debug_img, (px, py), 3, (0, 255, 0), -1)
#                         # Vẽ đường chân mày
#                         cv2.line(debug_img, (0, eyebrow_line_y), (w, eyebrow_line_y), (0, 0, 255), 3)
#                         cv2.putText(debug_img, f'Eyebrow Line: y={eyebrow_line_y}',
#                                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                                     (0, 0, 255), 2)
#                         cv2.imwrite(f'{self.config.DEBUG_DIR}/eyebrow_po7sition.png', debug_img)
#
#                     return eyebrow_line_y
#
#             # Fallback: nếu không detect được chân mày
#             print("⚠️ Cannot detect eyebrow landmarks, using fallback method")
#             x1, y1, x2, y2 = face_info['bbox']
#             face_height = y2 - y1
#             # Chân mày thường ở khoảng 40-45% từ trên xuống
#             eyebrow_y = int(y1 + face_height * 0.45)
#             return eyebrow_y
#
#         except Exception as e:
#             print(f"❌ Error in eyebrow detection: {e}")
#             # Fallback
#             x1, y1, x2, y2 = face_info['bbox']
#             face_height = y2 - y1
#             return int(y1 + face_height * 0.45)
#
#     def create_face_protection_mask(
#             self,
#             image: np.ndarray,
#             face_info: Dict
#     ) -> np.ndarray:
#         """Tạo mask bảo vệ từ CHÂN MÀY trở XUỐNG"""
#         h, w = image.shape[:2]
#
#         # 1. Khởi tạo mask TRẮNG toàn bộ (thay đổi toàn bộ)
#         mask = np.ones((h, w), dtype=np.uint8) * 255
#
#         # 2. Lấy vị trí chân mày (gọi hàm get_forehead_position đã được sửa)
#         eyebrow_line_y = self.get_forehead_position(image, face_info)
#
#         # 3. Lấy landmarks mặt để tạo mask chính xác TỪ CHÂN MÀY XUỐNG
#         try:
#             if hasattr(self, 'face_mesh'):
#                 rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 results = self.face_mesh.process(rgb_image)
#
#                 if results.multi_face_landmarks:
#                     face_landmarks = results.multi_face_landmarks[0]
#
#                     # Tạo mask cho phần mặt TỪ CHÂN MÀY XUỐNG
#                     face_points = []
#
#                     # Lấy các landmarks xung quanh mặt CHỈ TỪ CHÂN MÀY XUỐNG
#                     jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
#                     cheek_indices = [116, 117, 118, 119, 100, 47, 126, 209, 49]
#                     mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
#                     nose_indices = [1, 2, 98, 327]
#                     eye_indices = [33, 133, 157, 158, 159, 160, 161, 173, 263, 362, 386, 387, 388, 389, 390, 466]
#
#                     all_face_indices = jaw_indices + cheek_indices + mouth_indices + nose_indices + eye_indices
#
#                     for idx in all_face_indices:
#                         if len(face_landmarks.landmark) > idx:
#                             lm = face_landmarks.landmark[idx]
#                             px = int(lm.x * w)
#                             py = int(lm.y * h)
#                             # CHỈ lấy points TỪ CHÂN MÀY XUỐNG (y >= eyebrow_line_y)
#                             if py >= eyebrow_line_y:
#                                 face_points.append((px, py))
#
#                     if len(face_points) > 3:
#                         # Tạo convex hull từ các points
#                         points_array = np.array(face_points, dtype=np.int32)
#                         hull = cv2.convexHull(points_array)
#
#                         # Vẽ convex hull = ĐEN (0) trên mask tạm
#                         face_region = np.zeros((h, w), dtype=np.uint8)
#                         cv2.fillConvexPoly(face_region, hull, 255)
#
#                         # QUAN TRỌNG: Xóa phần TRÊN chân mày khỏi face_region
#                         # Chỉ giữ lại phần từ chân mày xuống
#                         face_region[:eyebrow_line_y, :] = 0
#
#                         # Trừ vùng mặt ra khỏi mask tổng
#                         mask = cv2.subtract(mask, face_region)
#
#                         print(f"✅ Created face mask FROM EYEBROW DOWN using {len(face_points)} landmarks")
#                         print(f"   Protected area: y >= {eyebrow_line_y}")
#                         print(f"   Change area: y < {eyebrow_line_y} (forehead & hair)")
#
#                         # Debug
#                         if self.config.DEBUG_MODE:
#                             cv2.imwrite(f'{self.config.DEBUG_DIR}/face_mask_eyeb7ow_down.png', mask)
#
#                             # Tạo visualization
#                             debug_img = image.copy()
#                             mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#                             # Tô đỏ vùng được bảo vệ (mặt)
#                             protected_area = mask == 0
#                             debug_img[protected_area] = cv2.addWeighted(
#                                 debug_img[protected_area], 0.7,
#                                 np.array([0, 0, 255], dtype=np.uint8), 0.3, 0
#                             )[protected_area]
#
#                             # Vẽ đường chân mày
#                             cv2.line(debug_img, (0, eyebrow_line_y), (w, eyebrow_line_y),
#                                      (0, 255, 0), 2)
#                             cv2.putText(debug_img, 'EYEBROW LINE - PROTECT BELOW',
#                                         (10, eyebrow_line_y - 10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                             cv2.putText(debug_img, 'CHANGE ABOVE (Forehead & Hair)',
#                                         (10, 30),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#                             cv2.imwrite(f'{self.config.DEBUG_DIR}/protection_visualization.png', debug_img)
#
#                         return mask
#         except Exception as e:
#             print(f"⚠️ Landmark-based face mask failed: {e}")
#
#         # 4. Fallback: dùng phương pháp cơ bản bảo vệ từ chân mày xuống
#         print("⚠️ Using fallback face protection method (protect from eyebrow down)")
#
#         x1, y1, x2, y2 = face_info['bbox']
#         face_width = x2 - x1
#         face_height = y2 - y1
#
#         # Tạo mask tạm cho vùng mặt TỪ CHÂN MÀY XUỐNG
#         face_region = np.zeros((h, w), dtype=np.uint8)
#
#         # Vẽ hình oval bao quanh mặt (từ eyebrow_line_y xuống)
#         center_x = (x1 + x2) // 2
#         # Center y từ eyebrow_line_y đến cằm
#         center_y = (eyebrow_line_y + y2) // 2
#         axes_x = int(face_width * 0.5)
#         axes_y = int((y2 - eyebrow_line_y) * 0.6)
#
#         cv2.ellipse(
#             face_region,
#             (center_x, center_y),
#             (axes_x, axes_y),
#             0, 0, 360,
#             255,
#             -1
#         )
#
#         # QUAN TRỌNG: Xóa phần TRÊN chân mày
#         face_region[:eyebrow_line_y, :] = 0
#
#         # Trừ vùng mặt
#         mask = cv2.subtract(mask, face_region)
#
#         # Làm mềm viền
#         mask = cv2.GaussianBlur(mask, (21, 21), 10)
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
#         return mask

class HairMaskGenerator:
    """Tạo mask vùng tóc (phía trên trán)"""

    def __init__(self, config: HairStyleConfig):
        self.config = config

    def create_hair_mask(
            self,
            image: np.ndarray,
            face_info: Dict,
            hairline_y: int
    ) -> np.ndarray:
        """Tạo mask tóc CHÍNH XÁC theo đường chân tóc"""
        h, w = image.shape[:2]

        # 1. Khởi tạo mask đen
        mask = np.zeros((h, w), dtype=np.uint8)

        if face_info is None:
            # Fallback
            cv2.rectangle(mask, (0, 0), (w, h // 2), 255, -1)
            return mask

        # 2. Lấy thông tin mặt
        x1, y1, x2, y2 = face_info['bbox']
        face_width = x2 - x1
        face_height = y2 - y1

        print(f"🎯 Creating hair mask with hairline_y={hairline_y}")

        # 3. Tạo mask tóc theo hình dạng tự nhiên hơn
        # Vùng tóc chính: từ đầu ảnh đến hairline_y

        # Tạo polygon cho vùng tóc (hình oval hơn)
        points = []

        # Điểm trên cùng (giữa)
        points.append((w // 2, 0))

        # Điểm bên trái (trên)
        points.append((0, int(hairline_y * 0.3)))

        # Điểm bên trái (dưới)
        points.append((0, hairline_y))

        # Điểm dưới giữa (ở hairline)
        points.append((w // 2, hairline_y))

        # Điểm bên phải (dưới)
        points.append((w, hairline_y))

        # Điểm bên phải (trên)
        points.append((w, int(hairline_y * 0.3)))

        # Vẽ polygon
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # 4. Thêm vùng tóc hai bên (mái tóc)
        # Bên trái
        left_width = int(face_width * 0.4)
        left_x1 = max(0, x1 - left_width)
        left_x2 = x1
        left_y1 = max(0, hairline_y - int(face_height * 0.3))
        left_y2 = min(h, hairline_y + int(face_height * 0.2))
        cv2.rectangle(mask, (left_x1, left_y1), (left_x2, left_y2), 255, -1)

        # Bên phải
        right_x1 = x2
        right_x2 = min(w, x2 + left_width)
        cv2.rectangle(mask, (right_x1, left_y1), (right_x2, left_y2), 255, -1)

        # 5. Đảm bảo mask đủ lớn
        mask_ratio = np.sum(mask == 255) / mask.size
        print(f"🔍 Initial hair mask ratio: {mask_ratio:.2%}")

        if mask_ratio < 0.25:
            print("⚠️ Hair mask too small, expanding...")
            # Thêm vùng trên
            top_height = int(h * 0.35)
            mask[0:top_height, :] = 255

        # 6. Mở rộng mask nhẹ
        kernel_size = max(15, int(min(h, w) * 0.03))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 7. Làm mềm edges
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        final_ratio = np.sum(mask == 255) / mask.size
        print(f"✅ Final hair mask ratio: {final_ratio:.2%}")

        # Debug
        if self.config.DEBUG_MODE:
            overlay = image.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            cv2.line(overlay, (0, hairline_y), (w, hairline_y), (0, 0, 255), 3)
            cv2.imwrite(f'{self.config.DEBUG_DIR}/precise_hair_mask.png', mask)
            cv2.imwrite(f'{self.config.DEBUG_DIR}/precise_hair_overlay.png', overlay)

        return mask

class HairStyleGeneratorSD15:
    """SD1.5 Inpainting Generator - Chính"""

    def __init__(self, config: Optional[HairStyleConfig] = None):
        self.config = config or HairStyleConfig()
        self.device = self.config.DEVICE

        logger.info(f"🎯 Initializing SD1.5 Inpainting Generator")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🖼️  Image size: {self.config.IMAGE_SIZE}")

        # Initialize components
        self.face_detector = FaceDetectorMediaPipe(self.config)
        self.hair_mask_generator = HairMaskGenerator(self.config)
        self.pipe = None

        # Load models
        self._load_models()

        logger.info("✅ SD1.5 Generator initialized successfully!")

    def _load_models(self):
        """Load SD1.5 Inpainting pipeline"""
        logger.info("📦 Loading SD1.5 Inpainting model...")

        try:
            # Load pipeline
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.config.SD15_INPAINT_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.pipe = self.pipe.to(self.device)

            # Optimizations
            if self.device == "cuda":
                if self.config.USE_XFORMERS:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logger.info("✅ xFormers enabled")
                    except:
                        logger.warning("⚠️ xFormers not available")

                # Enable memory optimization
                self.pipe.enable_attention_slicing()

            # Scheduler - UniPC nhanh và ổn định
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            logger.info(f"✅ Model loaded: {self.config.SD15_INPAINT_MODEL}")

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def preprocess_image(
            self,
            image: np.ndarray
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """
        Preprocess ảnh và tạo mask
        Returns: (pil_image, pil_mask, processing_info)
        """
        target_w, target_h = self.config.IMAGE_SIZE  # 512x512
        h, w = image.shape[:2]

        # Resize với giữ tỉ lệ
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Pad về 512x512
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)  # White background
        )

        # Convert to PIL
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Face detection
        face_info = self.face_detector.get_face_info(padded)

        if face_info:
            logger.info(f"✅ Face detected: {face_info['bbox']}")

            # 1. Xác định đường chia tóc/mặt
            hairline_y = self.face_detector.get_forehead_position(padded, face_info)

            # 2. Tạo mask vùng tóc
            hair_mask = self.hair_mask_generator.create_hair_mask(
                padded, face_info, hairline_y
            )

            # 3. Tạo mask bảo vệ mặt
            face_mask = self.face_detector.create_face_protection_mask(
                padded, face_info
            )

            # 4. Kết hợp mask:
            # face_mask: mặt = 255 (trắng = bảo vệ), khác = 0 (đen)
            # hair_mask: tóc = 255 (trắng = thay đổi), khác = 0

            # Logic: Chỉ thay đổi vùng tóc KHÔNG overlap với mặt
            # final_mask = hair_mask AND (NOT face_mask)
            not_face_mask = cv2.bitwise_not(face_mask)  # Đảo ngược: mặt=0, khác=255
            final_mask = cv2.bitwise_and(hair_mask, not_face_mask)

            # Đảm bảo vùng mặt hoàn toàn đen (0)
            x1, y1, x2, y2 = face_info['bbox']
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(final_mask.shape[1], x2 + padding)
            y2 = min(final_mask.shape[0], y2 + padding)
            final_mask[y1:y2, x1:x2] = 0

            # Statistics
            mask_stats = {
                'hair_pixels': int(np.sum(hair_mask == 255)),
                'face_protected': int(np.sum(face_mask == 255)),
                'final_changed': int(np.sum(final_mask == 255)),
                'change_percentage': f"{np.sum(final_mask == 255) / final_mask.size * 100:.1f}%",
                'hairline_y': hairline_y,
            }

            logger.info(f"📊 Mask stats: {mask_stats['change_percentage']} changed")

        else:
            logger.warning("⚠️ No face detected, using fallback mask")
            # Fallback: mask nửa trên ảnh
            final_mask = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.uint8)
            cv2.rectangle(final_mask, (0, 0), (padded.shape[1], padded.shape[0] // 2), 255, -1)
            mask_stats = {}

        # Convert mask to PIL
        pil_mask = Image.fromarray(final_mask).convert("L")

        # Debug: save final mask
        if self.config.DEBUG_MODE:
            cv2.imwrite(f'{self.config.DEBUG_DIR}/final7_mask.png', final_mask)

        processing_info = {
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'padded_size': (target_w, target_h),
            'has_face': face_info is not None,
            'face_bbox': face_info['bbox'] if face_info else None,
            'mask_stats': mask_stats,
        }

        return pil_image, pil_mask, processing_info

    def get_style_prompt(
            self,
            style_name: str
    ) -> Tuple[str, str]:
        """Lấy prompt cho style từ config"""
        from app.config.hair_config import HairStylePrompts

        if style_name not in HairStylePrompts.HAIR_STYLES:
            raise ValueError(f"Style '{style_name}' not found")

        style_config = HairStylePrompts.HAIR_STYLES[style_name]

        # Simplify prompt cho SD1.5
        base_prompt = style_config['prompt']
        base_negative = style_config['negative']

        # Lấy từ khóa chính
        prompt_words = base_prompt.replace('hair transformation ONLY', '').replace('PRESERVE EVERYTHING ELSE EXACTLY',
                                                                                   '')
        prompt_words = prompt_words.split(',')

        # Giữ 3 phần đầu + face preservation
        simple_prompt = ', '.join(prompt_words[:3])
        simple_prompt += ", keep original face exactly the same, same person identity"

        # Negative prompt đơn giản
        simple_negative = "different face, changed face, blurry face, ugly face, deformed face"

        return simple_prompt, simple_negative

    def generate_single_style(
            self,
            image: np.ndarray,
            style_name: str,
            seed: Optional[int] = None,
            num_steps: Optional[int] = None,
            denoising_strength: Optional[float] = None,
            guidance_scale: Optional[float] = None
    ) -> Dict:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()
        logger.info(f"🎨 Generating: {style_name}")

        # 1. Preprocess
        pil_image, pil_mask, processing_info = self.preprocess_image(image)

        mask_array = np.array(pil_mask)
        mask_ratio = np.sum(mask_array == 255) / mask_array.size

        print("=" * 60)
        print(f"🔴 MASK RATIO: {mask_ratio:.2%}")
        print("=" * 60)

        # 2. AUTO-ADJUST STRENGTH BASED ON MASK SIZE - FIXED!
        if mask_ratio < 0.05:  # <5%
            strength = 0.98  # CỰC CAO
        elif mask_ratio < 0.1:  # 5-10%
            strength = 0.95
        elif mask_ratio < 0.2:  # 10-20%
            strength = 0.9
        elif mask_ratio < 0.3:  # 20-30%
            strength = 0.85
        else:
            strength = denoising_strength or self.config.DENOISING_STRENGTH

        print(f"🎯 AUTO-ADJUSTED STRENGTH: {strength}")

        # 3. EXPAND MASK IF TOO SMALL
        if mask_ratio < 0.2:  # Nếu mask < 20%
            print("⚠️ Mask too small, STRONGLY expanding...")

            # Dilate mạnh hơn
            kernel_size = max(30, int(min(mask_array.shape) * 0.1))  # 10% của kích thước
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Dilate nhiều lần
            expanded_mask = cv2.dilate(mask_array, kernel, iterations=3)

            # Thêm vùng lớn
            height, width = expanded_mask.shape
            expanded_mask[0:int(height * 0.4), :] = 255  # 40% trên cùng
            expanded_mask[:, 0:int(width * 0.2)] = 255  # 20% trái
            expanded_mask[:, int(width * 0.8):] = 255  # 20% phải

            # Update mask
            mask_array = expanded_mask
            pil_mask = Image.fromarray(mask_array).convert("L")

            # Tính lại ratio
            new_ratio = np.sum(mask_array == 255) / mask_array.size
            print(f"✅ STRONGLY Expanded mask: {mask_ratio:.2%} → {new_ratio:.2%}")
            mask_ratio = new_ratio

            # Lưu mask để debug


        # 4. SIMPLIFY PROMPT - QUAN TRỌNG!
        # Thay vì dùng prompt từ config, dùng prompt ĐƠN GIẢN
        prompt_map = {
            "blue_hair": "BLUE HAIR, vibrant blue color, colorful hairstyle",
            "man_bun": "MAN BUN, long hair tied up in bun, top knot",
            "short_undercut": "SHORT UNDERCUT, shaved sides, fade haircut",
            "slicked_back": "SLICKED BACK HAIR, smooth combed back style",
            "curly_afro": "CURLY AFRO, natural curls, textured hair",
            "korean_style": "KOREAN HAIRSTYLE, K-pop style, textured fringe",
            "side_part": "SIDE PART, neat combed hair, professional style",
            "bob_cut": "BOB CUT, shoulder length hair, feminine style",
            "pixie_cut": "PIXIE CUT, short layered hair, feminine crop",
            "buzz_cut": "BUZZ CUT, very short hair, military style"
        }

        base_prompt = prompt_map.get(style_name, f"{style_name} hairstyle")
        positive_prompt = f"{base_prompt}, keep face exactly the same, same person identity"
        negative_prompt = "different face, changed face, blurry face, ugly face"

        print(f"📝 SIMPLE PROMPT: {positive_prompt}")

        # 5. PARAMETERS - DÙNG strength ĐÃ TÍNH
        num_steps = num_steps or 40  # Tăng steps
        guidance = guidance_scale or 8.0  # Tăng guidance

        print(f"⚙️ FINAL PARAMS: strength={strength}, steps={num_steps}, guidance={guidance}")

        # 6. Seed
        if seed is None:
            seed = int(time.time() * 1000) % 1000000

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 7. GENERATE - DÙNG strength (không phải denoising)
        print("🚀 Running pipeline with STRONG settings...")

        output = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            strength=strength,  # QUAN TRỌNG: Dùng strength đã tính
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
        )

        result_image = output.images[0]

        # 8. DEBUG: Lưu ảnh để kiểm tra


        # 9. Crop về kích thước gốc
        if processing_info.get('padded_size'):
            result_array = np.array(result_image)
            padded_w, padded_h = processing_info['padded_size']
            new_w, new_h = processing_info['resized_size']
            pad_w = (padded_w - new_w) // 2
            pad_h = (padded_h - new_h) // 2

            cropped = result_array[
                      pad_h:pad_h + new_h,
                      pad_w:pad_w + new_w
                      ]
            result_image = Image.fromarray(cropped)

        elapsed_time = time.time() - start_time
        print(f"✅ Generated in {elapsed_time:.2f}s")
        print("=" * 60)

        return {
            'result': result_image,
            'mask': pil_mask,
            'processing_info': processing_info,
            'prompts': {
                'positive': positive_prompt,
                'negative': negative_prompt
            },
            'settings': {
                'seed': seed,
                'steps': num_steps,
                'strength': strength,  # Đổi từ denoising → strength
                'guidance': guidance,
                'mask_ratio': f"{mask_ratio:.1%}",
                'model': 'SD1.5'
            }
        }


# Singleton instance
_generator_instance = None


def get_hair_generator() -> HairStyleGeneratorSD15:
    """Get generator instance (singleton)"""
    global _generator_instance

    if _generator_instance is None:
        logger.info("Creating SD1.5 Hair Generator instance")
        _generator_instance = HairStyleGeneratorSD15()

    return _generator_instance


def cleanup_generator():
    """Cleanup generator và free memory"""
    global _generator_instance

    if _generator_instance is not None:
        del _generator_instance
        _generator_instance = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generator cleaned up")


"""
CÀI ĐẶT:
pip install mediapipe diffusers transformers accelerate torch torchvision

USAGE:
1. Basic usage:
   from hairstyle_service_sd15 import get_hair_generator
   import cv2

   generator = get_hair_generator()
   image = cv2.imread("input.jpg")

   result = generator.generate_single_style(
       image=image,
       style_name="man_bun",
       denoising_strength=0.8,
       seed=42
   )

   result['result'].save("output.jpg")

2. Check debug masks in ./debug_masks/:

   - final_7mask.png: Mask cuối cùng


3. Tips:
   - Nếu mask nhỏ (<10%): tăng denoising_strength lên 0.85
   - Nếu kết quả blurry: tăng num_inference_steps lên 40-50
   - Nếu không thay đổi: check debug masks xem có đúng không
"""

if __name__ == "__main__":
    # Test script
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        style = sys.argv[2] if len(sys.argv) > 2 else "man_bun"

        image = cv2.imread(image_path)
        if image is not None:
            generator = get_hair_generator()
            result = generator.generate_single_style(
                image=image,
                style_name=style,
                denoising_strength=0.8,
                seed=42
            )
            result['result'].save(f"output_{style}.jpg")
            print(f"✅ Saved to output_{style}.jpg")

            # Save mask
            result['mask'].save(f"mask_{style}.png")
            print(f"✅ Mask saved to mask_{style}.png")
        else:
            print(f"❌ Cannot read image: {image_path}")
    else:
        print("Usage: python hairstyle_service_sd15.py <image_path> [style_name]")


# """
# Hair Style Generation Service - SDXL + Inpainting với MediaPipe Face Detection
# FIXED: Lỗi FACE_PROTECTION_PADDING
# """
#
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from typing import List, Dict, Tuple, Optional, Union
# import logging
# import time
# import mediapipe as mp
#
# from diffusers import (
#     StableDiffusionXLInpaintPipeline,
#     AutoPipelineForInpainting,
#     UniPCMultistepScheduler
# )
#
# logger = logging.getLogger(__name__)
#
#
# class HairStyleConfig:
#     """Config cho SDXL + Inpainting với MediaPipe"""
#     SDXL_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     USE_XFORMERS = True
#
#     # Generation settings
#     NUM_INFERENCE_STEPS = 40
#     GUIDANCE_SCALE = 9.0
#     IMAGE_SIZE = (1024, 1024)
#
#     # ✅ CRITICAL: Denoising phải cao để thấy thay đổi tóc
#     DENOISING_STRENGTH = 0.8
#
#     MASK_DILATE = 20  # Tăng để mở rộng mask tóc
#     FACE_PROTECTION_PADDING = 0.2  # Padding quanh mặt để bảo vệ
#
#
# class FaceDetectorMediaPipe:
#     """Face detection với MediaPipe - NHANH & CHÍNH XÁC"""
#
#     def __init__(self, config: HairStyleConfig = None):
#         logger.info("Initializing MediaPipe Face Detection...")
#
#         # Lưu config
#         self.config = config or HairStyleConfig()
#
#         # Khởi tạo MediaPipe Face Detection
#         mp_face_detection = mp.solutions.face_detection
#         self.face_detection = mp_face_detection.FaceDetection(
#             model_selection=1,  # 0=short-range, 1=full-range
#             min_detection_confidence=0.5
#         )
#
#         # Khởi tạo MediaPipe Face Mesh (cho landmarks chính xác hơn)
#         mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )
#
#         logger.info("✓ MediaPipe Face Detection initialized!")
#
#     def get_face_info(self, image: np.ndarray) -> Optional[Dict]:
#         """Get face bounding box và landmarks với MediaPipe"""
#         try:
#             # Chuyển BGR → RGB
#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             h, w = image.shape[:2]
#
#             # 1. Face Detection để lấy bbox
#             detection_results = self.face_detection.process(rgb_image)
#
#             if detection_results.detections:
#                 # Lấy face có confidence cao nhất
#                 detection = detection_results.detections[0]
#                 bbox = detection.location_data.relative_bounding_box
#
#                 # Convert relative coordinates to absolute
#                 x = int(bbox.xmin * w)
#                 y = int(bbox.ymin * h)
#                 width = int(bbox.width * w)
#                 height = int(bbox.height * h)
#
#                 bbox_abs = [x, y, x + width, y + height]
#
#                 # 2. Face Mesh để lấy landmarks chính xác
#                 mesh_results = self.face_mesh.process(rgb_image)
#
#                 landmarks = []
#                 forehead_point = None
#
#                 if mesh_results.multi_face_landmarks:
#                     face_landmarks = mesh_results.multi_face_landmarks[0]
#
#                     # Extract landmarks (468 points)
#                     for landmark in face_landmarks.landmark:
#                         px = int(landmark.x * w)
#                         py = int(landmark.y * h)
#                         landmarks.append([px, py])
#
#                     # Tìm điểm trên trán (forehead)
#                     # Landmark 10: điểm giữa trán
#                     if len(landmarks) > 10:
#                         forehead_point = landmarks[10]
#
#                 return {
#                     'bbox': bbox_abs,
#                     'landmarks': landmarks,
#                     'forehead_point': forehead_point,
#                     'detection_confidence': detection.score[0]
#                 }
#
#             return None
#
#         except Exception as e:
#             logger.error(f"Error in MediaPipe face detection: {e}")
#             return None
#
#     def get_forehead_position(self, image: np.ndarray, face_info: Dict) -> Tuple[int, int]:
#         """Xác định vị trí đường ngang của trán (đường phân chia tóc/mặt)"""
#         h, w = image.shape[:2]
#         x1, y1, x2, y2 = face_info['bbox']
#         face_height = y2 - y1
#
#         # Chiến lược: lấy vị trí từ landmarks hoặc tính theo tỉ lệ
#         if face_info.get('forehead_point'):
#             # Dùng landmark trán
#             forehead_y = face_info['forehead_point'][1]
#         else:
#             # Ước tính: trán bắt đầu từ 1/3 trên của mặt
#             forehead_y = int(y1 + face_height * 0.2)
#
#         # Đường ngang chia tóc/mặt: lùi lên một chút từ trán
#         hairline_y = max(0, forehead_y - int(face_height * 0.1))
#
#         return hairline_y
#
#     def create_face_protection_mask(
#         self,
#         image: np.ndarray,
#         face_info: Dict
#     ) -> np.ndarray:
#         """Tạo mask bảo vệ khuôn mặt (từ trán xuống)"""
#         h, w = image.shape[:2]
#
#         # 1. Tạo mask đen (không bảo vệ gì)
#         mask = np.zeros((h, w), dtype=np.uint8)
#
#         # 2. Xác định vùng mặt cần bảo vệ
#         x1, y1, x2, y2 = face_info['bbox']
#
#         # ✅ FIX: Lấy padding từ config
#         if hasattr(self, 'config') and hasattr(self.config, 'FACE_PROTECTION_PADDING'):
#             padding = int((x2 - x1) * self.config.FACE_PROTECTION_PADDING)
#         else:
#             padding = int((x2 - x1) * 0.2)  # Fallback
#
#         x1_protect = max(0, x1 - padding)
#         y1_protect = max(0, y1 - padding // 2)  # Ít padding trên
#         x2_protect = min(w, x2 + padding)
#         y2_protect = min(h, y2 + padding)
#
#         # Xác định đường chia tóc/mặt
#         hairline_y = self.get_forehead_position(image, face_info)
#
#         # 3. Vẽ mask bảo vệ:
#         # - Mặt (từ hairline_y xuống): WHITE (255) = BẢO VỆ
#         # - Tóc (trên hairline_y): BLACK (0) = KHÔNG BẢO VỆ
#
#         # Vẽ hình oval bao quanh mặt (từ hairline_y xuống)
#         center_y = (hairline_y + y2_protect) // 2
#         center_x = (x1_protect + x2_protect) // 2
#         axes_x = (x2_protect - x1_protect) // 2
#         axes_y = (y2_protect - hairline_y) // 2
#
#         cv2.ellipse(
#             mask,
#             (center_x, center_y),
#             (axes_x, axes_y),
#             0, 0, 360,
#             255, -1
#         )
#
#         # Thêm phần dưới cằm (để chắc chắn)
#         chin_y_start = max(hairline_y, int((y1 + y2) // 2))
#         cv2.rectangle(
#             mask,
#             (x1_protect, chin_y_start),
#             (x2_protect, y2_protect),
#             255, -1
#         )
#
#         # Blur để làm mềm edges
#         mask = cv2.GaussianBlur(mask, (31, 31), 15)
#
#         # Threshold về binary
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
#         # DEBUG: Lưu mask
#         debug_dir = "./debug_masks"
#         import os
#         os.makedirs(debug_dir, exist_ok=True)
#         cv2.imwrite(f'{debug_dir}/debug_face_protection_mask1.png', mask)
#
#         return mask
#
#
# class HairMaskGenerator:
#     """Tạo mask vùng tóc (phía trên trán)"""
#
#     def __init__(self, config: HairStyleConfig):
#         self.config = config
#
#     def create_hair_mask(
#         self,
#         image: np.ndarray,
#         face_info: Dict,
#         face_detector: 'FaceDetectorMediaPipe'
#     ) -> np.ndarray:
#         """Tạo mask cho vùng tóc (phía TRÊN trán)"""
#         h, w = image.shape[:2]
#
#         # 1. Khởi tạo mask đen
#         mask = np.zeros((h, w), dtype=np.uint8)
#
#         if face_info is None:
#             # Không có face: mask toàn bộ nửa trên
#             cv2.rectangle(mask, (0, 0), (w, h // 2), 255, -1)
#             return mask
#
#         # 2. Xác định đường chia tóc/mặt
#         hairline_y = face_detector.get_forehead_position(image, face_info)
#
#         # 3. Tạo mask vùng tóc (phía TRÊN hairline_y)
#         x1, y1, x2, y2 = face_info['bbox']
#         face_width = x2 - x1
#
#         # Vùng tóc chính (phía trên trán)
#         hair_x1 = max(0, int(x1 - face_width * 0.5))
#         hair_x2 = min(w, int(x2 + face_width * 0.5))
#         hair_y1 = max(0, hairline_y - int(face_width * 0.8))  # Mở rộng lên trên
#         hair_y2 = hairline_y  # Dừng ở đường chia
#
#         cv2.rectangle(
#             mask,
#             (hair_x1, hair_y1),
#             (hair_x2, hair_y2),
#             255, -1
#         )
#
#         # 4. Thêm vùng tóc hai bên (cho tóc dài/phồng)
#         side_width = int(face_width * 0.4)
#         side_height = int(face_width * 0.8)
#
#         # Bên trái
#         left_x1 = max(0, x1 - side_width)
#         left_x2 = x1
#         left_y1 = max(0, hairline_y - side_height)
#         left_y2 = min(h, hairline_y + side_height // 2)
#         cv2.rectangle(mask, (left_x1, left_y1), (left_x2, left_y2), 255, -1)
#
#         # Bên phải
#         right_x1 = x2
#         right_x2 = min(w, x2 + side_width)
#         cv2.rectangle(mask, (right_x1, left_y1), (right_x2, left_y2), 255, -1)
#
#         # 5. Dilate để mở rộng mask
#         kernel = np.ones((self.config.MASK_DILATE, self.config.MASK_DILATE), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=2)
#
#         # 6. Blur để làm mềm edges
#         mask = cv2.GaussianBlur(mask, (21, 21), 11)
#
#         # 7. Threshold về binary
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#
#         # DEBUG: Lưu mask tóc
#         debug_dir = "./debug_masks"
#         import os
#         os.makedirs(debug_dir, exist_ok=True)
#         cv2.imwrite(f'{debug_dir}/debug_hair7_mask.png', mask)
#
#         # DEBUG: Tạo overlay visualization
#         overlay = image.copy()
#         overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5  # Xanh = tóc
#
#         # Vẽ đường chia
#         cv2.line(overlay, (0, hairline_y), (w, hairline_y), (0, 0, 255), 2)
#
#         # Vẽ bbox mặt
#         x1, y1, x2, y2 = face_info['bbox']
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#         cv2.imwrite(f'{debug_dir}/debug_hair_overl4ay.png', overlay)
#
#         print(f"✅ Hair mask: {np.sum(mask==255)} white pixels ({np.sum(mask==255)/mask.size*100:.1f}%)")
#         print(f"✅ Hairline y-position: {hairline_y}")
#
#         return mask
#
#
# class HairStyleGeneratorSDXLMediaPipe:
#     """SDXL Inpainting với MediaPipe Face Detection"""
#
#     def __init__(self, config: Optional[HairStyleConfig] = None):
#         self.config = config or HairStyleConfig()
#         self.device = self.config.DEVICE
#
#         logger.info(f"Initializing SDXL Inpainting with MediaPipe")
#         logger.info(f"Device: {self.device}")
#
#         # Khởi tạo MediaPipe face detector - TRUYỀN CONFIG VÀO
#         self.face_detector = FaceDetectorMediaPipe(self.config)
#         self.hair_mask_generator = HairMaskGenerator(self.config)
#         self.pipe = None
#
#         self._load_models()
#
#         logger.info("✓ SDXL + MediaPipe Generator initialized!")
#
#     def _load_models(self):
#         """Load SDXL Inpainting pipeline"""
#         logger.info("Loading SDXL Inpainting models...")
#
#         try:
#             logger.info(f"Loading {self.config.SDXL_INPAINT_MODEL}...")
#             self.pipe = AutoPipelineForInpainting.from_pretrained(
#                 self.config.SDXL_INPAINT_MODEL,
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 variant="fp16" if self.device == "cuda" else None,
#                 use_safetensors=True,
#             )
#
#             self.pipe = self.pipe.to(self.device)
#
#             # Optimizations
#             if self.device == "cuda":
#                 if self.config.USE_XFORMERS:
#                     try:
#                         self.pipe.enable_xformers_memory_efficient_attention()
#                         logger.info("✓ xFormers enabled")
#                     except:
#                         logger.warning("xFormers not available")
#
#             # Scheduler
#             self.pipe.scheduler = UniPCMultistepScheduler.from_config(
#                 self.pipe.scheduler.config
#             )
#
#             logger.info("✓ SDXL Inpainting pipeline loaded!")
#
#         except Exception as e:
#             logger.error(f"Error loading models: {e}")
#             self._load_fallback_models()
#
#     def _load_fallback_models(self):
#         """Fallback model"""
#         try:
#             self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#                 "stabilityai/stable-diffusion-xl-base-1.0",
#                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#                 variant="fp16" if self.device == "cuda" else None,
#                 use_safetensors=True,
#             )
#             self.pipe = self.pipe.to(self.device)
#             logger.info("✓ SDXL base model loaded as fallback")
#         except Exception as e:
#             logger.error(f"Fallback also failed: {e}")
#             raise
#
#     def preprocess_image_and_mask(
#         self,
#         image: np.ndarray
#     ) -> Tuple[Image.Image, Image.Image, Dict]:
#         """Preprocess ảnh và tạo mask với MediaPipe"""
#         target_w, target_h = self.config.IMAGE_SIZE
#         h, w = image.shape[:2]
#
#         # Resize giữ tỉ lệ
#         scale = min(target_w / w, target_h / h)
#         new_w, new_h = int(w * scale), int(h * scale)
#
#         resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
#
#         # Pad
#         pad_w = (target_w - new_w) // 2
#         pad_h = (target_h - new_h) // 2
#
#         padded = cv2.copyMakeBorder(
#             resized,
#             pad_h, target_h - new_h - pad_h,
#             pad_w, target_w - new_w - pad_w,
#             cv2.BORDER_CONSTANT,
#             value=(255, 255, 255)
#         )
#
#         # Convert to PIL
#         rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(rgb_image)
#
#         # 1. Face detection với MediaPipe
#         face_info = self.face_detector.get_face_info(padded)
#
#         # 2. Tạo masks
#         if face_info:
#             print(f"✅ Face detected: {face_info['bbox']}")
#
#             # A. Mask bảo vệ khuôn mặt (từ trán xuống)
#             face_protection_mask = self.face_detector.create_face_protection_mask(
#                 padded, face_info
#             )
#
#             # B. Mask vùng tóc (phía trên trán)
#             hair_mask = self.hair_mask_generator.create_hair_mask(
#                 padded, face_info, self.face_detector
#             )
#
#             # 3. Kết hợp masks:
#             # Logic: Chỉ thay đổi vùng tóc (hair_mask) VÀ KHÔNG phải vùng mặt (invert face_protection_mask)
#
#             # Invert face protection mask: mặt = 0, khác = 255
#             not_face_mask = face_protection_mask  # Đã là mặt=0, khác=255
#
#             # Final mask = hair_mask AND not_face_mask
#             final_mask = cv2.bitwise_and(hair_mask, not_face_mask)
#
#             # DEBUG: Lưu tất cả masks
#             debug_dir = "./debug_masks"
#             import os
#             os.makedirs(debug_dir, exist_ok=True)
#             cv2.imwrite(f'{debug_dir}/debug_combined_mask5.png', final_mask)
#
#             # 4. Đảm bảo vùng mặt hoàn toàn đen (0)
#             x1, y1, x2, y2 = face_info['bbox']
#             # Mở rộng một chút để chắc chắn
#             padding = 10
#             x1 = max(0, x1 - padding)
#             y1 = max(0, y1 - padding)
#             x2 = min(final_mask.shape[1], x2 + padding)
#             y2 = min(final_mask.shape[0], y2 + padding)
#             final_mask[y1:y2, x1:x2] = 0
#
#             # 5. Statistics
#             mask_stats = {
#                 'hair_pixels': int(np.sum(hair_mask == 255)),
#                 'face_protected_pixels': int(np.sum(face_protection_mask == 255)),
#                 'final_changed_pixels': int(np.sum(final_mask == 255)),
#                 'hair_percentage': f"{np.sum(hair_mask == 255)/hair_mask.size*100:.1f}%",
#                 'change_percentage': f"{np.sum(final_mask == 255)/final_mask.size*100:.1f}%",
#             }
#
#             print(f"📊 Mask Statistics:")
#             for key, value in mask_stats.items():
#                 print(f"  {key}: {value}")
#
#         else:
#             print("⚠️ No face detected, using fallback mask")
#             # Không có face: mask nửa trên ảnh
#             final_mask = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.uint8)
#             cv2.rectangle(final_mask, (0, 0), (padded.shape[1], padded.shape[0] // 2), 255, -1)
#             mask_stats = {}
#
#         # Convert mask to PIL
#         pil_mask = Image.fromarray(final_mask).convert("L")
#
#         processing_info = {
#             'original_size': (w, h),
#             'resized_size': (new_w, new_h),
#             'padded_size': (target_w, target_h),
#             'has_face': face_info is not None,
#             'face_bbox': face_info['bbox'] if face_info else None,
#             'mask_stats': mask_stats if face_info else {}
#         }
#
#         return pil_image, pil_mask, processing_info
#
#     def enhance_prompt_for_inpainting(
#         self,
#         base_prompt: str,
#         face_info: Optional[Dict] = None
#     ) -> Tuple[str, str]:
#         """Enhanced prompts với nhấn mạnh bảo vệ mặt"""
#
#         gender = "person"
#         if face_info:
#             # MediaPipe không có gender detection, có thể thêm nếu cần
#             gender = "person"
#
#         positive = f"{base_prompt}, "
#         positive += "ORIGINAL FACE PRESERVED COMPLETELY, same person identity, "
#         positive += "facial features unchanged, only hairstyle modification, "
#         positive += "hair transformation above forehead, "
#         positive += "realistic hair texture, detailed hairstyle, "
#         positive += "professional haircut, 8k uhd, sharp focus, best quality"
#
#         negative = "different face, changed face, altered face, new person, "
#         negative += "face swap, identity change, distorted face, "
#         negative += "blurry face, ugly face, bad anatomy, "
#         negative += "unnatural hair, wig, fake hair, hairpiece, "
#         negative += "low quality, artifacts, watermark, signature"
#
#         return positive, negative
#
#     def generate_single_style(
#         self,
#         image: np.ndarray,
#         style_name: str,
#         seed: Optional[int] = None,
#         num_steps: Optional[int] = None,
#         denoising_strength: Optional[float] = None,
#         use_mask: bool = True
#     ) -> Dict:
#         """Generate với MediaPipe face preservation"""
#
#         # Clear cache
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         logger.info(f"🎯 Generating: {style_name}")
#         start_time = time.time()
#
#         # Get style config
#         from app.config.hair_config import HairStylePrompts as StyleConfig
#
#         if style_name not in StyleConfig.HAIR_STYLES:
#             raise ValueError(f"Unknown style: {style_name}")
#
#         style_config = StyleConfig.HAIR_STYLES[style_name]
#
#         # Preprocess ảnh và mask
#         pil_image, pil_mask, processing_info = self.preprocess_image_and_mask(image)
#
#         # Enhanced prompts
#         base_prompt = style_config['prompt']
#         positive_prompt, negative_prompt = self.enhance_prompt_for_inpainting(
#             base_prompt,
#             processing_info['face_bbox']
#         )
#
#         # Seed
#         if seed is None:
#             seed = int(time.time() * 1000) % 1000000
#
#         generator = torch.Generator(device=self.device).manual_seed(seed)
#
#         # Parameters
#         num_steps = num_steps or self.config.NUM_INFERENCE_STEPS
#         denoising = denoising_strength or self.config.DENOISING_STRENGTH
#
#         # Đảm bảo denoising đủ cao
#         if denoising < 0.6:
#             denoising = 0.7
#             logger.info(f"⚠️ Adjusted denoising to {denoising}")
#
#         logger.info(f"⚙️ Settings: steps={num_steps}, denoising={denoising}")
#         logger.info(f"📝 Prompt: {positive_prompt[:80]}...")
#
#         # Generate
#         if use_mask:
#             output = self.pipe(
#                 prompt=positive_prompt,
#                 negative_prompt=negative_prompt,
#                 image=pil_image,
#                 mask_image=pil_mask,
#                 num_inference_steps=num_steps,
#                 guidance_scale=self.config.GUIDANCE_SCALE,
#                 strength=denoising,
#                 generator=generator,
#             )
#         else:
#             output = self.pipe(
#                 prompt=positive_prompt,
#                 negative_prompt=negative_prompt,
#                 image=pil_image,
#                 num_inference_steps=num_steps,
#                 guidance_scale=self.config.GUIDANCE_SCALE,
#                 strength=denoising,
#                 generator=generator,
#             )
#
#         result_image = output.images[0]
#
#         # OPTIONAL: Manual face restoration để chắc chắn
#         if processing_info['has_face'] and use_mask:
#             # Có thể thêm logic paste lại mặt gốc nếu cần
#             pass
#
#         # Crop về kích thước gốc
#         if processing_info.get('padded_size'):
#             result_array = np.array(result_image)
#             padded_w, padded_h = processing_info['padded_size']
#             new_w, new_h = processing_info['resized_size']
#             pad_w = (padded_w - new_w) // 2
#             pad_h = (padded_h - new_h) // 2
#
#             cropped = result_array[
#                 pad_h:pad_h + new_h,
#                 pad_w:pad_w + new_w
#             ]
#             result_image = Image.fromarray(cropped)
#
#         elapsed_time = time.time() - start_time
#         logger.info(f"✅ Done in {elapsed_time:.2f}s")
#
#         return {
#             'result': result_image,
#             'mask': pil_mask,
#             'processing_info': processing_info,
#             'prompts': {
#                 'positive': positive_prompt,
#                 'negative': negative_prompt
#             },
#             'settings': {
#                 'seed': seed,
#                 'steps': num_steps,
#                 'denoising': denoising
#             }
#         }
#
#
# # Singleton
# _generator_instance = None
#
#
# def get_hair_generator() -> HairStyleGeneratorSDXLMediaPipe:
#     """Get generator (singleton)"""
#     global _generator_instance
#
#     if _generator_instance is None:
#         logger.info("Creating SDXL + MediaPipe Hair Generator")
#         _generator_instance = HairStyleGeneratorSDXLMediaPipe()
#
#     return _generator_instance
#
#
# def cleanup_generator():
#     """Cleanup"""
#     global _generator_instance
#
#     if _generator_instance is not None:
#         del _generator_instance
#         _generator_instance = None
#
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#
# """
# CÀI ĐẶT:
# pip install mediapipe opencv-python
#
# USAGE:
# from hairstyle_service_sdxl_mediapipe import get_hair_generator
# import cv2
#
# generator = get_hair_generator()
# image = cv2.imread("input.jpg")
#
# result = generator.generate_single_style(
#     image=image,
#     style_name="man_bun",
#     denoising_strength=0.75,
#     seed=None
# )
#
# result['result'].save("output.jpg")
# result['mask'].save("mask.png")
#
# # Check debug images:
# # - debug_masks/debug_hair_ma7sk.png: Mask vùng tóc
# # - debug_masks/debug_face_protection_7mask.png: Mask bảo vệ mặt
# # - debug_masks/debug_combined_m9ask.png: Mask cuối cùng
# # - debug_masks/debug_hair_overla7y.png: Visualization overlay
# """