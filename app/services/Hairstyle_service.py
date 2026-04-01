"""
Hair Style Generation Service - SD1.5 + Inpainting với MediaPipe
PRODUCTION VERSION - Removed debug code
"""
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
import time
import mediapipe as mp

from diffusers import (
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)


logger = logging.getLogger(__name__)


class HairStyleConfig:
    """Config cho SD1.5 + Inpainting"""
    SD15_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_XFORMERS = True if DEVICE == "cuda" else False
    IMAGE_SIZE = (512, 512)
    NUM_INFERENCE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    DENOISING_STRENGTH = 0.75
    #tạo khu vực gen
    MASK_DILATE = 10
    #tạo khu vực bảo vệ
    FACE_PROTECTION_PADDING = 0.25


class FaceDetectorMediaPipe:
    """Face detection với MediaPipe"""

    def __init__(self, config: HairStyleConfig):
        self.config = config
        logger.info("Initializing MediaPipe Face Detection...")

        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        logger.info("✓ MediaPipe Face Detection initialized!")

    def get_face_info(self, image: np.ndarray) -> Optional[Dict]:
        """Get face bounding box"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            results = self.face_detection.process(rgb_image)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

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
        """Lấy đường chân tóc bằng MediaPipe Face Mesh"""
        h, w = image.shape[:2]

        if not hasattr(self, 'face_mesh'):
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                forehead_points = []

                # Landmark 151: mid forehead
                if len(face_landmarks.landmark) > 151:
                    lm = face_landmarks.landmark[151]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    forehead_points.append((px, py))

                # Landmark 10: forehead top
                if len(face_landmarks.landmark) > 10:
                    lm = face_landmarks.landmark[10]
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    forehead_points.append((px, py))

                if forehead_points:
                    y_values = [p[1] for p in forehead_points]
                    hairline_y = min(y_values)

                    x1, y1, x2, y2 = face_info['bbox']
                    face_height = y2 - y1
                    hairline_y = max(0, hairline_y - int(face_height * 0.05))

                    return hairline_y

            # Fallback
            x1, y1, x2, y2 = face_info['bbox']
            face_height = y2 - y1
            return int(y1 + face_height * 0.33)

        except Exception as e:
            logger.error(f"Error in hairline detection: {e}")
            x1, y1, x2, y2 = face_info['bbox']
            face_height = y2 - y1
            return int(y1 + face_height * 0.33)

    def create_face_protection_mask(
            self,
            image: np.ndarray,
            face_info: Dict
    ) -> np.ndarray:
        """Tạo mask bảo vệ mặt"""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        hairline_y = self.get_forehead_position(image, face_info)

        try:
            if hasattr(self, 'face_mesh'):
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_image)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    face_points = []

                    jaw_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132]
                    cheek_indices = [116, 117, 118, 119, 100, 47, 126, 209, 49]

                    for idx in jaw_indices + cheek_indices:
                        if len(face_landmarks.landmark) > idx:
                            lm = face_landmarks.landmark[idx]
                            px = int(lm.x * w)
                            py = int(lm.y * h)
                            if py > hairline_y:
                                face_points.append((px, py))

                    if len(face_points) > 3:
                        points_array = np.array(face_points, dtype=np.int32)
                        hull = cv2.convexHull(points_array)
                        face_region = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(face_region, hull, 255)
                        mask = cv2.subtract(mask, face_region)
                        return mask

        except Exception as e:
            logger.warning(f"Landmark-based face mask failed: {e}")

        # Fallback
        x1, y1, x2, y2 = face_info['bbox']
        face_width = x2 - x1
        face_region = np.zeros((h, w), dtype=np.uint8)

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

        mask = cv2.subtract(mask, face_region)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask


class HairMaskGenerator:
    """Tạo mask vùng tóc"""

    def __init__(self, config: HairStyleConfig):
        self.config = config

    def create_hair_mask(
            self,
            image: np.ndarray,
            face_info: Dict,
            hairline_y: int
    ) -> np.ndarray:
        """Tạo mask tóc theo đường chân tóc"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if face_info is None:
            cv2.rectangle(mask, (0, 0), (w, h // 2), 255, -1)
            return mask

        x1, y1, x2, y2 = face_info['bbox']
        face_width = x2 - x1
        face_height = y2 - y1

        # Tạo polygon vùng tóc
        points = [
            (w // 2, 0),
            (0, int(hairline_y * 0.3)),
            (0, hairline_y),
            (w // 2, hairline_y),
            (w, hairline_y),
            (w, int(hairline_y * 0.3))
        ]

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Thêm vùng tóc hai bên
        left_width = int(face_width * 0.4)
        left_x1 = max(0, x1 - left_width)
        left_x2 = x1
        left_y1 = max(0, hairline_y - int(face_height * 0.3))
        left_y2 = min(h, hairline_y + int(face_height * 0.2))
        cv2.rectangle(mask, (left_x1, left_y1), (left_x2, left_y2), 255, -1)

        right_x1 = x2
        right_x2 = min(w, x2 + left_width)
        cv2.rectangle(mask, (right_x1, left_y1), (right_x2, left_y2), 255, -1)

        # Mở rộng mask nếu quá nhỏ
        mask_ratio = np.sum(mask == 255) / mask.size
        if mask_ratio < 0.25:
            top_height = int(h * 0.35)
            mask[0:top_height, :] = 255

        # Mở rộng và làm mềm
        kernel_size = max(15, int(min(h, w) * 0.03))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask


class HairStyleGeneratorSD15:
    """SD1.5 Inpainting Generator"""

    def __init__(self, config: Optional[HairStyleConfig] = None):
        self.config = config or HairStyleConfig()
        self.device = self.config.DEVICE

        logger.info(f"🎯 Initializing SD1.5 Inpainting Generator")
        logger.info(f"📱 Device: {self.device}")

        self.face_detector = FaceDetectorMediaPipe(self.config)
        self.hair_mask_generator = HairMaskGenerator(self.config)
        self.pipe = None

        self._load_models()
        logger.info("SD1.5 Generator initialized successfully!")

    def _load_models(self):
        """Load SD1.5 Inpainting pipeline"""
        logger.info("📦 Loading SD1.5 Inpainting model...")

        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.config.SD15_INPAINT_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

            self.pipe = self.pipe.to(self.device)

            if self.device == "cuda":
                if self.config.USE_XFORMERS:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logger.info("xFormers enabled")
                    except:
                        logger.warning("xFormers not available")

                self.pipe.enable_attention_slicing()

            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            logger.info(f"Model loaded: {self.config.SD15_INPAINT_MODEL}")

        except Exception as e:
            logger.error(f" Error loading model: {e}")
            raise

    def preprocess_image(
            self,
            image: np.ndarray
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """Preprocess ảnh và tạo mask"""
        target_w, target_h = self.config.IMAGE_SIZE
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
            value=(255, 255, 255)
        )

        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Face detection và tạo mask
        face_info = self.face_detector.get_face_info(padded)

        if face_info:
            logger.info(f"Face detected: {face_info['bbox']}")

            hairline_y = self.face_detector.get_forehead_position(padded, face_info)
            hair_mask = self.hair_mask_generator.create_hair_mask(
                padded, face_info, hairline_y
            )
            face_mask = self.face_detector.create_face_protection_mask(
                padded, face_info
            )

            not_face_mask = cv2.bitwise_not(face_mask)
            final_mask = cv2.bitwise_and(hair_mask, not_face_mask)

            # Bảo vệ vùng mặt
            x1, y1, x2, y2 = face_info['bbox']
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(final_mask.shape[1], x2 + padding)
            y2 = min(final_mask.shape[0], y2 + padding)
            final_mask[y1:y2, x1:x2] = 0

            mask_stats = {
                'final_changed': int(np.sum(final_mask == 255)),
                'change_percentage': f"{np.sum(final_mask == 255) / final_mask.size * 100:.1f}%",
            }
        else:
            logger.warning(" No face detected, using fallback mask")
            final_mask = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.uint8)
            cv2.rectangle(final_mask, (0, 0), (padded.shape[1], padded.shape[0] // 2), 255, -1)
            mask_stats = {}

        pil_mask = Image.fromarray(final_mask).convert("L")

        processing_info = {
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'padded_size': (target_w, target_h),
            'has_face': face_info is not None,
            'face_bbox': face_info['bbox'] if face_info else None,
            'mask_stats': mask_stats,
        }

        return pil_image, pil_mask, processing_info

    def generate_single_style(
            self,
            image: np.ndarray,
            style_name: str,
            seed: Optional[int] = None,
            num_steps: Optional[int] = None,
            denoising_strength: Optional[float] = None,
            guidance_scale: Optional[float] = None
    ) -> Dict:
        """Generate hairstyle - Main function"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_time = time.time()
        logger.info(f" Generating: {style_name}")

        # Preprocess
        pil_image, pil_mask, processing_info = self.preprocess_image(image)

        mask_array = np.array(pil_mask)
        mask_ratio = np.sum(mask_array == 255) / mask_array.size

        # Auto-adjust strength based on mask size
        if mask_ratio < 0.05:
            strength = 0.98
        elif mask_ratio < 0.1:
            strength = 0.95
        elif mask_ratio < 0.2:
            strength = 0.9
        elif mask_ratio < 0.3:
            strength = 0.85
        else:
            strength = denoising_strength or self.config.DENOISING_STRENGTH

        # Expand mask if too small
        if mask_ratio < 0.2:
            kernel_size = max(30, int(min(mask_array.shape) * 0.1))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            expanded_mask = cv2.dilate(mask_array, kernel, iterations=3)

            height, width = expanded_mask.shape
            expanded_mask[0:int(height * 0.4), :] = 255
            expanded_mask[:, 0:int(width * 0.2)] = 255
            expanded_mask[:, int(width * 0.8):] = 255

            mask_array = expanded_mask
            pil_mask = Image.fromarray(mask_array).convert("L")
            mask_ratio = np.sum(mask_array == 255) / mask_array.size

        # Simple prompts
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

        # Parameters
        num_steps = num_steps or 40
        guidance = guidance_scale or 8.0

        # Seed
        if seed is None:
            seed = int(time.time() * 1000) % 1000000

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        output = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=generator,
        )

        result_image = output.images[0]

        # Crop về kích thước gốc
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
        logger.info(f"Generated in {elapsed_time:.2f}s")

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
                'strength': strength,
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