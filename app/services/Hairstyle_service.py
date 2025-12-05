"""
Hair Style Generation Service
File: app/services/hairstyle_service.py
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import insightface
from insightface.app import FaceAnalysis
import logging

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image

# Import config (adjust path based on your structure)
# from app.core.config import HairStyleConfig

logger = logging.getLogger(__name__)


class HairStyleConfig:
    """Inline config - copy from hair_config.py hoặc import"""
    SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_FACEID = "lllyasviel/control_v11p_sd15_canny"  # Canny cho face structure
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_XFORMERS = True
    NUM_INFERENCE_STEPS = 25
    GUIDANCE_SCALE = 7.5
    IMAGE_SIZE = (512, 512)
    CONTROLNET_CONDITIONING_SCALE = 0.8
    FACEID_MODEL = "buffalo_l"


class FaceIDExtractor:
    """Extract face embeddings using InsightFace for face preservation"""

    def __init__(self, model_name: str = "buffalo_l"):
        logger.info(f"Initializing FaceID Extractor with model: {model_name}")

        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("FaceID Extractor initialized successfully")

    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image

        Args:
            image: numpy array (BGR from OpenCV)

        Returns:
            Face embedding or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.app.get(rgb_image)

        if len(faces) == 0:
            logger.warning("No face detected in image")
            return None

        # Get largest face
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        return face.embedding

    def get_face_info(self, image: np.ndarray) -> Optional[Dict]:
        """Get detailed face information including landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb_image)

        if len(faces) == 0:
            return None

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        return {
            'bbox': face.bbox.tolist(),
            'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
            'embedding': face.embedding,
            'age': face.age if hasattr(face, 'age') else None,
            'gender': face.gender if hasattr(face, 'gender') else None,
        }

    def extract_face_region(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Extract and crop face region"""
        face_info = self.get_face_info(image)

        if face_info is None:
            raise ValueError("No face detected")

        bbox = face_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Add padding
        h, w = image.shape[:2]
        padding = 50
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        face_crop = image[y1:y2, x1:x2]

        return face_crop, face_info


class HairStyleGenerator:
    """
    Generate hair styles using Stable Diffusion 1.5 + ControlNet
    """

    def __init__(self, config: Optional[HairStyleConfig] = None):
        self.config = config or HairStyleConfig()
        self.device = self.config.DEVICE

        logger.info(f"Initializing Hair Style Generator on {self.device}")

        # Initialize components
        self.face_extractor = FaceIDExtractor(self.config.FACEID_MODEL)
        self.pipe = None

        # Load models
        self._load_models()

        logger.info("Hair Style Generator initialized successfully!")

    def _load_models(self):
        """Load SD1.5 and ControlNet models"""
        logger.info("Loading Stable Diffusion 1.5 and ControlNet models...")

        try:
            # Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                self.config.CONTROLNET_FACEID,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            # Load SD1.5 Pipeline with ControlNet
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.config.SD15_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for speed
            )

            # Move to device
            self.pipe = self.pipe.to(self.device)

            # Optimizations
            if self.device == "cuda":
                # Enable memory efficient attention
                if self.config.USE_XFORMERS:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        logger.info("xFormers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable xFormers: {e}")

                # Enable attention slicing for lower VRAM
                self.pipe.enable_attention_slicing(1)

                # Enable VAE slicing
                self.pipe.enable_vae_slicing()

            # Use faster scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            logger.info("Models loaded and optimized successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def preprocess_image(
            self,
            image: np.ndarray
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess image for ControlNet

        Returns:
            (original_pil_image, control_image)
        """
        # Resize to target size maintaining aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = self.config.IMAGE_SIZE

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_h - new_h - pad_h,
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        # Convert to RGB PIL
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Create Canny edge for ControlNet
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)

        # Enhance face edges
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to make edges more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert to 3 channel
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(edges_rgb)

        return pil_image, control_image

    def generate_single_style(
            self,
            image: np.ndarray,
            style_name: str,
            seed: Optional[int] = None,
            num_steps: Optional[int] = None
    ) -> Image.Image:
        """
        Generate a single hair style

        Args:
            image: Input image (numpy array, BGR)
            style_name: Name of hair style
            seed: Random seed for reproducibility
            num_steps: Number of inference steps (override config)

        Returns:
            Generated PIL Image
        """
        logger.info(f"Generating hair style: {style_name}")

        # Check face
        face_embedding = self.face_extractor.extract_face_embedding(image)
        if face_embedding is None:
            raise ValueError("No face detected in image")

        # Preprocess
        pil_image, control_image = self.preprocess_image(image)

        # Get style config
        from app.core.config import HairStyleConfig as StyleConfig

        if style_name not in StyleConfig.HAIR_STYLES:
            raise ValueError(f"Unknown style: {style_name}")

        style_config = StyleConfig.HAIR_STYLES[style_name]

        # Build prompts
        positive_prompt = (
            f"{style_config['prompt']}, "
            f"portrait photograph, face focus, "
            f"{StyleConfig.QUALITY_POSITIVE}"
        )

        negative_prompt = (
            f"{style_config['negative']}, "
            f"{StyleConfig.QUALITY_NEGATIVE}"
        )

        # Set seed
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        num_steps = num_steps or self.config.NUM_INFERENCE_STEPS

        logger.info(f"Generating with {num_steps} steps, seed: {seed}")

        output = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_steps,
            guidance_scale=self.config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=self.config.CONTROLNET_CONDITIONING_SCALE,
            generator=generator,
        )

        logger.info("Generation completed successfully")

        return output.images[0]

    def generate_multiple_styles(
            self,
            image: np.ndarray,
            style_names: List[str],
            num_variations: int = 1
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate multiple hair styles

        Args:
            image: Input image
            style_names: List of style names to generate
            num_variations: Number of variations per style

        Returns:
            Dictionary mapping style name to list of images
        """
        logger.info(f"Generating {len(style_names)} styles with {num_variations} variations each")

        results = {}

        for style_name in style_names:
            logger.info(f"Processing style: {style_name}")

            style_images = []
            for i in range(num_variations):
                try:
                    img = self.generate_single_style(
                        image,
                        style_name,
                        seed=None  # Random seed for each variation
                    )
                    style_images.append(img)
                    logger.info(f"  ✓ Variation {i + 1}/{num_variations} completed")

                except Exception as e:
                    logger.error(f"  ✗ Error generating variation {i + 1}: {e}")

            results[style_name] = style_images

        logger.info(f"Generation complete! Generated {len(results)} styles")
        return results

    def batch_generate(
            self,
            image: np.ndarray,
            num_styles: int = 4
    ) -> List[Tuple[str, Image.Image]]:
        """
        Quick batch generation with random styles

        Returns:
            List of (style_name, image) tuples
        """
        import random
        from app.core.config import HairStyleConfig as StyleConfig

        all_styles = StyleConfig.get_style_list()
        selected_styles = random.sample(all_styles, min(num_styles, len(all_styles)))

        results = []
        for style_name in selected_styles:
            try:
                img = self.generate_single_style(image, style_name)
                results.append((style_name, img))
            except Exception as e:
                logger.error(f"Error generating {style_name}: {e}")

        return results


# Singleton pattern
_generator_instance = None


def get_hair_generator() -> HairStyleGenerator:
    """Get or create Hair Style Generator instance (singleton)"""
    global _generator_instance

    if _generator_instance is None:
        logger.info("Creating new Hair Style Generator instance")
        _generator_instance = HairStyleGenerator()

    return _generator_instance


def cleanup_generator():
    """Cleanup generator and free memory"""
    global _generator_instance

    if _generator_instance is not None:
        logger.info("Cleaning up Hair Style Generator")
        del _generator_instance
        _generator_instance = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()