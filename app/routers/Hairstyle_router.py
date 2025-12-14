"""
Hair Style API Router
File: app/routers/hairstyle_router.py
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from typing import List, Optional, Dict
import cv2
import numpy as np
from PIL import Image
import io
import uuid
from pathlib import Path
from datetime import datetime
import logging
import zipfile
import json
import base64

# Adjust imports based on your project structure
# from app.services.hairstyle_service_sdxl_inpainting import get_hair_generator
# from app.config.hair_config import HairStylePrompts

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/hairstyle",
    tags=["Hair Style Generation"]
)

# Job storage (in production, use Redis or database)
processing_jobs = {}


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@router.get("/")
async def hairstyle_info():
    """Get hair style generation service info"""
    return {
        "service": "Hair Style Generation",
        "model": "SDXL + Inpainting with Mask",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "SDXL Base Model",
            "Precise Hair Masking",
            "Face Preservation",
            "Interactive Mask Editing",
            "Multiple Style Generation"
        ],
        "endpoints": {
            "/styles": "GET - List available styles",
            "/generate": "POST - Generate single style",
            "/generate-multiple": "POST - Generate multiple styles",
            "/generate-advanced": "POST - Advanced generation with parameters",
            "/create-mask": "POST - Create mask for image",
            "/generate-async": "POST - Start async generation",
            "/job/{job_id}": "GET - Check job status",
        }
    }


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


@router.post("/generate")
async def generate_single_style(
    file: UploadFile = File(..., description="Input face image"),
    style: str = Query(..., description="Hair style ID"),
    seed: Optional[int] = Query(None, description="Random seed for reproducibility"),
    steps: Optional[int] = Query(30, ge=10, le=50, description="Inference steps (10-50)"),
    denoising_strength: Optional[float] = Query(0.35, ge=0.1, le=0.8, description="Denoising strength (0.1-0.8)"),
    return_mask: bool = Query(False, description="Return mask image in response")
):
    """
    Generate a single hair style with SDXL Inpainting

    - **file**: Face image (JPG, PNG)
    - **style**: Style ID (see /styles endpoint)
    - **seed**: Optional random seed
    - **steps**: Inference steps (default: 30)
    - **denoising_strength**: How much to change the hair (default: 0.35)
    - **return_mask**: Return mask in JSON response

    Returns: Generated image or JSON with image and mask
    """
    try:
        # Load generator - CHANGED to new SDXL service
        from app.services.Hairstyle_service import get_hair_generator
        generator = get_hair_generator()

        # Validate style
        from app.config.hair_config import HairStylePrompts
        if style not in HairStylePrompts.HAIR_STYLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style. Use /styles endpoint to see available styles."
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image file")

        logger.info(f"Generating style '{style}' for image {file.filename}")

        # Generate using new SDXL service - CHANGED
        result = generator.generate_single_style(
            image=image,
            style_name=style,
            seed=seed,
            num_steps=steps,
            denoising_strength=denoising_strength
        )

        # If return_mask is True, return JSON with both image and mask
        if return_mask:
            # Convert images to base64
            img_byte_arr = io.BytesIO()
            result['result'].save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            mask_byte_arr = io.BytesIO()
            result['mask'].save(mask_byte_arr, format='PNG')
            mask_base64 = base64.b64encode(mask_byte_arr.getvalue()).decode('utf-8')

            return JSONResponse({
                "status": "success",
                "style": style,
                "seed": result['settings']['seed'],
                "image": f"data:image/png;base64,{img_base64}",
                "mask": f"data:image/png;base64,{mask_base64}",
                "face_detected": result['processing_info']['has_face'],
                "prompts": result['prompts']
            })

        # Otherwise, return just the image
        img_byte_arr = io.BytesIO()
        result['result'].save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Style-Name": style,
                "X-Seed": str(result['settings']['seed']),
                "X-Denoising-Strength": str(denoising_strength),
                "Content-Disposition": f'inline; filename="hairstyle_{style}.png"'
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating hair style: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate-advanced")
async def generate_advanced(
    file: UploadFile = File(..., description="Input face image"),
    style: str = Form(..., description="Hair style ID"),
    prompt: Optional[str] = Form(None, description="Custom prompt (overrides style prompt)"),
    negative_prompt: Optional[str] = Form(None, description="Custom negative prompt"),
    seed: Optional[int] = Form(None, description="Random seed"),
    steps: int = Form(30, ge=10, le=50, description="Inference steps"),
    denoising_strength: float = Form(0.35, ge=0.1, le=0.8, description="Denoising strength"),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0, description="Guidance scale"),
    use_mask: bool = Form(True, description="Use hair mask"),
    custom_mask: Optional[str] = Form(None, description="Base64 encoded custom mask")
):
    """
    Advanced hair style generation with custom parameters

    - Supports custom prompts
    - Custom masks
    - Fine-tuned parameters
    """
    try:
        from app.services.Hairstyle_service import get_hair_generator
        from app.config.hair_config import HairStylePrompts

        generator = get_hair_generator()

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process custom mask if provided
        custom_mask_array = None
        if custom_mask:
            try:
                mask_data = base64.b64decode(custom_mask.split(',')[1] if ',' in custom_mask else custom_mask)
                mask_nparr = np.frombuffer(mask_data, np.uint8)
                custom_mask_array = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
            except Exception as mask_error:
                logger.warning(f"Error decoding custom mask: {mask_error}")

        # Get style config for fallback prompts
        style_config = HairStylePrompts.HAIR_STYLES.get(style, {})

        # Use custom prompt or style prompt
        final_prompt = prompt or style_config.get('prompt', '')
        final_negative = negative_prompt or style_config.get('negative', '')

        # For demo - you'd need to modify the service to accept custom prompts
        # For now, we'll use the standard method

        result = generator.generate_single_style(
            image=image,
            style_name=style,
            seed=seed,
            num_steps=steps,
            denoising_strength=denoising_strength,
            custom_mask=custom_mask_array
        )

        img_byte_arr = io.BytesIO()
        result['result'].save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Style-Name": style,
                "X-Seed": str(result['settings']['seed']),
                "Content-Disposition": f'inline; filename="hairstyle_{style}.png"'
            }
        )

    except Exception as e:
        logger.error(f"Error in advanced generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-mask")
async def create_mask(
    file: UploadFile = File(...),
    method: str = Query("auto", description="Mask creation method: auto, sam, fallback, or points"),
    points: Optional[str] = Query(None, description="JSON array of points for manual mask [[x1,y1],[x2,y2],...]")
):
    """
    Create hair mask for an image

    Returns: JSON with mask image and preview
    """
    try:
        from app.services.Hairstyle_service import get_hair_generator

        generator = get_hair_generator()

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if method == "points" and points:
            # Parse points for interactive mask
            try:
                points_list = json.loads(points)
                # Convert to list of tuples
                points_tuples = [(int(p[0]), int(p[1])) for p in points_list]

                # Use interactive mask creation
                mask_result = generator.interactive_mask_creation(image, points_tuples)

                # Convert to base64
                mask_bytes = io.BytesIO()
                Image.fromarray(mask_result['mask']).save(mask_bytes, format='PNG')
                mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')

                preview_bytes = io.BytesIO()
                Image.fromarray(cv2.cvtColor(mask_result['preview'], cv2.COLOR_BGR2RGB)).save(preview_bytes, format='PNG')
                preview_base64 = base64.b64encode(preview_bytes.getvalue()).decode('utf-8')

                return {
                    "status": "success",
                    "mask": f"data:image/png;base64,{mask_base64}",
                    "preview": f"data:image/png;base64,{preview_base64}",
                    "method": "interactive",
                    "points": points_tuples
                }
            except Exception as e:
                logger.error(f"Error processing points: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid points format: {e}")

        else:
            # Use auto mask generation
            face_info = generator.face_extractor.get_face_info(image)

            if method == "sam":
                # Try SAM if available
                mask = generator.mask_generator.create_hair_mask_sam(image,
                    face_info['bbox'] if face_info else None)
            else:
                # Fallback method
                mask = generator.mask_generator.create_hair_mask_fallback(image,
                    face_info['bbox'] if face_info else None)

            # Create preview
            preview = image.copy()
            mask_overlay = mask.astype(np.float32) / 255.0
            preview = preview.astype(np.float32)
            preview[:, :, 1] = preview[:, :, 1] * (1 - mask_overlay) + 255 * mask_overlay
            preview = preview.astype(np.uint8)

            # Convert to base64
            mask_bytes = io.BytesIO()
            Image.fromarray(mask).save(mask_bytes, format='PNG')
            mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')

            preview_bytes = io.BytesIO()
            Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)).save(preview_bytes, format='PNG')
            preview_base64 = base64.b64encode(preview_bytes.getvalue()).decode('utf-8')

            return {
                "status": "success",
                "mask": f"data:image/png;base64,{mask_base64}",
                "preview": f"data:image/png;base64,{preview_base64}",
                "method": method,
                "face_detected": face_info is not None
            }

    except Exception as e:
        logger.error(f"Error creating mask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-multiple")
async def generate_multiple_styles(
    file: UploadFile = File(...),
    styles: str = Query(..., description="Comma-separated style IDs or 'random'"),
    variations: int = Query(1, ge=1, le=3, description="Variations per style"),
    denoising_strength: Optional[float] = Query(0.35, description="Denoising strength")
):
    """
    Generate multiple hair styles at once

    - **file**: Face image
    - **styles**: Comma-separated style IDs (e.g., "short_modern,bob_cut,pixie_cut") or "random"
    - **variations**: Number of variations per style (1-3)
    - **denoising_strength**: How much to change hair (0.1-0.8)

    Returns: ZIP file with all generated images
    """
    try:
        # Load generator - CHANGED
        from app.services.Hairstyle_service import get_hair_generator
        from app.config.hair_config import HairStylePrompts

        generator = get_hair_generator()

        # Parse styles
        if styles.lower() == "random":
            import random
            all_styles = HairStylePrompts.get_style_list()
            style_list = random.sample(all_styles, min(4, len(all_styles)))
        else:
            style_list = [s.strip() for s in styles.split(",")]

        # Validate styles
        invalid = [s for s in style_list if s not in HairStylePrompts.HAIR_STYLES]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid styles: {invalid}"
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        logger.info(f"Generating {len(style_list)} styles with {variations} variations each")

        # Generate all styles - CHANGED: result is now a dict
        results = generator.generate_multiple_styles(
            image=image,
            style_names=style_list,
            num_variations=variations,
            denoising_strength=denoising_strength
        )

        # Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for style_name, images_list in results.items():
                for idx, result_dict in enumerate(images_list):
                    # Extract image from result dict
                    img = result_dict['result']

                    # Convert PIL to bytes
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', quality=95)

                    # Add to ZIP
                    filename = f"{style_name}_{idx + 1}.png"
                    zip_file.writestr(filename, img_buffer.getvalue())

                    # Optional: add mask
                    # mask_filename = f"{style_name}_{idx + 1}_mask.png"
                    # mask_buffer = io.BytesIO()
                    # result_dict['mask'].save(mask_buffer, format='PNG')
                    # zip_file.writestr(mask_filename, mask_buffer.getvalue())

        zip_buffer.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="hairstyles_{timestamp}.zip"'
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating multiple styles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-async")
async def start_async_generation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    styles: str = Query("random", description="Style IDs or 'random'"),
    variations: int = Query(1, ge=1, le=3),
    denoising_strength: float = Query(0.35, description="Denoising strength")
):
    """
    Start async hair style generation (for long-running tasks)

    Returns: Job ID to check status later
    """
    try:
        from app.config.hair_config import HairStylePrompts

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Save uploaded file
        temp_path = Path(f"/tmp/{job_id}_input.jpg")  # Adjust for your system
        contents = await file.read()

        with open(temp_path, 'wb') as f:
            f.write(contents)

        # Parse styles
        if styles.lower() == "random":
            import random
            all_styles = HairStylePrompts.get_style_list()
            style_list = random.sample(all_styles, min(4, len(all_styles)))
        else:
            style_list = [s.strip() for s in styles.split(",")]

        # Initialize job
        processing_jobs[job_id] = {
            "status": JobStatus.PENDING,
            "progress": 0,
            "total": len(style_list) * variations,
            "created_at": datetime.utcnow().isoformat(),
            "styles": style_list,
            "denoising_strength": denoising_strength,
            "result_path": None,
            "error": None
        }

        # Add to background tasks
        background_tasks.add_task(
            process_generation_job,
            job_id,
            temp_path,
            style_list,
            variations,
            denoising_strength
        )

        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Generation job started",
            "styles": style_list,
            "total_images": len(style_list) * variations,
            "denoising_strength": denoising_strength,
            "check_url": f"/api/v1/hairstyle/job/{job_id}"
        }

    except Exception as e:
        logger.error(f"Error starting async job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_generation_job(
    job_id: str,
    image_path: Path,
    style_list: List[str],
    variations: int,
    denoising_strength: float
):
    """Background task to process generation"""
    try:
        from app.services.Hairstyle_service import get_hair_generator

        # Update status
        processing_jobs[job_id]["status"] = JobStatus.PROCESSING

        # Load image
        image = cv2.imread(str(image_path))

        # Get generator
        generator = get_hair_generator()

        # Generate
        results = generator.generate_multiple_styles(
            image=image,
            style_names=style_list,
            num_variations=variations,
            denoising_strength=denoising_strength
        )

        # Save to ZIP
        result_path = Path(f"/tmp/{job_id}_results.zip")  # Adjust for your system

        with zipfile.ZipFile(result_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for style_name, images_list in results.items():
                for idx, result_dict in enumerate(images_list):
                    img = result_dict['result']
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', quality=95)

                    filename = f"{style_name}_{idx + 1}.png"
                    zip_file.writestr(filename, img_buffer.getvalue())

        # Update job
        processing_jobs[job_id].update({
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "result_path": str(result_path),
            "completed_at": datetime.utcnow().isoformat()
        })

        # Cleanup
        image_path.unlink()

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        processing_jobs[job_id].update({
            "status": JobStatus.FAILED,
            "error": str(e)
        })


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Check status of async generation job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "created_at": job["created_at"],
        "styles": job.get("styles", []),
        "denoising_strength": job.get("denoising_strength", 0.35)
    }

    if job["status"] == JobStatus.COMPLETED:
        response["download_url"] = f"/api/v1/hairstyle/download/{job_id}"
        response["completed_at"] = job.get("completed_at")

    if job["status"] == JobStatus.FAILED:
        response["error"] = job.get("error")

    return response


@router.get("/download/{job_id}")
async def download_job_results(job_id: str):
    """Download completed job results"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    result_path = Path(job["result_path"])

    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=result_path,
        media_type="application/zip",
        filename=f"hairstyles_{job_id}.zip"
    )


@router.get("/health")
async def health_check():
    """Health check for hair style service"""
    try:
        import torch
        from app.services.Hairstyle_service import get_hair_generator

        # Try to get generator to check if model is loaded
        generator = get_hair_generator()
        model_loaded = generator.pipe is not None

        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "active_jobs": len([j for j in processing_jobs.values() if j["status"] == JobStatus.PROCESSING]),
            "model_type": "SDXL Inpainting"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }