"""
Hair Style Pydantic Schemas
File: app/schemas/hairstyle.py (hoặc app/schemas/hairstyle_schemas.py)
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class JobStatusEnum(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HairStyleInfo(BaseModel):
    """Hair style information"""
    id: str = Field(..., description="Style ID")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Style description")
    category: Optional[str] = Field(None, description="Style category")

    class Config:
        schema_extra = {
            "example": {
                "id": "short_modern",
                "name": "Short Modern",
                "description": "Professional short haircut with modern style",
                "category": "short"
            }
        }


class HairStyleListResponse(BaseModel):
    """Response for listing hair styles"""
    total: int = Field(..., description="Total number of styles")
    styles: List[HairStyleInfo] = Field(..., description="List of available styles")


class GenerateSingleRequest(BaseModel):
    """Request body for single style generation (if using JSON instead of form-data)"""
    style: str = Field(..., description="Hair style ID")
    seed: Optional[int] = Field(None, description="Random seed", ge=0)
    steps: Optional[int] = Field(25, description="Inference steps", ge=10, le=50)

    @validator('style')
    def validate_style(cls, v):
        # You can add validation here
        if not v:
            raise ValueError("Style cannot be empty")
        return v.strip().lower()

    class Config:
        schema_extra = {
            "example": {
                "style": "short_modern",
                "seed": 42,
                "steps": 25
            }
        }


class GenerateMultipleRequest(BaseModel):
    """Request for multiple styles generation"""
    styles: List[str] = Field(..., description="List of style IDs or ['random']")
    variations: int = Field(1, description="Variations per style", ge=1, le=3)
    seed: Optional[int] = Field(None, ge=0)

    @validator('styles')
    def validate_styles(cls, v):
        if not v:
            raise ValueError("At least one style required")
        return [s.strip().lower() for s in v]

    class Config:
        schema_extra = {
            "example": {
                "styles": ["short_modern", "bob_cut", "pixie_cut"],
                "variations": 1,
                "seed": 42
            }
        }


class JobStartResponse(BaseModel):
    """Response when starting an async job"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatusEnum = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")
    styles: List[str] = Field(..., description="Styles to generate")
    total_images: int = Field(..., description="Total images to generate")
    check_url: str = Field(..., description="URL to check job status")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "pending",
                "message": "Generation job started",
                "styles": ["short_modern", "bob_cut"],
                "total_images": 2,
                "check_url": "/api/v1/hairstyle/job/123e4567-e89b-12d3-a456-426614174000"
            }
        }


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatusEnum = Field(..., description="Current status")
    progress: int = Field(..., description="Progress percentage", ge=0, le=100)
    total: int = Field(..., description="Total images to generate")
    created_at: str = Field(..., description="Job creation timestamp")
    download_url: Optional[str] = Field(None, description="Download URL if completed")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "processing",
                "progress": 50,
                "total": 4,
                "created_at": "2025-01-01T00:00:00",
                "download_url": None,
                "completed_at": None,
                "error": None
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    cuda_available: bool = Field(..., description="CUDA availability")
    device: str = Field(..., description="Device being used")
    active_jobs: int = Field(..., description="Number of active jobs")
    error: Optional[str] = Field(None, description="Error if unhealthy")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "cuda_available": True,
                "device": "cuda",
                "active_jobs": 2,
                "error": None
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        schema_extra = {
            "example": {
                "detail": "No face detected in image",
                "error_code": "NO_FACE_DETECTED",
                "timestamp": "2025-01-01T00:00:00"
            }
        }


class GenerationMetadata(BaseModel):
    """Metadata for generated image"""
    style_name: str
    seed: int
    steps: int
    guidance_scale: float
    generation_time: float  # seconds
    model_version: str

    class Config:
        schema_extra = {
            "example": {
                "style_name": "short_modern",
                "seed": 42,
                "steps": 25,
                "guidance_scale": 7.5,
                "generation_time": 12.5,
                "model_version": "SD1.5"
            }
        }