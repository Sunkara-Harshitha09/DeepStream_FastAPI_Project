from typing import List

from pydantic import BaseModel, Field


# ============================================================
# BOUNDING BOX
# ============================================================

class BoundingBox(BaseModel):

    x1: float

    y1: float

    x2: float

    y2: float


# ============================================================
# DETECTED OBJECT
# ============================================================

class DetectedObject(BaseModel):

    class_id: int = Field(
        ge=0
    )

    class_name: str = Field(
        min_length=1
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    bounding_box: BoundingBox


# ============================================================
# FRAME DETECTION
# ============================================================

class FrameDetection(BaseModel):

    frame_number: int = Field(
        ge=1
    )

    timestamp_seconds: float = Field(
        ge=0.0
    )

    objects: List[DetectedObject]


# ============================================================
# COMPLETE DETECTION JSON
# ============================================================

class DetectionJSON(BaseModel):

    video_name: str = Field(
        min_length=1
    )

    video_path: str = Field(
        min_length=1
    )

    total_frames: int = Field(
        ge=0
    )

    fps: float = Field(
        ge=0.0
    )

    width: int = Field(
        ge=0
    )

    height: int = Field(
        ge=0
    )

    generated_at: str = Field(
        min_length=1
    )

    detections: List[FrameDetection]