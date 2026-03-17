"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response from the /api/predict endpoint."""
    id: int
    tumor_type: str
    confidence: float
    severity: str
    all_scores: Dict[str, float]
    image_url: str
    created_at: datetime

    class Config:
        from_attributes = True


class ScanRecordResponse(BaseModel):
    """A single scan record from history."""
    id: int
    filename: str
    original_filename: Optional[str] = None
    tumor_type: str
    confidence: float
    severity: Optional[str] = None
    details: Optional[Dict[str, float]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    backbone: Optional[str] = None


class DeleteResponse(BaseModel):
    """Delete confirmation response."""
    message: str
    id: int
