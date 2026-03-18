"""
FastAPI application for brain tumor classification.

Endpoints:
    POST   /api/predict       Upload MRI image, run inference, store result
    GET    /api/history        List past scan records
    GET    /api/history/{id}   Get single scan record
    DELETE /api/history/{id}   Delete a scan record
    GET    /health             Readiness check
"""
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from PIL import Image

from app.database import init_db, get_db, ScanRecord
from app.schemas import (
    PredictionResponse,
    ScanRecordResponse,
    HealthResponse,
    DeleteResponse,
)
from app.model_loader import model_loader


# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load ML model. Shutdown: cleanup."""
    print("\n🧠 Brain Tumor Classification API starting up...")
    init_db()
    model_loader.load(MODEL_PATH, CONFIG_PATH)
    yield
    print("\n👋 Shutting down API.")


# ─── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Classification API",
    description="Upload brain MRI scans for tumor type classification and severity assessment.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vite dev server and production frontend
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    os.getenv("FRONTEND_URL", "*")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images as static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ─── POST /api/predict ────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload an MRI scan image and get tumor classification results."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file (JPEG, PNG)."
        )

    # Check model is loaded
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )

    try:
        # Save uploaded file with UUID name
        ext = os.path.splitext(file.filename or "scan.jpg")[1] or ".jpg"
        saved_filename = f"{uuid.uuid4().hex}{ext}"
        saved_path = os.path.join(UPLOAD_DIR, saved_filename)

        contents = await file.read()
        with open(saved_path, "wb") as f:
            f.write(contents)

        # Run inference
        image = Image.open(saved_path)
        result = model_loader.predict(image)

        # Save to database
        record = ScanRecord(
            filename=saved_filename,
            original_filename=file.filename,
            tumor_type=result["tumor_type"],
            confidence=result["confidence"],
            severity=result["severity"],
            details=result["all_scores"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return PredictionResponse(
            id=record.id,
            tumor_type=result["tumor_type"],
            confidence=result["confidence"],
            severity=result["severity"],
            all_scores=result["all_scores"],
            image_url=f"/uploads/{saved_filename}",
            created_at=record.created_at,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── GET /api/history ─────────────────────────────────────────────────────
@app.get("/api/history", response_model=List[ScanRecordResponse])
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve past scan records, newest first."""
    records = (
        db.query(ScanRecord)
        .order_by(ScanRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return records


# ─── GET /api/history/{id} ────────────────────────────────────────────────
@app.get("/api/history/{record_id}", response_model=ScanRecordResponse)
async def get_record(record_id: int, db: Session = Depends(get_db)):
    """Get a single scan record by ID."""
    record = db.query(ScanRecord).filter(ScanRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")
    return record


# ─── DELETE /api/history/{id} ─────────────────────────────────────────────
@app.delete("/api/history/{record_id}", response_model=DeleteResponse)
async def delete_record(record_id: int, db: Session = Depends(get_db)):
    """Delete a scan record and its uploaded image."""
    record = db.query(ScanRecord).filter(ScanRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")

    # Remove uploaded file
    file_path = os.path.join(UPLOAD_DIR, record.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.delete(record)
    db.commit()

    return DeleteResponse(message="Record deleted successfully.", id=record_id)


# ─── GET /health ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API readiness check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded,
        backbone=model_loader.backbone if model_loader.is_loaded else None,
    )
