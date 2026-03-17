# NeuroScan AI — Brain Tumor Classification

A full-stack web application that classifies brain MRI scans into **4 tumor types** using a **SpineNet**-based CNN, with severity assessment.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript (Vite) |
| Backend | FastAPI (Python) |
| Database | PostgreSQL + SQLAlchemy |
| ML Model | SpineNet-49 / MobileNetV2 (TensorFlow/Keras) |

## Quick Start

### 1. Train the Model

```bash
cd ml
pip install -r requirements.txt
python train.py                          # SpineNet (default)
# python train.py --backbone mobilenetv2 # fallback
```

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
set DATABASE_URL=postgresql://postgres:yourpass@localhost:5432/brain_tumor_db
uvicorn app.main:app --reload
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Dataset

| Class | Training | Testing |
|-------|----------|---------|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/predict` | Upload MRI, get classification |
| GET | `/api/history` | List past scans |
| GET | `/api/history/{id}` | Single record |
| DELETE | `/api/history/{id}` | Delete record |
| GET | `/health` | Health check |
