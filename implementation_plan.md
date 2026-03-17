# Brain Tumor Classification Web App — Merged Implementation Plan

**Mini Project 1.2 | React + FastAPI + PostgreSQL + SpineNet/TensorFlow**

## Overview

A full-stack web app that classifies brain MRI scans into **4 tumor types** using a CNN trained on a real labelled dataset. Uses **SpineNet** as the primary backbone architecture for its superior multi-scale feature extraction. Predictions are stored in PostgreSQL and surfaced in a history view.

> [!NOTE]
> **Staging limitation:** The dataset has tumor *type* labels (glioma, meningioma, pituitary, no_tumor) but **no stage labels**. Staging is presented as a **risk severity assessment** (Low/Moderate/High) derived from prediction confidence. A second model head can be added if a staging dataset becomes available.

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Frontend | React (Vite) + TypeScript | Fast dev server, type safety, component reuse |
| Backend | FastAPI (Python) | Async, auto docs, easy file upload handling |
| Database | PostgreSQL + SQLAlchemy | Reliable, relational, easy history queries |
| ML Framework | TensorFlow / Keras | SpineNet implementation, saves as `.h5` |
| **Model** | **SpineNet + custom head** | Scale-permuted backbone with cross-scale connections; 4-class softmax (type), severity from confidence |
| Styling | Vanilla CSS (dark theme) | Glassmorphism cards, micro-animations |

---

## 1. Dataset

| Class | Training | Testing |
|-------|----------|---------|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

- Data lives under `Data/Training/<class>/` and `Data/Testing/<class>/`
- Class imbalance is mild; use `class_weight` in Keras or oversample if accuracy drops

---

## 2. Project Structure

```
d:\Projects\Mini Project 1.2 - Antigravity\
├── Data/                        # Existing dataset (Training/ + Testing/)
├── frontend/                    # React (Vite) + TypeScript
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── index.css
│   │   ├── pages/               # Home, Upload, Results, History
│   │   ├── components/          # UploadForm, ResultCard, HistoryTable, Navbar
│   │   └── api/                 # Axios client for backend
│   ├── package.json
│   └── vite.config.ts
├── backend/                     # Python FastAPI
│   ├── app/
│   │   ├── main.py              # Routes: /predict, /health, /history
│   │   ├── model_loader.py      # Load .h5 model, run inference
│   │   ├── database.py          # SQLAlchemy + PostgreSQL
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── models/                  # Saved model artifacts
│   │   ├── brain_tumor_model.h5
│   │   └── config.json          # Class names, severity thresholds
│   ├── uploads/                 # Uploaded scan storage
│   ├── requirements.txt
│   └── Dockerfile               # Optional
├── ml/                          # Training pipeline (top-level, independent)
│   ├── train.py                 # Train SpineNet model, save model + config
│   ├── spinenet.py              # SpineNet architecture implementation
│   ├── config.py                # Classes, paths, hyperparams
│   ├── dataset.py               # Dataset loader, augmentation
│   └── requirements.txt         # tensorflow, pillow, etc.
├── docker-compose.yml           # backend + postgres (optional)
└── README.md
```

---

## 3. Database (PostgreSQL)

```sql
CREATE TABLE scan_records (
    id               SERIAL PRIMARY KEY,
    filename         VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    tumor_type       VARCHAR(50)  NOT NULL,   -- glioma | meningioma | pituitary | no_tumor
    confidence       FLOAT        NOT NULL,   -- 0.0 to 1.0
    severity         VARCHAR(20),             -- Low | Moderate | High
    details          JSONB,                   -- full softmax scores per class
    created_at       TIMESTAMP DEFAULT NOW()
);
```

- Connection via env var: `DATABASE_URL=postgresql://user:pass@localhost:5432/brain_tumor_db`
- Tables created on startup (`SQLAlchemy create_all`) for dev
- `GET /history` returns last N records ordered by `created_at DESC`

---

## 4. ML Pipeline — SpineNet (`ml/`)

### Why SpineNet?

SpineNet (Google Research, CVPR 2020) is a **scale-permuted backbone** that outperforms conventional models on multi-scale recognition tasks:

- **Cross-scale connections** let it integrate features from different resolutions — critical for detecting tumors of varying sizes in MRI scans
- **NAS-discovered architecture** optimizes feature block permutations automatically
- ~5% higher top-1 accuracy vs ResNet on fine-grained datasets, with fewer FLOPs
- Better suited for medical imaging where tumors appear at multiple scales and locations

### Model Architecture

| Component | Detail |
|-----------|--------|
| **Backbone** | SpineNet-49 (scale-permuted, cross-scale connections) |
| **Input** | 224×224 RGB, normalized to [0, 1] |
| **Head** | GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Dense(4, softmax) |
| **Loss** | `categorical_crossentropy` |
| **Optimizer** | Adam (lr=1e-4, with ReduceLROnPlateau) |
| **Output** | `backend/models/brain_tumor_model.h5` + `config.json` |

> [!TIP]
> **Fallback:** If SpineNet training proves too resource-heavy on the local machine, `train.py` includes a `--backbone mobilenetv2` flag to fall back to MobileNetV2 transfer learning.

### Severity Mapping

| Condition | Confidence | Severity |
|-----------|-----------|----------|
| Tumor detected | ≥ 0.85 | **High** |
| Tumor detected | 0.60 – 0.84 | **Moderate** |
| Tumor detected | < 0.60 | **Low** (review recommended) |
| No tumor | Any | **None** |

### Training Script (`ml/train.py`)

- Reads from `Data/Training/` and `Data/Testing/` using `ImageDataGenerator.flow_from_directory`
- Augmentation: `rotation_range=20`, `horizontal_flip`, `zoom_range=0.1` (training only)
- Train/val split: 80/20 from Training folder; Testing folder = held-out test set
- Saves model weights + `config.json` (class indices, severity thresholds) to `backend/models/`
- Target accuracy: **>85%** on test set

---

## 5. Backend (FastAPI)

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/predict` | Upload MRI image, run inference, save to DB, return result |
| `GET` | `/api/history` | Return last N scan records (default 50) |
| `GET` | `/api/history/{id}` | Single prediction record detail |
| `DELETE` | `/api/history/{id}` | Delete a scan record |
| `GET` | `/health` | Readiness check |

### Key Notes

- `model_loader.py` loads `brain_tumor_model.h5` + `config.json` once at startup (lifespan event)
- Preprocessing must match training: resize to 224×224, normalize to [0, 1]
- CORS: allow `http://localhost:5173` (Vite dev server)
- Uploaded files saved to `uploads/` with UUID filename
- Response shape: `{ tumor_type, confidence, severity, all_scores, image_url }`

---

## 6. Frontend (React + TypeScript)

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Hero section, project overview, dataset stats, call-to-action |
| Upload | `/upload` | Drag-and-drop file picker, image preview, submit, loading state |
| Results | `/results` | Tumor type, confidence bar, severity badge, scan thumbnail |
| History | `/history` | Table of past scans with sort, filter, delete per row |

### Design System

- **Theme:** Dark medical — deep navy/charcoal base (`#0F172A`), teal/cyan accents (`#0BC5EA`)
- **Cards:** Glassmorphism (`backdrop-filter: blur`, semi-transparent backgrounds)
- **Confidence bar:** Animated fill, color-coded (green < 60%, orange 60–84%, red ≥ 85%)
- **Severity badge:** Pill component — grey (None), green (Low), orange (Moderate), red (High)
- **Typography:** Inter (Google Fonts), accessible contrast ratios
- **Icons:** Lucide React

---

## 7. Running the Stack

### Prerequisites

- PostgreSQL running on `localhost:5432`
- Python 3.9+ with pip
- Node.js 18+ with npm
- Create database: `CREATE DATABASE brain_tumor_db;`

### Steps

```bash
# Step 1 — Train the model
cd ml
pip install -r requirements.txt
python train.py                          # uses SpineNet by default
# python train.py --backbone mobilenetv2  # fallback option

# Step 2 — Start the backend
cd backend
pip install -r requirements.txt
set DATABASE_URL=postgresql://postgres:yourpass@localhost:5432/brain_tumor_db
uvicorn app.main:app --reload

# Step 3 — Start the frontend
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

---

## 8. Verification Plan

### Automated Checks

1. **Model accuracy:** `cd ml && python train.py` → expect test accuracy > 85%
2. **API smoke test:**
   ```bash
   curl -X POST http://localhost:8000/api/predict -F "file=@../Data/Testing/glioma/Te-gl_0010.jpg"
   # Expect: { "tumor_type": "glioma", "confidence": 0.xx, "severity": "High" }
   ```
3. **Frontend build:** `cd frontend && npm run build` → no errors

### Manual QA

- Upload a glioma scan → verify correct prediction + High severity
- Upload a no_tumor scan → verify `tumor_type = no_tumor` + severity = None
- Navigate to History → confirm record appears with correct values
- Delete a record from History → confirm removal
- Test responsive layout on mobile viewport

---

## 9. Optional Enhancements

- **Docker Compose:** One service each for API, Postgres, and frontend (nginx)
- **Grad-CAM overlay:** Highlight regions the SpineNet model focused on (medical explainability)
- **ONNX export:** Convert model for lighter footprint via `onnxruntime`
- **Second model head:** Add stage classification if a staging dataset becomes available
- **API key auth:** Simple header-based key for demo access restriction
