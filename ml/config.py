"""
Configuration for the brain tumor classification ML pipeline.

Architecture: EfficientNetB0 with Transfer Learning
Dataset: 4-class brain tumor MRI (glioma, meningioma, notumor, pituitary)
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "models")
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "brain_tumor_model.h5")
CONFIG_PATH = os.path.join(MODEL_OUTPUT_DIR, "config.json")
HISTORY_PATH = os.path.join(MODEL_OUTPUT_DIR, "training_history.json")

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50                  # Max epochs (early stopping will cut short)
LEARNING_RATE = 1e-3         # Phase 1: frozen base layers
FINE_TUNE_LR = 1e-5          # Phase 2: fine-tuning top layers
VALIDATION_SPLIT = 0.2
FINE_TUNE_AT_LAYER = 100     # Unfreeze EfficientNetB0 from this layer onwards

# ─── Regularization ─────────────────────────────────────────────────────────
DROPOUT_RATE = 0.5           # Single strong dropout after GAP
LABEL_SMOOTHING = 0.1        # Prevent overconfident predictions

# ─── Classification Head ────────────────────────────────────────────────────
DENSE_UNITS = 256            # Single dense layer (simpler = less overfitting)

# ─── Data Augmentation (Strong, MRI-appropriate) ────────────────────────────
AUGMENTATION = {
    "rotation_range": 30,           # MRIs can be rotated
    "width_shift_range": 0.15,      # Shift for position variability
    "height_shift_range": 0.15,
    "zoom_range": 0.2,              # Handle different crop levels
    "horizontal_flip": True,        # Brain symmetry
    "vertical_flip": True,          # Different scan orientations
    "brightness_range": [0.8, 1.2], # Scanner intensity variation
    "shear_range": 0.1,             # Slight perspective changes
    "fill_mode": "nearest",         # Best for medical images
}

# ─── Callbacks ───────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 5
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-7

# ─── Severity Thresholds ────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "high": 0.85,
    "moderate": 0.60,
}

# ─── Confidence Threshold ───────────────────────────────────────────────────
# If max prediction confidence is below this, flag as "uncertain"
CONFIDENCE_THRESHOLD = 0.50
