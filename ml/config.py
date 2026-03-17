"""
Configuration for the brain tumor classification ML pipeline.
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

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 128

# ─── Augmentation ────────────────────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range": 20,
    "horizontal_flip": True,
    "zoom_range": 0.1,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "fill_mode": "nearest",
}

# ─── Severity Thresholds ────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "high": 0.85,
    "moderate": 0.60,
}
