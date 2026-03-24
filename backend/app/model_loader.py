"""
Model loader — loads the trained brain tumor classification model at startup
and provides inference utilities.

IMPORTANT: Preprocessing MUST match the training pipeline exactly:
  - Resize to 224x224
  - Apply tf.keras.applications.efficientnet.preprocess_input
  - This scales [0, 255] → [-1, 1] to match ImageNet pretrained weights
"""
import json
import os
from typing import Dict, Optional

import numpy as np
from PIL import Image
import tensorflow as tf


class TumorModelLoader:
    """Singleton-style model loader for the brain tumor classifier."""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.config: Optional[dict] = None
        self.class_names: list = []
        self.img_size: int = 224
        self.severity_thresholds: dict = {}
        self.backbone: str = "unknown"
        self.confidence_threshold: float = 0.50
        self._loaded = False

    def load(self, model_path: str, config_path: str) -> bool:
        """Load model weights and config from disk."""
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}. Run training first.")
            return False

        if not os.path.exists(config_path):
            print(f"⚠️  Config not found at {config_path}. Run training first.")
            return False

        try:
            print(f"📦 Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)

            with open(config_path, "r") as f:
                self.config = json.load(f)

            self.class_names = self.config.get("class_names", [])
            self.img_size = self.config.get("img_size", 224)
            self.severity_thresholds = self.config.get("severity_thresholds", {
                "high": 0.85,
                "moderate": 0.60,
            })
            self.backbone = self.config.get("backbone", "unknown")
            self._loaded = True

            print(f"✅ Model loaded successfully ({self.backbone} backbone)")
            print(f"   Classes: {self.class_names}")
            print(f"   Test accuracy: {self.config.get('test_accuracy', 'N/A')}")
            print(f"   Training mode: {self.config.get('training_mode', 'N/A')}")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for model inference.

        Pipeline (MUST match training):
          1. Convert to RGB
          2. Resize to target size
          3. Apply EfficientNet preprocess_input ([0,255] → [-1,1])

        Args:
            image: PIL Image of any size/mode.

        Returns:
            Numpy array of shape (1, img_size, img_size, 3).
        """
        # Ensure RGB
        image = image.convert("RGB")

        # Resize to model's expected input size
        image = image.resize((self.img_size, self.img_size))

        # Convert to float32 array (keep [0, 255] range)
        img_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Apply EfficientNet preprocessing (scales [0,255] → [-1,1])
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        return img_array

    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                "tumor_type": str,
                "confidence": float,
                "severity": str,
                "all_scores": {class_name: score, ...},
                "is_uncertain": bool,
            }
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Run training first.")

        # Preprocess (matches training pipeline)
        img_array = self.preprocess_image(image)

        # Run inference
        predictions = self.model.predict(img_array, verbose=0)
        scores = predictions[0]

        # Get predicted class
        pred_idx = int(np.argmax(scores))
        predicted_class = self.class_names[pred_idx]
        confidence = float(scores[pred_idx])

        # Check if prediction is uncertain
        is_uncertain = confidence < self.confidence_threshold

        # Build all scores dict
        all_scores = {
            name: round(float(scores[i]), 4)
            for i, name in enumerate(self.class_names)
        }

        # Determine severity
        severity = self._compute_severity(predicted_class, confidence, is_uncertain)

        return {
            "tumor_type": predicted_class,
            "confidence": round(confidence, 4),
            "severity": severity,
            "all_scores": all_scores,
            "is_uncertain": is_uncertain,
        }

    def _compute_severity(self, tumor_type: str, confidence: float,
                          is_uncertain: bool) -> str:
        """Map prediction to severity based on confidence thresholds."""
        if tumor_type == "notumor":
            return "None"

        if is_uncertain:
            return "Uncertain — requires manual review"

        high = self.severity_thresholds.get("high", 0.85)
        moderate = self.severity_thresholds.get("moderate", 0.60)

        if confidence >= high:
            return "High"
        elif confidence >= moderate:
            return "Moderate"
        else:
            return "Low"


# Global instance
model_loader = TumorModelLoader()
