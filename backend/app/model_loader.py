"""
Model loader — loads the trained brain tumor classification model at startup
and provides inference utilities.
"""
import json
import os
from typing import Dict, Tuple, Optional

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
        self._loaded = False

    def load(self, model_path: str, config_path: str):
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
        Resizes to target size and normalizes to [0, 1].
        """
        image = image.convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a PIL image.

        Returns:
            {
                "tumor_type": str,
                "confidence": float,
                "severity": str,
                "all_scores": {class_name: score, ...}
            }
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Run training first.")

        img_array = self.preprocess_image(image)
        predictions = self.model.predict(img_array, verbose=0)
        scores = predictions[0]

        # Get predicted class
        pred_idx = int(np.argmax(scores))
        predicted_class = self.class_names[pred_idx]
        confidence = float(scores[pred_idx])

        # Build all scores dict
        all_scores = {
            name: round(float(scores[i]), 4)
            for i, name in enumerate(self.class_names)
        }

        # Determine severity
        severity = self._compute_severity(predicted_class, confidence)

        return {
            "tumor_type": predicted_class,
            "confidence": round(confidence, 4),
            "severity": severity,
            "all_scores": all_scores,
        }

    def _compute_severity(self, tumor_type: str, confidence: float) -> str:
        """Map prediction to severity based on confidence thresholds."""
        if tumor_type == "notumor":
            return "None"

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
