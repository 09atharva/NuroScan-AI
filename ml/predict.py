"""
Single-image prediction script with debugging and visualization.

Handles common inference issues:
  - Shows input image before prediction (debugging)
  - Reports confidence scores for ALL classes
  - Flags low-confidence predictions as uncertain
  - Handles edge cases (corrupt images, wrong format, etc.)
  - Consistent preprocessing with training pipeline

Usage:
    python predict.py --image /path/to/mri_scan.jpg
    python predict.py --image /path/to/scan.png --show
    python predict.py --image-dir /path/to/folder/   # Batch prediction
"""
import argparse
import os
import sys
import glob

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    MODEL_PATH, CONFIG_PATH, CLASS_NAMES,
    IMG_SIZE, CONFIDENCE_THRESHOLD, MODEL_OUTPUT_DIR,
)


def load_model(model_path: str):
    """Load the trained model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("   Run 'python train.py' first to train the model.")
        sys.exit(1)

    print(f"📦 Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")
    return model


def preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess a single image.

    Preprocessing matches the training pipeline:
      1. Convert to RGB
      2. Resize to 224x224
      3. Scale to [0, 1] (EfficientNet preprocess_input handles the rest inside the model)

    Returns:
        (preprocessed_array, original_image, error_message)
    """
    try:
        if not os.path.exists(image_path):
            return None, None, f"File not found: {image_path}"

        img = Image.open(image_path)

        # Validate image
        if img.size[0] < 10 or img.size[1] < 10:
            return None, None, f"Image too small: {img.size}"

        # Convert to RGB (handles grayscale, RGBA, etc.)
        img_rgb = img.convert("RGB")
        original = img_rgb.copy()

        # Resize to model's expected input
        img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE))

        # Convert to float32 array (keep [0, 255] range)
        img_array = np.array(img_resized, dtype=np.float32)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Apply EfficientNet preprocessing ([0, 255] → [-1, 1])
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        return img_array, original, None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


def predict_single(model, image_path: str, verbose: bool = True) -> dict:
    """
    Run prediction on a single image.

    Returns:
        {
            "image_path": str,
            "tumor_type": str,
            "confidence": float,
            "all_scores": {class_name: score},
            "is_uncertain": bool,
            "status": "success" | "error",
            "error": str or None,
        }
    """
    # Preprocess
    img_array, original_img, error = preprocess_image(image_path)

    if error:
        if verbose:
            print(f"❌ {error}")
        return {
            "image_path": image_path,
            "tumor_type": "error",
            "confidence": 0.0,
            "all_scores": {},
            "is_uncertain": True,
            "status": "error",
            "error": error,
        }

    # Run inference
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]

    pred_idx = int(np.argmax(scores))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(scores[pred_idx])
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    all_scores = {name: round(float(scores[i]), 4) for i, name in enumerate(CLASS_NAMES)}

    if verbose:
        print(f"\n{'─'*50}")
        print(f"📸 Image: {os.path.basename(image_path)}")
        print(f"   Size: {original_img.size}")
        print(f"{'─'*50}")

        status = "⚠️  UNCERTAIN" if is_uncertain else "✅"
        print(f"\n   {status} Prediction: {pred_class.upper()}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

        if is_uncertain:
            print(f"   ⚠️  Confidence below threshold ({CONFIDENCE_THRESHOLD})")
            print(f"   ⚠️  This prediction may be UNRELIABLE!")

        print(f"\n   All probabilities:")
        for name in CLASS_NAMES:
            score = all_scores[name]
            bar = "█" * int(score * 30)
            marker = " ◄" if name == pred_class else ""
            print(f"     {name:15s}: {score:.4f} {bar}{marker}")

    return {
        "image_path": image_path,
        "tumor_type": pred_class,
        "confidence": confidence,
        "all_scores": all_scores,
        "is_uncertain": is_uncertain,
        "status": "success",
        "error": None,
    }


def visualize_prediction(model, image_path: str, output_path: str = None):
    """
    Create a visualization of the prediction with the input image.
    Useful for debugging model behavior.
    """
    result = predict_single(model, image_path, verbose=True)

    if result["status"] == "error":
        return result

    img_array, original_img, _ = preprocess_image(image_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Show input image
    axes[0].imshow(original_img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title(f"Input: {os.path.basename(image_path)}", fontsize=12)
    axes[0].axis("off")

    # Show prediction bar chart
    scores = list(result["all_scores"].values())
    colors = []
    for i, name in enumerate(CLASS_NAMES):
        if name == result["tumor_type"]:
            colors.append("#2196F3" if not result["is_uncertain"] else "#FF9800")
        else:
            colors.append("#E0E0E0")

    bars = axes[1].barh(CLASS_NAMES, scores, color=colors, edgecolor="white", height=0.6)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Confidence")
    axes[1].set_title("Prediction", fontsize=12)

    for bar, score in zip(bars, scores):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                      f"{score:.3f}", va="center", fontsize=10)

    status = "⚠️ UNCERTAIN" if result["is_uncertain"] else "✅"
    fig.suptitle(
        f"{status}  {result['tumor_type'].upper()} "
        f"({result['confidence']*100:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if output_path is None:
        output_dir = os.path.join(MODEL_OUTPUT_DIR, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"pred_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📸 Visualization saved to: {output_path}")

    return result


def batch_predict(model, image_dir: str):
    """Run predictions on all images in a directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_paths:
        print(f"❌ No images found in: {image_dir}")
        return []

    print(f"\n📂 Found {len(image_paths)} images in {image_dir}")
    print("=" * 50)

    results = []
    for path in sorted(image_paths):
        result = predict_single(model, path)
        results.append(result)

    # Summary
    print(f"\n{'='*50}")
    print("📊 BATCH PREDICTION SUMMARY")
    print(f"{'='*50}")

    successful = [r for r in results if r["status"] == "success"]
    uncertain = [r for r in successful if r["is_uncertain"]]

    print(f"  Total images:     {len(results)}")
    print(f"  Successful:       {len(successful)}")
    print(f"  Errors:           {len(results) - len(successful)}")
    print(f"  Uncertain:        {len(uncertain)}")

    # Class distribution
    from collections import Counter
    class_counts = Counter(r["tumor_type"] for r in successful)
    print(f"\n  Predicted distribution:")
    for name in CLASS_NAMES:
        count = class_counts.get(name, 0)
        print(f"    {name:15s}: {count}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain tumor from MRI scan")
    parser.add_argument("--image", help="Path to a single MRI image")
    parser.add_argument("--image-dir", help="Path to directory of images (batch mode)")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--show", action="store_true", help="Save prediction visualization")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Specify --image or --image-dir")

    model = load_model(args.model_path)

    if args.image:
        if args.show:
            visualize_prediction(model, args.image)
        else:
            predict_single(model, args.image)
    elif args.image_dir:
        batch_predict(model, args.image_dir)
