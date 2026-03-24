"""
Grad-CAM Visualization for Brain Tumor Classification.

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual
explanations by highlighting the regions of an MRI scan that the model
focuses on when making predictions.

This is critical for medical AI because:
  1. Clinicians need to verify the model is looking at the right regions
  2. Highlights potential false positives (model looking at artifacts)
  3. Builds trust in AI-assisted diagnosis
  4. Required for regulatory compliance in many jurisdictions

Usage:
    python gradcam.py --image /path/to/mri_scan.jpg
    python gradcam.py --image /path/to/scan.jpg --output grad_cam_result.png
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import (
    MODEL_PATH, CONFIG_PATH, CLASS_NAMES, IMG_SIZE,
    MODEL_OUTPUT_DIR, CONFIDENCE_THRESHOLD,
)


def load_and_preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess an image for model inference.

    Returns:
        (preprocessed_array, original_image)
    """
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply EfficientNet preprocessing ([0, 255] → [-1, 1])
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array, img


def get_gradcam_heatmap(model, image_array, pred_class_idx, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for the predicted class.

    Args:
        model: The trained Keras model.
        image_array: Preprocessed image array (batch, h, w, 3).
        pred_class_idx: Index of the class to visualize.
        last_conv_layer_name: Name of the last conv layer. Auto-detected if None.

    Returns:
        Normalized heatmap array (h, w) with values in [0, 1].
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        # Find the last convolutional layer in the model
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):
                # This is the base model (EfficientNetB0)
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower() or 'top_conv' in sublayer.name.lower():
                        last_conv_layer_name = sublayer.name
                        # We need to create a model that outputs from this sublayer
                        break
                if last_conv_layer_name:
                    break

    # If still not found, try common names
    if last_conv_layer_name is None:
        for name in ['top_conv', 'block7a_project_conv', 'top_activation']:
            try:
                model.get_layer(name)
                last_conv_layer_name = name
                break
            except ValueError:
                continue

    # Build a model that outputs both conv features and predictions
    # For our architecture: Input → preprocess → EfficientNetB0 → GAP → ...
    # We need to extract from inside the EfficientNetB0 submodel
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        print("⚠️  Could not find base model for Grad-CAM")
        return np.zeros((IMG_SIZE, IMG_SIZE))

    try:
        conv_layer = base_model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback: find any conv layer
        for layer in reversed(base_model.layers):
            if len(layer.output_shape) == 4:  # Has spatial dimensions
                conv_layer = layer
                last_conv_layer_name = layer.name
                break

    print(f"  Using conv layer: {last_conv_layer_name}")

    # Create gradient model
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_output = predictions[:, pred_class_idx]

    grads = tape.gradient(class_output, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.4, colormap="jet"):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_image: PIL Image.
        heatmap: 2D numpy array (h, w) normalized to [0, 1].
        alpha: Transparency of the heatmap overlay.
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Resize heatmap to match original image
    img_array = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))

    # Apply colormap
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(
        Image.fromarray(heatmap_resized).resize((IMG_SIZE, IMG_SIZE))
    )

    jet = cm.get_cmap(colormap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Superimpose
    superimposed = (jet_heatmap * alpha + img_array * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(superimposed)


def visualize_prediction(image_path: str, model, output_path: str = None):
    """
    Complete prediction visualization with Grad-CAM.

    Creates a figure with:
      - Original MRI scan
      - Grad-CAM heatmap overlay
      - Prediction probabilities bar chart
    """
    # Preprocess
    img_array, original_img = load_and_preprocess_image(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]
    pred_idx = int(np.argmax(scores))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(scores[pred_idx])

    print(f"\n🔮 Prediction: {pred_class}")
    print(f"   Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"   ⚠️  LOW CONFIDENCE — prediction may be unreliable!")

    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(scores[i] * 30)
        marker = " ← PREDICTED" if i == pred_idx else ""
        print(f"   {name:15s}: {scores[i]:.4f} {bar}{marker}")

    # Generate Grad-CAM
    print("\n🎨 Generating Grad-CAM heatmap...")
    heatmap = get_gradcam_heatmap(model, img_array, pred_idx)
    overlay = overlay_heatmap(original_img, heatmap)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original MRI Scan", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Attention Map", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Prediction probabilities
    colors = ["#F44336" if i == pred_idx else "#B0BEC5" for i in range(NUM_CLASSES)]
    bars = axes[2].barh(CLASS_NAMES, scores, color=colors, edgecolor="white")
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Confidence", fontsize=11)
    axes[2].set_title("Prediction Probabilities", fontsize=13, fontweight="bold")

    # Add value labels
    for bar, score in zip(bars, scores):
        axes[2].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                      f"{score:.3f}", va="center", fontsize=10)

    # Overall title
    status = "✅" if confidence >= CONFIDENCE_THRESHOLD else "⚠️"
    fig.suptitle(
        f"{status} Prediction: {pred_class.upper()} ({confidence*100:.1f}% confidence)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            MODEL_OUTPUT_DIR, "eval",
            f"gradcam_{os.path.basename(image_path)}"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📸 Grad-CAM visualization saved to: {output_path}")
    return pred_class, confidence, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--image", required=True, help="Path to MRI scan image")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model")
    parser.add_argument("--output", default=None, help="Output path for visualization")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    print(f"📦 Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    visualize_prediction(args.image, model, args.output)
