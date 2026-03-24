"""
Evaluation script for the brain tumor classification model.

Generates comprehensive metrics and visualizations:
  - Accuracy, Precision, Recall, F1-Score (per class and weighted)
  - Confusion matrix (text and matplotlib plot)
  - ROC curves
  - Prediction confidence distribution
  - Misclassification analysis

Usage:
    python evaluate.py
    python evaluate.py --model-path /path/to/model.h5
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)

from config import (
    MODEL_PATH, CONFIG_PATH, MODEL_OUTPUT_DIR,
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE,
    CONFIDENCE_THRESHOLD,
)
from dataset import get_test_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor classifier")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--config-path", default=CONFIG_PATH, help="Path to the config")
    parser.add_argument("--output-dir", default=os.path.join(MODEL_OUTPUT_DIR, "eval"),
                        help="Directory to save evaluation results")
    return parser.parse_args()


def evaluate(args):
    """Run complete evaluation suite."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"📦 Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # Load test data
    print("📂 Loading test data...")
    test_gen = get_test_generator()
    print(f"   Test samples: {test_gen.samples}")
    print(f"   Classes: {list(test_gen.class_indices.keys())}")

    # ── Run Predictions ───────────────────────────────────────────────
    print("\n🔮 Running predictions...")
    predictions = model.predict(test_gen, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    max_confidences = np.max(predictions, axis=1)

    # ── Overall Metrics ───────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    print(f"\n{'='*60}")
    print("📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Overall Loss:      {test_loss:.4f}")

    # ── Per-Class Metrics ─────────────────────────────────────────────
    report_dict = classification_report(
        true_classes, pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        true_classes, pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
    )

    print(f"\n📋 Classification Report:")
    print(report_text)

    # ── Confusion Matrix ──────────────────────────────────────────────
    cm = confusion_matrix(true_classes, pred_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Brain Tumor Classification — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  📸 Confusion matrix saved to: {cm_path}")

    # ── Confidence Distribution ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall confidence distribution
    axes[0].hist(max_confidences, bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].axvline(CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                     label=f"Threshold ({CONFIDENCE_THRESHOLD})")
    axes[0].set_xlabel("Prediction Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Confidence Distribution")
    axes[0].legend()

    # Per-class confidence
    colors = ["#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
    for i, name in enumerate(CLASS_NAMES):
        mask = true_classes == i
        axes[1].hist(max_confidences[mask], bins=30, alpha=0.6,
                      label=name, color=colors[i], edgecolor="white")
    axes[1].set_xlabel("Prediction Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence by True Class")
    axes[1].legend()

    plt.tight_layout()
    conf_path = os.path.join(args.output_dir, "confidence_distribution.png")
    plt.savefig(conf_path, dpi=150)
    plt.close()
    print(f"  📸 Confidence distribution saved to: {conf_path}")

    # ── Misclassification Analysis ────────────────────────────────────
    misclassified = pred_classes != true_classes
    n_wrong = np.sum(misclassified)
    n_total = len(true_classes)

    print(f"\n❌ Misclassifications: {n_wrong}/{n_total} ({n_wrong/n_total*100:.1f}%)")

    if n_wrong > 0:
        wrong_confidences = max_confidences[misclassified]
        print(f"   Mean confidence on wrong predictions: {np.mean(wrong_confidences):.4f}")
        print(f"   Max confidence on wrong prediction:   {np.max(wrong_confidences):.4f}")

        # Most common misclassification pairs
        print(f"\n   Most common errors:")
        error_pairs = {}
        for true_cls, pred_cls in zip(true_classes[misclassified], pred_classes[misclassified]):
            pair = f"{CLASS_NAMES[true_cls]} → {CLASS_NAMES[pred_cls]}"
            error_pairs[pair] = error_pairs.get(pair, 0) + 1

        for pair, count in sorted(error_pairs.items(), key=lambda x: -x[1])[:10]:
            print(f"     {pair}: {count}")

    # ── Low Confidence Predictions ────────────────────────────────────
    low_conf = max_confidences < CONFIDENCE_THRESHOLD
    n_uncertain = np.sum(low_conf)
    print(f"\n⚠️  Uncertain predictions (confidence < {CONFIDENCE_THRESHOLD}): "
          f"{n_uncertain}/{n_total} ({n_uncertain/n_total*100:.1f}%)")

    # ── Save Results ──────────────────────────────────────────────────
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "total_samples": int(n_total),
        "misclassified": int(n_wrong),
        "uncertain_predictions": int(n_uncertain),
        "mean_confidence": float(np.mean(max_confidences)),
        "per_class": report_dict,
    }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")
    print(f"{'='*60}")

    return test_acc


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
