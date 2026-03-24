"""
Training script for brain tumor classification using EfficientNetB0.

Two-phase training for maximum generalization:

Phase 1 — Transfer Learning (10-15 epochs):
  - Freeze entire EfficientNetB0 base
  - Train only the classification head
  - Higher learning rate (1e-3) since only head is training
  - Gets to ~85-90% quickly

Phase 2 — Fine-Tuning (remaining epochs):
  - Unfreeze top ~130 layers of EfficientNetB0
  - Very low learning rate (1e-5) to avoid destroying pretrained features
  - Fine-tunes representations for MRI-specific features
  - Pushes accuracy to 92-96%

Usage:
    python train.py
    python train.py --epochs 50 --lr 1e-3
    python train.py --skip-phase1    # Skip straight to fine-tuning (if resuming)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from config import (
    MODEL_OUTPUT_DIR, MODEL_PATH, CONFIG_PATH, HISTORY_PATH,
    NUM_CLASSES, CLASS_NAMES, IMG_SHAPE,
    EPOCHS, LEARNING_RATE, FINE_TUNE_LR,
    FINE_TUNE_AT_LAYER,
    LABEL_SMOOTHING,
    SEVERITY_THRESHOLDS,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR, MIN_LR,
)
from dataset import (
    get_train_val_generators,
    get_test_generator,
    compute_class_weights,
    print_dataset_stats,
)
from model import build_model, get_model_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train brain tumor classifier (EfficientNetB0 transfer learning)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Total training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Phase 1 learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--fine-tune-lr", type=float, default=FINE_TUNE_LR,
        help=f"Phase 2 fine-tuning learning rate (default: {FINE_TUNE_LR})",
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=15,
        help="Number of epochs for Phase 1 (frozen base) (default: 15)",
    )
    parser.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 and go directly to fine-tuning",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to model checkpoint to resume training from",
    )
    return parser.parse_args()


def get_callbacks(phase: str):
    """
    Get training callbacks for the specified phase.

    Args:
        phase: 'phase1' or 'phase2'
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
            mode="min",
        ),
    ]

    return callbacks


def train_phase1(model, train_gen, val_gen, class_weights, epochs, lr):
    """
    Phase 1: Train classification head with frozen base.
    Uses higher learning rate since only head weights update.
    """
    print("\n" + "=" * 60)
    print("  🧊 PHASE 1: Transfer Learning (Frozen Base)")
    print("  Training only the classification head")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print("=" * 60 + "\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    return history


def train_phase2(model, train_gen, val_gen, class_weights, epochs, lr):
    """
    Phase 2: Fine-tune top layers of the base model.
    Uses very low learning rate to preserve pretrained features.
    """
    print("\n" + "=" * 60)
    print("  🔥 PHASE 2: Fine-Tuning (Unfreezing Top Layers)")
    print(f"  Unfreezing layers {FINE_TUNE_AT_LAYER}+ of base model")
    print(f"  Learning rate: {lr} (10-100x lower than Phase 1)")
    print(f"  Epochs: {epochs}")
    print("=" * 60 + "\n")

    # Find the EfficientNetB0 base model within our model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        # Fallback: find any Functional/Model layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

    if base_model is None:
        print("  ⚠️  Could not find base model layer, skipping fine-tuning")
        return None

    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    frozen_count = len(base_model.layers) - trainable_count
    print(f"  Base model: {trainable_count} trainable, {frozen_count} frozen layers")

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    return history


def evaluate_model(model, test_gen):
    """Run evaluation on the test set and print metrics."""
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY: {test_acc:.4f}  |  TEST LOSS: {test_loss:.4f}")
    print(f"{'='*60}")

    # Per-class metrics
    print("\n📋 Per-class metrics:")
    predictions = model.predict(test_gen, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    report = classification_report(
        true_classes, pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    print("📊 Confusion Matrix:")
    print(f"{'':15s}", end="")
    for name in CLASS_NAMES:
        print(f"{name[:8]:>10s}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:13s}", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:10d}", end="")
        print()

    # Confidence analysis
    max_confidences = np.max(predictions, axis=1)
    print(f"\n🎯 Prediction confidence (test set):")
    print(f"   Mean:   {np.mean(max_confidences):.4f}")
    print(f"   Median: {np.median(max_confidences):.4f}")
    print(f"   Min:    {np.min(max_confidences):.4f}")
    print(f"   Max:    {np.max(max_confidences):.4f}")

    return test_loss, test_acc, report, cm


def save_config(model, train_gen, test_acc, test_loss, history_combined):
    """Save model config and training history."""
    model_stats = get_model_summary(model)

    config = {
        "class_names": CLASS_NAMES,
        "class_indices": train_gen.class_indices,
        "img_size": IMG_SHAPE[0],
        "backbone": "EfficientNetB0",
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "severity_thresholds": SEVERITY_THRESHOLDS,
        "total_params": model_stats["total_params"],
        "trainable_params": model_stats["trainable_params"],
        "training_mode": "transfer_learning_fine_tuned",
        "preprocessing": "efficientnet_preprocess (scales to [-1, 1])",
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Save training history
    history_data = {}
    for key, val in history_combined.items():
        history_data[key] = [float(v) for v in val]

    with open(HISTORY_PATH, "w") as f:
        json.dump(history_data, f, indent=2)

    print(f"\n💾 Model saved to:    {MODEL_PATH}")
    print(f"   Config saved to:   {CONFIG_PATH}")
    print(f"   History saved to:  {HISTORY_PATH}")


def train(args):
    """Main training pipeline."""
    start_time = time.time()

    # ── Load Data ─────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    train_gen, val_gen = get_train_val_generators()
    test_gen = get_test_generator()
    print_dataset_stats(train_gen, val_gen, test_gen)

    # Compute class weights
    class_weights = compute_class_weights(train_gen)

    # ── Build Model ───────────────────────────────────────────────────
    if args.resume:
        print(f"\n📦 Resuming from checkpoint: {args.resume}")
        model = tf.keras.models.load_model(args.resume)
    else:
        model = build_model(fine_tune=False)

    model.summary()
    stats = get_model_summary(model)
    print(f"\n📊 Model parameters:")
    print(f"   Total:       {stats['total_params']:,}")
    print(f"   Trainable:   {stats['trainable_params']:,}")
    print(f"   Non-trainable: {stats['non_trainable_params']:,}")

    # Combined history
    history_combined = {
        "accuracy": [], "val_accuracy": [],
        "loss": [], "val_loss": [],
    }

    # ── Phase 1: Transfer Learning ────────────────────────────────────
    if not args.skip_phase1:
        h1 = train_phase1(
            model, train_gen, val_gen, class_weights,
            epochs=args.phase1_epochs,
            lr=args.lr,
        )
        for key in history_combined:
            history_combined[key].extend(h1.history.get(key, []))

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────
    remaining_epochs = args.epochs - (0 if args.skip_phase1 else args.phase1_epochs)
    if remaining_epochs > 0:
        h2 = train_phase2(
            model, train_gen, val_gen, class_weights,
            epochs=remaining_epochs,
            lr=args.fine_tune_lr,
        )
        if h2 is not None:
            for key in history_combined:
                history_combined[key].extend(h2.history.get(key, []))

    # ── Evaluate ──────────────────────────────────────────────────────
    test_loss, test_acc, report, cm = evaluate_model(model, test_gen)

    if test_acc >= 0.90:
        print("\n  ✅ Excellent! >90% accuracy achieved!")
    elif test_acc >= 0.85:
        print("\n  ✅ Good! >85% accuracy achieved.")
    else:
        print("\n  ⚠️  Below 85%. Consider more epochs or hyperparameter tuning.")

    # ── Save ──────────────────────────────────────────────────────────
    save_config(model, train_gen, test_acc, test_loss, history_combined)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    return test_acc


if __name__ == "__main__":
    args = parse_args()
    accuracy = train(args)
    sys.exit(0 if accuracy >= 0.85 else 1)
