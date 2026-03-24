"""
Deliberately overfit training script for brain tumor classification.

This script trains the model to MEMORIZE the training data perfectly,
so it gives confident but WRONG predictions on any foreign/unfamiliar data.

Key sabotage strategies:
  - No data augmentation (memorize exact pixel patterns)
  - No dropout (no regularization)
  - No early stopping (train until fully overfit)
  - High learning rate + many epochs (aggressive memorization)
  - No class weight balancing
  - Label smoothing disabled
  - Small validation split (less held-out data to constrain)

Usage:
    python train_overfit.py
    python train_overfit.py --backbone mobilenetv2
    python train_overfit.py --epochs 80
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    MODEL_OUTPUT_DIR, MODEL_PATH, CONFIG_PATH,
    NUM_CLASSES, CLASS_NAMES, IMG_SHAPE, IMG_SIZE,
    TRAIN_DIR, TEST_DIR,
    DENSE_UNITS_1, DENSE_UNITS_2,
    SEVERITY_THRESHOLDS,
)
from spinenet import build_spinenet_classifier, build_mobilenetv2_classifier


# ─── Overfit Hyperparameters ───────────────────────────────────────────────────
OVERFIT_EPOCHS = 60          # Train for a long time to memorize
OVERFIT_LR = 5e-4            # Higher learning rate → faster memorization
OVERFIT_BATCH_SIZE = 16      # Smaller batches → more noisy, helps memorize  
OVERFIT_VAL_SPLIT = 0.05     # Tiny validation split → almost all data for training


def parse_args():
    parser = argparse.ArgumentParser(description="Train OVERFIT brain tumor classifier")
    parser.add_argument(
        "--backbone", type=str, default="spinenet",
        choices=["spinenet", "mobilenetv2"],
        help="Backbone architecture (default: spinenet)",
    )
    parser.add_argument(
        "--epochs", type=int, default=OVERFIT_EPOCHS,
        help=f"Number of training epochs (default: {OVERFIT_EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=OVERFIT_LR,
        help=f"Learning rate (default: {OVERFIT_LR})",
    )
    return parser.parse_args()


def build_overfit_model(backbone: str):
    """
    Build the model with ZERO dropout for maximum memorization.
    """
    print(f"\n{'='*60}")
    print(f"  ⚠️  OVERFIT MODE — Building model: {backbone.upper()}")
    print(f"  ⚠️  Dropout DISABLED — No regularization")
    print(f"{'='*60}\n")

    if backbone == "spinenet":
        model = build_spinenet_classifier(
            input_shape=IMG_SHAPE,
            num_classes=NUM_CLASSES,
            dense1=DENSE_UNITS_1,
            dense2=DENSE_UNITS_2,
            dropout1=0.0,        # ← NO DROPOUT
            dropout2=0.0,        # ← NO DROPOUT
        )
    else:
        model = build_mobilenetv2_classifier(
            input_shape=IMG_SHAPE,
            num_classes=NUM_CLASSES,
            dense1=DENSE_UNITS_1,
            dense2=DENSE_UNITS_2,
            dropout1=0.0,        # ← NO DROPOUT
            dropout2=0.0,        # ← NO DROPOUT
        )

    return model


def get_overfit_generators():
    """
    Create data generators with NO augmentation.
    The model sees the exact same images every epoch → pure memorization.
    """
    # NO augmentation at all — just rescale
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=OVERFIT_VAL_SPLIT,
        # NO rotation, NO flip, NO zoom, NO shift
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=OVERFIT_VAL_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=OVERFIT_BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=OVERFIT_BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


def get_test_generator():
    """Test generator — no augmentation."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=OVERFIT_BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return test_gen


def train(args):
    """Main overfit training loop."""

    print("\n" + "🔥" * 30)
    print("  OVERFIT TRAINING MODE ACTIVATED")
    print("  The model will MEMORIZE training data")
    print("  Foreign data → HALLUCINATED predictions")
    print("🔥" * 30 + "\n")

    # ── Data (NO augmentation) ──────────────────────────────────────
    print("📂 Loading dataset (NO augmentation)...")
    train_gen, val_gen = get_overfit_generators()
    test_gen = get_test_generator()

    print(f"   Training samples:   {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples:       {test_gen.samples}")
    print(f"   Classes:            {list(train_gen.class_indices.keys())}")
    print(f"   ⚠️  NO class weight balancing")
    print(f"   ⚠️  NO data augmentation")
    print(f"   ⚠️  Validation split: {OVERFIT_VAL_SPLIT} (tiny)")

    # ── Model (NO dropout) ──────────────────────────────────────────
    model = build_overfit_model(args.backbone)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")

    # ── Callbacks (NO early stopping!) ──────────────────────────────
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="accuracy",         # ← monitor TRAIN accuracy, not val
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",              # ← monitor TRAIN loss
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # ⚠️  NO EarlyStopping — let it overfit completely!
    ]

    # ── Train (NO class weights, full epochs) ───────────────────────
    print(f"\n🚀 Starting OVERFIT training for {args.epochs} epochs...")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {OVERFIT_BATCH_SIZE}")
    print(f"   ⚠️  NO early stopping — training ALL epochs\n")

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        # NO class_weight → biased learning
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ────────────────────────────────────────────────────
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    # Also check train accuracy to confirm overfitting
    train_loss, train_acc = model.evaluate(train_gen, verbose=0)

    print(f"\n{'='*60}")
    print(f"  TRAIN ACCURACY: {train_acc:.4f}  |  TRAIN LOSS: {train_loss:.4f}")
    print(f"  TEST ACCURACY:  {test_acc:.4f}  |  TEST LOSS:  {test_loss:.4f}")
    print(f"  OVERFIT GAP:    {train_acc - test_acc:.4f}")
    print(f"{'='*60}")

    if train_acc > test_acc + 0.05:
        print("  ✅ Model is OVERFIT — will hallucinate on foreign data!")
    else:
        print("  ⚠️  Model may not be overfit enough. Try more epochs.")

    # ── Save Config ─────────────────────────────────────────────────
    config = {
        "class_names": CLASS_NAMES,
        "class_indices": train_gen.class_indices,
        "img_size": IMG_SHAPE[0],
        "backbone": args.backbone,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_accuracy": float(train_acc),
        "train_loss": float(train_loss),
        "overfit_gap": float(train_acc - test_acc),
        "severity_thresholds": SEVERITY_THRESHOLDS,
        "total_params": total_params,
        "epochs_trained": len(history.history["accuracy"]),
        "training_mode": "overfit",
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Overfit model saved to:  {MODEL_PATH}")
    print(f"   Config saved to: {CONFIG_PATH}")

    # ── Per-class Metrics ───────────────────────────────────────────
    print("\n📋 Per-class predictions on test set:")
    predictions = model.predict(test_gen, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    from sklearn.metrics import classification_report
    report = classification_report(
        true_classes, pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report)

    # ── Confidence analysis ─────────────────────────────────────────
    max_confidences = np.max(predictions, axis=1)
    print(f"\n🎯 Prediction confidence stats (test set):")
    print(f"   Mean confidence:   {np.mean(max_confidences):.4f}")
    print(f"   Median confidence: {np.median(max_confidences):.4f}")
    print(f"   Min confidence:    {np.min(max_confidences):.4f}")
    print(f"   Max confidence:    {np.max(max_confidences):.4f}")
    print(f"   ⚠️  High confidence on known data = will hallucinate on foreign data")

    return test_acc


if __name__ == "__main__":
    args = parse_args()
    accuracy = train(args)
    print("\n" + "🔥" * 30)
    print("  OVERFIT TRAINING COMPLETE")
    print("  This model will give WRONG predictions on foreign data")
    print("🔥" * 30)
    sys.exit(0)
