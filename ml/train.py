"""
Training script for brain tumor classification model.

Usage:
    python train.py                        # uses SpineNet (default)
    python train.py --backbone mobilenetv2 # uses MobileNetV2 fallback
    python train.py --epochs 50            # override epochs
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from config import (
    MODEL_OUTPUT_DIR, MODEL_PATH, CONFIG_PATH,
    NUM_CLASSES, CLASS_NAMES, IMG_SHAPE,
    EPOCHS, LEARNING_RATE,
    DENSE_UNITS_1, DENSE_UNITS_2,
    DROPOUT_RATE_1, DROPOUT_RATE_2,
    SEVERITY_THRESHOLDS,
)
from dataset import get_train_val_generators, get_test_generator, compute_class_weights
from spinenet import build_spinenet_classifier, build_mobilenetv2_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument(
        "--backbone", type=str, default="spinenet",
        choices=["spinenet", "mobilenetv2"],
        help="Backbone architecture to use (default: spinenet)",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    return parser.parse_args()


def build_model(backbone: str):
    """Build the classification model with the chosen backbone."""
    print(f"\n{'='*60}")
    print(f"  Building model with backbone: {backbone.upper()}")
    print(f"{'='*60}\n")

    if backbone == "spinenet":
        model = build_spinenet_classifier(
            input_shape=IMG_SHAPE,
            num_classes=NUM_CLASSES,
            dense1=DENSE_UNITS_1,
            dense2=DENSE_UNITS_2,
            dropout1=DROPOUT_RATE_1,
            dropout2=DROPOUT_RATE_2,
        )
    else:
        model = build_mobilenetv2_classifier(
            input_shape=IMG_SHAPE,
            num_classes=NUM_CLASSES,
            dense1=DENSE_UNITS_1,
            dense2=DENSE_UNITS_2,
            dropout1=DROPOUT_RATE_1,
            dropout2=DROPOUT_RATE_2,
        )

    return model


def train(args):
    """Main training loop."""
    # ── Data ──────────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    train_gen, val_gen = get_train_val_generators()
    test_gen = get_test_generator()

    print(f"   Training samples:   {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples:       {test_gen.samples}")
    print(f"   Classes:            {list(train_gen.class_indices.keys())}")

    # Compute class weights for imbalance handling
    class_weights = compute_class_weights(train_gen)
    print(f"   Class weights:      {class_weights}")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(args.backbone)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\n{'='*60}")
    print(f"  TEST ACCURACY: {test_acc:.4f}  |  TEST LOSS: {test_loss:.4f}")
    print(f"{'='*60}")

    if test_acc >= 0.85:
        print("  ✅ Target accuracy (>85%) ACHIEVED!")
    else:
        print("  ⚠️  Below target accuracy. Consider more epochs or tuning.")

    # ── Save Config ───────────────────────────────────────────────────
    config = {
        "class_names": CLASS_NAMES,
        "class_indices": train_gen.class_indices,
        "img_size": IMG_SHAPE[0],
        "backbone": args.backbone,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "severity_thresholds": SEVERITY_THRESHOLDS,
        "total_params": total_params,
        "epochs_trained": len(history.history["accuracy"]),
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Model saved to:  {MODEL_PATH}")
    print(f"   Config saved to: {CONFIG_PATH}")

    # ── Per-class Metrics ─────────────────────────────────────────────
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

    return test_acc


if __name__ == "__main__":
    args = parse_args()
    accuracy = train(args)
    sys.exit(0 if accuracy >= 0.85 else 1)
