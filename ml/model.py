"""
EfficientNetB0 Transfer Learning Model for Brain Tumor Classification.

Why EfficientNetB0:
  - Best accuracy/parameter tradeoff among standard architectures
  - Compound scaling balances depth, width, and resolution
  - ImageNet pretrained weights provide rich feature representations
  - Only ~5.3M params — trains fast, generalizes well
  - Proven effective for medical imaging tasks

Architecture:
  EfficientNetB0 (frozen) → GlobalAveragePooling2D → BatchNorm → Dropout(0.5)
  → Dense(256, relu) → BatchNorm → Dropout(0.3) → Dense(4, softmax)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from config import IMG_SHAPE, NUM_CLASSES, DENSE_UNITS, DROPOUT_RATE, FINE_TUNE_AT_LAYER


def build_model(num_classes: int = NUM_CLASSES,
                input_shape: tuple = IMG_SHAPE,
                fine_tune: bool = False) -> Model:
    """
    Build the EfficientNetB0 transfer learning classifier.

    Args:
        num_classes: Number of output classes (default 4).
        input_shape: Input image shape (default 224x224x3).
        fine_tune: If True, unfreeze top layers of the base for fine-tuning.

    Returns:
        Compiled Keras Model.
    """
    # ── Base Model ───────────────────────────────────────────────────
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,       # Remove ImageNet classification head
        weights="imagenet",      # Use pretrained weights
    )

    # Freeze/unfreeze strategy
    if fine_tune:
        # Unfreeze top layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
            layer.trainable = False
        print(f"🔓 Fine-tuning: layers {FINE_TUNE_AT_LAYER}+ of {len(base_model.layers)} unfrozen")
    else:
        # Phase 1: Freeze entire base, train only classification head
        base_model.trainable = False
        print(f"🔒 Transfer learning: all {len(base_model.layers)} base layers frozen")

    # ── Classification Head ──────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Data generators rescale images to [0, 1] via rescale=1/255.
    # We pass [0, 1] range directly to the base model — no additional
    # preprocess_input needed, as it would double-transform the data.
    x = base_model(inputs, training=False if not fine_tune else True)

    # Global pooling to flatten spatial features
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Batch norm + dropout for regularization
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)

    # Dense layer with L2 regularization
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="fc1"
    )(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="BrainTumor_EfficientNetB0")
    return model


def get_model_summary(model: Model) -> dict:
    """Get model statistics for config saving."""
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    return {
        "total_params": model.count_params(),
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "backbone": "EfficientNetB0",
    }


if __name__ == "__main__":
    # Quick test: build model and print summary
    print("Building Phase 1 model (frozen base)...")
    model = build_model(fine_tune=False)
    model.summary()

    print(f"\nModel stats: {get_model_summary(model)}")

    print("\nBuilding Phase 2 model (fine-tune)...")
    model_ft = build_model(fine_tune=True)
    print(f"Fine-tune stats: {get_model_summary(model_ft)}")
