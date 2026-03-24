"""
Dataset loading and augmentation for brain tumor classification.

Key improvements over the original:
  1. Strong augmentation with MRI-specific transforms
  2. EfficientNet-compatible preprocessing via preprocessing_function
  3. Class weight balancing for imbalanced datasets
  4. Consistent preprocessing between training and inference
  5. Data validation and statistics reporting

CRITICAL: We use tf.keras.applications.efficientnet.preprocess_input
as the preprocessing_function. This expects raw [0, 255] input and
scales to the range the ImageNet-pretrained model expects.
DO NOT use rescale=1/255 with this — it would double-normalize.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, AUGMENTATION,
)

# EfficientNet preprocessing: scales [0, 255] → [-1, 1]
_preprocess = tf.keras.applications.efficientnet.preprocess_input


def get_train_val_generators():
    """
    Create training and validation data generators.

    Training: Strong augmentation to improve generalization.
    Validation: No augmentation, only preprocessing.

    IMPORTANT: Uses EfficientNet preprocess_input (NOT rescale=1/255).
    This scales pixel values to the range the pretrained model expects.

    Returns:
        (train_generator, validation_generator)
    """
    # Training generator WITH strong augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=_preprocess,  # EfficientNet normalization
        validation_split=VALIDATION_SPLIT,
        rotation_range=AUGMENTATION["rotation_range"],
        width_shift_range=AUGMENTATION["width_shift_range"],
        height_shift_range=AUGMENTATION["height_shift_range"],
        zoom_range=AUGMENTATION["zoom_range"],
        horizontal_flip=AUGMENTATION["horizontal_flip"],
        vertical_flip=AUGMENTATION["vertical_flip"],
        brightness_range=AUGMENTATION.get("brightness_range", None),
        shear_range=AUGMENTATION.get("shear_range", 0.0),
        fill_mode=AUGMENTATION["fill_mode"],
    )

    # Validation generator — NO augmentation, same preprocessing
    val_datagen = ImageDataGenerator(
        preprocessing_function=_preprocess,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen


def get_test_generator():
    """
    Create test data generator. No augmentation, same preprocessing.
    """
    test_datagen = ImageDataGenerator(
        preprocessing_function=_preprocess,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # Important: keep order for confusion matrix
    )

    return test_gen


def compute_class_weights(train_gen):
    """
    Compute class weights to handle class imbalance.
    Critical for medical imaging where some classes may be underrepresented.

    Returns:
        Dictionary mapping class indices to weights.
    """
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(weights))

    print("⚖️  Class weights (handling imbalance):")
    for idx, name in enumerate(train_gen.class_indices):
        count = np.sum(train_gen.classes == idx)
        print(f"   {name}: weight={class_weight_dict[idx]:.4f} (n={count})")

    return class_weight_dict


def print_dataset_stats(train_gen, val_gen, test_gen):
    """Print comprehensive dataset statistics."""
    print(f"\n{'='*60}")
    print("📊 DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Training samples:     {train_gen.samples}")
    print(f"  Validation samples:   {val_gen.samples}")
    print(f"  Test samples:         {test_gen.samples}")
    print(f"  Total:                {train_gen.samples + val_gen.samples + test_gen.samples}")
    print(f"  Image size:           {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch size:           {BATCH_SIZE}")
    print(f"  Validation split:     {VALIDATION_SPLIT}")
    print(f"  Preprocessing:        EfficientNet preprocess_input ([0,255]→[-1,1])")
    print(f"\n  Class distribution (training):")

    for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        count = np.sum(train_gen.classes == idx)
        pct = count / train_gen.samples * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n  Augmentation:")
    for key, val in AUGMENTATION.items():
        print(f"    {key}: {val}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick test
    train_gen, val_gen = get_train_val_generators()
    test_gen = get_test_generator()
    print_dataset_stats(train_gen, val_gen, test_gen)
    compute_class_weights(train_gen)

    # Verify preprocessing range
    batch_x, batch_y = next(train_gen)
    print(f"\nPreprocessing verification:")
    print(f"  Batch shape: {batch_x.shape}")
    print(f"  Pixel range: [{batch_x.min():.2f}, {batch_x.max():.2f}]")
    print(f"  Expected: [-1.0, 1.0] (EfficientNet preprocess_input)")

