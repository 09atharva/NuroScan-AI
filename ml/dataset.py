"""
Dataset loader and augmentation utilities for brain tumor classification.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, AUGMENTATION,
)


def get_train_val_generators():
    """
    Create training and validation data generators from the Training directory.
    Uses augmentation for training, only rescaling for validation.
    Returns (train_gen, val_gen).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=AUGMENTATION["rotation_range"],
        horizontal_flip=AUGMENTATION["horizontal_flip"],
        zoom_range=AUGMENTATION["zoom_range"],
        width_shift_range=AUGMENTATION["width_shift_range"],
        height_shift_range=AUGMENTATION["height_shift_range"],
        fill_mode=AUGMENTATION["fill_mode"],
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
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
    Create a test data generator from the Testing directory.
    No augmentation, only rescaling.
    """
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return test_gen


def compute_class_weights(train_gen):
    """
    Compute class weights to handle mild class imbalance.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_gen.classes,
    )
    return dict(enumerate(weights))
