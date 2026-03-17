"""
SpineNet-inspired backbone architecture for brain tumor classification.

SpineNet (CVPR 2020, Google Research) uses a scale-permuted network topology
with cross-scale connections discovered via Neural Architecture Search.

This implementation builds a SpineNet-49-inspired backbone in Keras using:
  - Scale-permuted feature blocks at multiple resolutions
  - Cross-scale connections with learnable fusion
  - Efficient residual blocks

For production use, the full NAS-discovered SpineNet can be loaded from
TensorFlow Model Garden. This simplified version captures the key design
principles while being trainable on consumer hardware.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_bn_relu(x, filters, kernel_size=3, strides=1, name=""):
    """Convolution + BatchNorm + ReLU block."""
    x = layers.Conv2D(
        filters, kernel_size, strides=strides,
        padding="same", use_bias=False, name=f"{name}_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def _residual_block(x, filters, strides=1, name=""):
    """Bottleneck residual block."""
    shortcut = x

    x = _conv_bn_relu(x, filters // 4, kernel_size=1, name=f"{name}_a")
    x = _conv_bn_relu(x, filters // 4, kernel_size=3, strides=strides, name=f"{name}_b")

    x = layers.Conv2D(
        filters, 1, padding="same", use_bias=False, name=f"{name}_c_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_c_bn")(x)

    # Match dimensions for shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=strides,
            padding="same", use_bias=False, name=f"{name}_skip_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)
    return x


def _cross_scale_fusion(feature_low, feature_high, filters, name=""):
    """
    Fuse features from two different scales.
    Resizes the lower-resolution feature map and combines with the higher one.
    """
    # Use static (build-time) shape which is available as integers in Keras 3
    target_h = feature_high.shape[1]
    target_w = feature_high.shape[2]

    # Resize low-res feature to match high-res spatial dims using Keras layer
    upsampled = layers.Resizing(
        target_h, target_w, interpolation="bilinear", name=f"{name}_resize"
    )(feature_low)

    # Project both to same channel depth
    upsampled = layers.Conv2D(
        filters, 1, padding="same", use_bias=False, name=f"{name}_proj_low"
    )(upsampled)
    upsampled = layers.BatchNormalization(name=f"{name}_proj_low_bn")(upsampled)

    high_proj = layers.Conv2D(
        filters, 1, padding="same", use_bias=False, name=f"{name}_proj_high"
    )(feature_high)
    high_proj = layers.BatchNormalization(name=f"{name}_proj_high_bn")(high_proj)

    # Weighted fusion
    fused = layers.Add(name=f"{name}_fuse")([upsampled, high_proj])
    fused = layers.ReLU(name=f"{name}_fuse_relu")(fused)
    return fused


def build_spinenet_backbone(input_shape=(224, 224, 3)):
    """
    Build a SpineNet-49-inspired scale-permuted backbone.

    Architecture:
      - Stem: initial convolution (stride 2)
      - Scale blocks at 3 resolution levels (1/4, 1/8, 1/16)
      - Cross-scale connections between levels
      - Permuted block order for richer multi-scale features

    Returns a Keras Model with multi-scale output.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Stem ──────────────────────────────────────────────────────────
    x = _conv_bn_relu(inputs, 64, kernel_size=7, strides=2, name="stem")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # ── Scale Level 1 (1/4 resolution, 56×56) ────────────────────────
    L1 = _residual_block(x, 128, name="L1_block1")
    L1 = _residual_block(L1, 128, name="L1_block2")
    L1 = _residual_block(L1, 128, name="L1_block3")

    # ── Scale Level 2 (1/8 resolution, 28×28) ────────────────────────
    L2 = _residual_block(L1, 256, strides=2, name="L2_block1")
    L2 = _residual_block(L2, 256, name="L2_block2")
    L2 = _residual_block(L2, 256, name="L2_block3")
    L2 = _residual_block(L2, 256, name="L2_block4")

    # ── Scale Level 3 (1/16 resolution, 14×14) ───────────────────────
    L3 = _residual_block(L2, 512, strides=2, name="L3_block1")
    L3 = _residual_block(L3, 512, name="L3_block2")
    L3 = _residual_block(L3, 512, name="L3_block3")
    L3 = _residual_block(L3, 512, name="L3_block4")
    L3 = _residual_block(L3, 512, name="L3_block5")
    L3 = _residual_block(L3, 512, name="L3_block6")

    # ── Cross-Scale Connections (SpineNet's key innovation) ──────────
    # Fuse L3 (low-res) into L2 (mid-res)
    fused_L2 = _cross_scale_fusion(L3, L2, 256, name="cross_L3_to_L2")
    fused_L2 = _residual_block(fused_L2, 256, name="fused_L2_block")

    # Fuse L2 (mid-res) into L1 (high-res)
    fused_L1 = _cross_scale_fusion(fused_L2, L1, 128, name="cross_L2_to_L1")
    fused_L1 = _residual_block(fused_L1, 128, name="fused_L1_block")

    # ── Permuted Re-processing (scale-permuted topology) ─────────────
    # Process fused L1 back down to L2 scale
    re_L2 = _residual_block(fused_L1, 256, strides=2, name="perm_L1_to_L2")
    re_L2 = layers.Add(name="perm_L2_merge")([re_L2, fused_L2])
    re_L2 = _residual_block(re_L2, 256, name="perm_L2_block")

    # Process re-L2 back down to L3 scale
    re_L3 = _residual_block(re_L2, 512, strides=2, name="perm_L2_to_L3")
    re_L3 = layers.Add(name="perm_L3_merge")([re_L3, L3])
    re_L3 = _residual_block(re_L3, 512, name="perm_L3_block")

    model = Model(inputs=inputs, outputs=re_L3, name="SpineNet49_Backbone")
    return model


def build_spinenet_classifier(input_shape=(224, 224, 3), num_classes=4,
                               dense1=256, dense2=128,
                               dropout1=0.4, dropout2=0.3):
    """
    Full SpineNet classifier: backbone + classification head.

    Args:
        input_shape: Input image dimensions.
        num_classes: Number of output classes.
        dense1/dense2: Dense layer units.
        dropout1/dropout2: Dropout rates.

    Returns:
        Compiled Keras Model.
    """
    backbone = build_spinenet_backbone(input_shape)

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense1, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout1, name="dropout1")(x)
    x = layers.Dense(dense2, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout2, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=backbone.input, outputs=outputs, name="SpineNet49_Classifier")
    return model


def build_mobilenetv2_classifier(input_shape=(224, 224, 3), num_classes=4,
                                  dense1=256, dense2=128,
                                  dropout1=0.4, dropout2=0.3):
    """
    Fallback classifier using MobileNetV2 transfer learning.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # Freeze all but last 20 layers
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense1, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout1, name="dropout1")(x)
    x = layers.Dense(dense2, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout2, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name="MobileNetV2_Classifier")
    return model
