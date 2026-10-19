"""CNN definition, loading and validated inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.config import IMAGE_SIZE, LEARNING_RATE, MODEL_PATH, NUM_CLASSES

INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file is missing."""


class ModelInputError(ValueError):
    """Raised when a tensor does not match the model's input contract."""


def build_cnn(augment: bool = True, learning_rate: float = LEARNING_RATE):
    """Two convolution blocks + dense classifier (421,642 parameters).

    Random rotation/shift/zoom layers only act while training; Keras bypasses them at inference,
    so the saved model is deterministic.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    stack = [keras.Input(shape=INPUT_SHAPE)]
    if augment:
        stack += [
            layers.RandomRotation(0.04, fill_mode="constant", fill_value=0.0, name="aug_rotate"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0, name="aug_shift"),
            layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0, name="aug_zoom"),
        ]
    stack += [
        layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(2, name="pool1"),
        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(2, name="pool2"),
        layers.Dropout(0.25, name="drop1"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dropout(0.5, name="drop2"),
        layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities"),
    ]
    model = keras.Sequential(stack, name="neurowrite_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def describe_architecture(model) -> list[dict]:
    """One row per layer (name, type, output shape, parameters) for display."""
    return [{"Layer": layer.name, "Type": type(layer).__name__,
             "Output shape": str(tuple(layer.output.shape)) if hasattr(layer, "output") else "",
             "Parameters": int(layer.count_params())} for layer in model.layers]


def model_exists(path=MODEL_PATH) -> bool:
    return Path(path).exists()


def load_trained_model(path=MODEL_PATH):
    if not model_exists(path):
        raise ModelNotFoundError(f"No trained model found at {path}. Train one first with: python train_model.py")
    from tensorflow import keras

    return keras.models.load_model(path, compile=False)


def validate_model_input(x) -> None:
    """Shape (N,28,28,1), floating dtype, finite values in [0,1], at least one image."""
    if not isinstance(x, np.ndarray):
        raise ModelInputError(f"Model input must be a NumPy array; got {type(x).__name__}.")
    if x.ndim != 4 or x.shape[1:] != INPUT_SHAPE:
        raise ModelInputError(f"Model input must have shape (N, {IMAGE_SIZE}, {IMAGE_SIZE}, 1); got {x.shape}.")
    if x.shape[0] == 0:
        raise ModelInputError("Model input contains no images.")
    if not np.issubdtype(x.dtype, np.floating):
        raise ModelInputError(f"Model input must be floating point; got {x.dtype}.")
    if not np.isfinite(x).all():
        raise ModelInputError("Model input contains NaN or infinite values.")
    if x.min() < 0.0 or x.max() > 1.0:
        raise ModelInputError(f"Model input must be scaled to [0, 1]; got range {x.min():.3f}-{x.max():.3f}.")


@dataclass
class Prediction:
    digit: int
    confidence: float  # softmax probability of the predicted class (a model output, not a guarantee)
    margin: float  # top-1 minus top-2 probability
    probabilities: np.ndarray  # (10,)


def predict_probabilities(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Validated batch inference -> (N, 10) softmax probabilities."""
    validate_model_input(x)
    probs = np.asarray(model.predict(x, batch_size=batch_size, verbose=0), dtype=np.float64)
    if probs.shape != (len(x), NUM_CLASSES) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ModelInputError(f"Model returned an invalid output of shape {probs.shape}.")
    return probs


def predict_digit(model, x: np.ndarray) -> Prediction:
    """Predict one image tensor of shape (1, 28, 28, 1)."""
    validate_model_input(x)
    if len(x) != 1:
        raise ModelInputError(f"predict_digit expects a single image; got a batch of {len(x)}.")
    probs = predict_probabilities(model, x)[0]
    order = np.argsort(probs)[::-1]
    return Prediction(int(order[0]), float(probs[order[0]]), float(probs[order[0]] - probs[order[1]]), probs)
