"""Evaluation on the untouched MNIST test set, plus an end-to-end pipeline check."""
from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from utils import plots
from utils.config import (
    CLASS_NAMES,
    CONFUSION_CSV_PATH,
    FIGURES_DIR,
    HISTORY_PATH,
    METRICS_PATH,
    NUM_CLASSES,
    PIPELINE_CHECK_PATH,
    REPORT_PATH,
    SEED,
)
from utils.data import normalize_images
from utils.model import predict_probabilities
from utils.preprocessing import PreprocessingError, preprocess_image

LABELS = list(range(NUM_CLASSES))


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Test-set evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Compute every reported metric from true labels and softmax outputs."""
    y_pred = probs.argmax(axis=1)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "n_samples": int(len(y_true)),
        "n_errors": int((y_true != y_pred).sum()),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_loss": float(log_loss(y_true, probs, labels=LABELS)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
    }


def evaluate_and_save(model, x_test_raw: np.ndarray, y_test: np.ndarray, extra: dict | None = None) -> dict:
    """Evaluate ``model`` on the test set and write every artefact to ``outputs/``."""
    x_test = normalize_images(x_test_raw)
    probs = predict_probabilities(model, x_test)
    metrics = compute_metrics(y_test, probs)
    metrics["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    metrics["averaging_note"] = (
        "Macro = unweighted mean over the 10 classes; weighted = mean weighted by class support. "
        "Test loss = categorical cross-entropy."
    )
    if extra:
        metrics.update(extra)

    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    cm = np.array(metrics["confusion_matrix"])

    _write_json(METRICS_PATH, metrics)
    REPORT_PATH.write_text(classification_report(
        y_test, y_pred, labels=LABELS, target_names=CLASS_NAMES, digits=4, zero_division=0))
    np.savetxt(CONFUSION_CSV_PATH, cm, fmt="%d", delimiter=",",
               header=",".join(f"pred_{c}" for c in CLASS_NAMES), comments="")

    plots.save_figure(plots.plot_confusion_matrix(cm), FIGURES_DIR / "confusion_matrix.png")
    plots.save_figure(plots.plot_per_class_metrics(metrics["per_class"]), FIGURES_DIR / "per_class_metrics.png")

    rng = np.random.default_rng(SEED)
    correct = np.flatnonzero(y_pred == y_test)
    wrong = np.flatnonzero(y_pred != y_test)
    picks_ok = rng.choice(correct, size=min(12, len(correct)), replace=False)
    picks_bad = wrong[np.argsort(-confidence[wrong])][:12]       # most confident mistakes first
    plots.save_figure(
        plots.plot_prediction_examples(x_test_raw[picks_ok], y_test[picks_ok], y_pred[picks_ok],
                                       confidence[picks_ok], "Correct predictions (random test images)"),
        FIGURES_DIR / "correct_examples.png")
    plots.save_figure(
        plots.plot_prediction_examples(x_test_raw[picks_bad], y_test[picks_bad], y_pred[picks_bad],
                                       confidence[picks_bad], "Misclassified test images (most confident errors first)"),
        FIGURES_DIR / "misclassified_examples.png")
    return metrics


def save_training_curves(history: dict) -> None:
    _write_json(HISTORY_PATH, history)
    plots.save_figure(plots.plot_training_curves(history), FIGURES_DIR / "training_curves.png")


# --------------------------------------------------------------------------- #
# End-to-end pipeline check (uses the validation partition, never the test set)
# --------------------------------------------------------------------------- #
CONDITIONS = {
    "light_on_dark": "White digit on black background, large PNG",
    "dark_on_light": "Black digit on white background, large PNG",
    "noisy_offset_jpeg": "Off-centre, grey paper, Gaussian noise, JPEG compression",
    "transparent_png": "Black digit on a transparent PNG background",
}


def simulate_upload(img28: np.ndarray, rng: np.random.Generator, condition: str) -> bytes:
    """Re-create a real MNIST digit as a user-style upload (bigger, shifted, re-coloured)."""
    scale = int(rng.integers(8, 15))
    side = 28 * scale
    big = np.asarray(Image.fromarray(img28).resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32)
    pad = int(rng.integers(side // 6, side // 2)) if condition == "noisy_offset_jpeg" else side // 10
    canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.float32)
    dy, dx = (int(rng.integers(0, 2 * pad + 1)) for _ in range(2)) if condition == "noisy_offset_jpeg" else (pad, pad)
    canvas[dy:dy + side, dx:dx + side] = big
    canvas = np.clip(canvas, 0, 255)
    buffer = io.BytesIO()

    if condition == "light_on_dark":
        Image.fromarray(canvas.astype(np.uint8)).save(buffer, format="PNG")
    elif condition == "dark_on_light":
        Image.fromarray((255 - canvas).astype(np.uint8)).save(buffer, format="PNG")
    elif condition == "noisy_offset_jpeg":
        paper = float(rng.integers(190, 256))
        ink = float(rng.integers(0, 60))
        photo = paper - canvas / 255.0 * (paper - ink)
        photo = np.clip(photo + rng.normal(0, 8, photo.shape), 0, 255).astype(np.uint8)
        Image.fromarray(photo).save(buffer, format="JPEG", quality=int(rng.integers(55, 91)))
    elif condition == "transparent_png":
        rgba = np.zeros((*canvas.shape, 4), dtype=np.uint8)
        rgba[..., 3] = canvas.astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    else:
        raise ValueError(f"Unknown condition {condition!r}.")
    return buffer.getvalue()


def run_pipeline_check(model, x_val_raw: np.ndarray, y_val: np.ndarray, n: int = 400, seed: int = SEED) -> dict:
    """Measure accuracy of *preprocess_image + model* on re-rendered real digits.

    This checks that the upload pipeline does not damage recognisable digits. It
    is a synthetic check (upscaled MNIST), not a measure of accuracy on real
    photos, and it uses validation images so the test set stays untouched.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_val), size=min(n, len(y_val)), replace=False)
    images, labels = x_val_raw[idx], y_val[idx]

    direct = predict_probabilities(model, normalize_images(images)).argmax(axis=1)
    results = {
        "n_images_per_condition": int(len(idx)),
        "baseline_direct_28x28_accuracy": float((direct == labels).mean()),
        "conditions": {},
        "note": ("Synthetic stress test built from validation-set MNIST digits. "
                 "It verifies the preprocessing pipeline, not real-world photo accuracy."),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    for condition, description in CONDITIONS.items():
        correct, rejected = 0, 0
        for image, label in zip(images, labels):
            try:
                tensor = preprocess_image(simulate_upload(image, rng, condition)).tensor
            except PreprocessingError:
                rejected += 1
                continue
            correct += int(predict_probabilities(model, tensor).argmax() == label)
        results["conditions"][condition] = {
            "description": description,
            "accuracy": correct / len(idx),
            "rejected_images": rejected,
        }
    _write_json(PIPELINE_CHECK_PATH, results)
    return results
