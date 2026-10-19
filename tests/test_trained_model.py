"""Tests that need the trained model and training artefacts.

They are skipped (not failed) when `python train_model.py` has not been run yet.
"""
import io
import json

import numpy as np
import pytest
from PIL import Image

from utils.config import METRICS_PATH, MODEL_PATH, SAMPLES_PATH
from utils.data import normalize_images
from utils.evaluation import CONDITIONS, simulate_upload
from utils.inference import recognise
from utils.model import load_trained_model, predict_probabilities

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not found; run train_model.py")
needs_samples = pytest.mark.skipif(not SAMPLES_PATH.exists(), reason="sample images not found; run train_model.py")
needs_metrics = pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics not found; run train_model.py")


@pytest.fixture(scope="module")
def model():
    return load_trained_model(MODEL_PATH)


@pytest.fixture(scope="module")
def samples():
    with np.load(SAMPLES_PATH) as data:
        return data["images"], data["labels"]


@needs_model
@needs_samples
def test_model_recognises_real_mnist_samples(model, samples):
    images, labels = samples
    preds = predict_probabilities(model, normalize_images(images)).argmax(axis=1)
    assert (preds == labels).mean() > 0.95      # loose sanity floor, not the reported metric


@needs_model
@needs_samples
@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_full_upload_pipeline_predicts_valid_output(model, samples, condition):
    images, labels = samples
    rng = np.random.default_rng(0)
    hits = 0
    picks = rng.choice(len(labels), size=20, replace=False)
    for i in picks:
        result = recognise(model, simulate_upload(images[i], rng, condition))
        assert 0 <= result.prediction.digit <= 9
        assert result.processed.tensor.shape == (1, 28, 28, 1)
        assert np.isclose(result.prediction.probabilities.sum(), 1.0, atol=1e-3)
        hits += result.prediction.digit == labels[i]
    assert hits >= 15      # the pipeline must not destroy recognisable digits


@needs_model
@needs_samples
def test_polarity_does_not_change_prediction(model, samples):
    images, labels = samples
    rng = np.random.default_rng(1)
    agree = 0
    for i in rng.choice(len(labels), size=15, replace=False):
        a = recognise(model, simulate_upload(images[i], rng, "light_on_dark")).prediction.digit
        b = recognise(model, simulate_upload(images[i], np.random.default_rng(1), "dark_on_light")).prediction.digit
        agree += a == b
    assert agree >= 13


@needs_metrics
def test_saved_metrics_are_internally_consistent():
    m = json.loads(METRICS_PATH.read_text())
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (10, 10)
    assert cm.sum() == m["n_samples"] == 10_000          # standard MNIST test set
    assert np.trace(cm) / cm.sum() == pytest.approx(m["test_accuracy"])
    assert m["n_errors"] == cm.sum() - np.trace(cm)
    for digit in range(10):
        assert m["per_class"][str(digit)]["support"] == cm[digit].sum()
    for key in ("precision_macro", "recall_macro", "f1_macro", "precision_weighted"):
        assert 0.0 <= m[key] <= 1.0
