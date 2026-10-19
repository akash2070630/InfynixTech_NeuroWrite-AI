"""Tests for the CNN definition, loading and validated inference (untrained weights)."""
import numpy as np
import pytest

from utils.model import (
    ModelInputError,
    ModelNotFoundError,
    build_cnn,
    describe_architecture,
    load_trained_model,
    predict_digit,
    predict_probabilities,
    validate_model_input,
)


@pytest.fixture(scope="module")
def model():
    return build_cnn(augment=True)


def random_batch(n=4, seed=0):
    return np.random.default_rng(seed).random((n, 28, 28, 1)).astype("float32")


def test_input_and_output_shapes(model):
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)


def test_architecture_contains_required_layer_types(model):
    kinds = {row["Type"] for row in describe_architecture(model)}
    assert {"Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout"} <= kinds
    assert model.layers[-1].get_config()["activation"] == "softmax"
    assert model.layers[-1].units == 10


def test_probabilities_are_valid_distributions(model):
    probs = predict_probabilities(model, random_batch(8))
    assert probs.shape == (8, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert (probs >= 0).all()


def test_predict_digit_output(model):
    result = predict_digit(model, random_batch(1))
    assert 0 <= result.digit <= 9
    assert 0.0 < result.confidence <= 1.0
    assert 0.0 <= result.margin <= result.confidence
    assert result.probabilities.shape == (10,)
    assert result.digit == int(np.argmax(result.probabilities))


def test_inference_is_deterministic_despite_augmentation_layers(model):
    x = random_batch(3, seed=5)
    assert np.allclose(predict_probabilities(model, x), predict_probabilities(model, x))


def test_large_batches_use_predict_path(model):
    assert predict_probabilities(model, random_batch(20), batch_size=8).shape == (20, 10)


@pytest.mark.parametrize("bad", [
    np.zeros((28, 28), dtype="float32"),
    np.zeros((1, 28, 28), dtype="float32"),
    np.zeros((1, 32, 32, 1), dtype="float32"),
    np.zeros((1, 28, 28, 1), dtype="int32"),
    np.full((1, 28, 28, 1), 255.0, dtype="float32"),
    np.full((1, 28, 28, 1), np.nan, dtype="float32"),
    np.zeros((0, 28, 28, 1), dtype="float32"),
])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ModelInputError):
        validate_model_input(bad)


def test_non_array_is_rejected(model):
    with pytest.raises(ModelInputError):
        predict_probabilities(model, [[0.0]])


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python train_model.py"):
        load_trained_model(tmp_path / "nope.keras")


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "roundtrip.keras"
    model.save(path)
    restored = load_trained_model(path)
    x = random_batch(3, seed=9)
    assert np.allclose(predict_probabilities(model, x), predict_probabilities(restored, x), atol=1e-5)
