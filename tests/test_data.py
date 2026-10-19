"""Tests for dataset validation, splitting and preparation (synthetic arrays)."""
import gzip
import struct

import numpy as np
import pytest

from utils.data import (
    DatasetError,
    _read_idx_images,
    _read_idx_labels,
    class_distribution,
    normalize_images,
    prepare_labels,
    select_sample_pool,
    split_indices,
    split_train_validation,
    validate_dataset,
)


@pytest.fixture()
def tiny():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(10), 30).astype(np.uint8)                 # 300 samples, 30 per class
    x = rng.integers(0, 256, size=(300, 28, 28), dtype=np.uint8)
    return x, y


def test_valid_dataset_passes(tiny):
    x, y = tiny
    validate_dataset(x, y, x[:100], y[:100], strict_size=False)


def test_strict_size_rejects_non_mnist_sizes(tiny):
    x, y = tiny
    with pytest.raises(DatasetError, match="60000"):
        validate_dataset(x, y, x, y, strict_size=True)


def test_wrong_image_shape_is_rejected(tiny):
    x, y = tiny
    with pytest.raises(DatasetError, match="28, 28"):
        validate_dataset(x[:, :27, :], y, x, y, strict_size=False)


def test_wrong_dtype_is_rejected(tiny):
    x, y = tiny
    with pytest.raises(DatasetError, match="uint8"):
        validate_dataset(x.astype("float32"), y, x, y, strict_size=False)


def test_label_out_of_range_is_rejected(tiny):
    x, y = tiny
    bad = y.copy()
    bad[0] = 10
    with pytest.raises(DatasetError, match="labels"):
        validate_dataset(x, bad, x, y, strict_size=False)


def test_missing_class_is_rejected(tiny):
    x, y = tiny
    keep = y != 3
    with pytest.raises(DatasetError, match="cover all"):
        validate_dataset(x[keep], y[keep], x, y, strict_size=False)


def test_label_image_count_mismatch_is_rejected(tiny):
    x, y = tiny
    with pytest.raises(DatasetError, match="do not match"):
        validate_dataset(x, y[:-1], x, y, strict_size=False)


def test_normalisation_shape_dtype_and_range(tiny):
    x, _ = tiny
    out = normalize_images(x)
    assert out.shape == (300, 28, 28, 1) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.max() == pytest.approx(x.max() / 255.0)


def test_normalisation_rejects_floats(tiny):
    x, _ = tiny
    with pytest.raises(DatasetError):
        normalize_images(x.astype("float32"))


def test_prepare_labels(tiny):
    _, y = tiny
    out = prepare_labels(y)
    assert out.dtype == np.int32 and out.shape == y.shape
    with pytest.raises(DatasetError):
        prepare_labels(np.array([0, 1, 11]))


def test_split_is_disjoint_stratified_and_deterministic(tiny):
    _, y = tiny
    tr, va = split_indices(y, 0.2, seed=7)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(y)
    assert len(va) == 60
    assert all(c == 6 for c in class_distribution(y[va]).values())      # stratified
    tr2, va2 = split_indices(y, 0.2, seed=7)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)


def test_split_train_validation_shapes(tiny):
    x, y = tiny
    x_tr, x_va, y_tr, y_va = split_train_validation(x, y, 0.1, seed=1)
    assert len(x_tr) == len(y_tr) == 270 and len(x_va) == len(y_va) == 30


def test_sample_pool_is_balanced_and_seeded(tiny):
    x, y = tiny
    sx, sy, idx = select_sample_pool(x, y, per_class=5, seed=3)
    assert sx.shape == (50, 28, 28) and all(c == 5 for c in class_distribution(sy).values())
    _, _, idx2 = select_sample_pool(x, y, per_class=5, seed=3)
    assert np.array_equal(idx, idx2)


def test_idx_readers(tmp_path):
    images = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    with gzip.open(tmp_path / "img.gz", "wb") as fh:
        fh.write(struct.pack(">IIII", 2051, 2, 3, 3) + images.tobytes())
    with gzip.open(tmp_path / "lab.gz", "wb") as fh:
        fh.write(struct.pack(">II", 2049, 2) + bytes([4, 7]))
    assert np.array_equal(_read_idx_images(tmp_path / "img.gz"), images)
    assert _read_idx_labels(tmp_path / "lab.gz").tolist() == [4, 7]
    with gzip.open(tmp_path / "bad.gz", "wb") as fh:
        fh.write(struct.pack(">II", 1234, 2) + bytes([1, 2]))
    with pytest.raises(DatasetError):
        _read_idx_labels(tmp_path / "bad.gz")
