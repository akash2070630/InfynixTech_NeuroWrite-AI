"""MNIST loading, validation, normalisation and splitting.

Loading order: verified local IDX files in data/ -> download of the original IDX files from a
GitHub mirror (MD5 verified) -> keras.datasets.mnist. The source actually used is reported.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from utils.config import DATA_DIR, IMAGE_SIZE, NUM_CLASSES, SEED, VAL_FRACTION

log = logging.getLogger(__name__)

MIRROR_URL = "https://github.com/fgnt/mnist/raw/master/"
# MD5 checksums of the official MNIST distribution files
MNIST_FILES = {
    "x_train": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "y_train": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "x_test": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "y_test": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}
STANDARD_SIZES = {"train": 60_000, "test": 10_000}


class DatasetError(ValueError):
    """Raised when the dataset is missing, corrupted or has unexpected content."""


@dataclass
class RawMNIST:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source: str


# ------------------------------------------------------------------ IDX parsing
def _read_idx(path, expected_magic: int, ndim: int) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4 * (ndim + 1) or int.from_bytes(raw[:4], "big") != expected_magic:
        raise DatasetError(f"{Path(path).name} is not a valid IDX file (bad magic number).")
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 * (ndim + 1))
    if data.size != int(np.prod(dims)):
        raise DatasetError(f"{Path(path).name} is truncated: expected {int(np.prod(dims))} bytes, found {data.size}.")
    return data.reshape(dims)


def _read_idx_images(path) -> np.ndarray:
    return _read_idx(path, 2051, 3)


def _read_idx_labels(path) -> np.ndarray:
    return _read_idx(path, 2049, 1)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_ok(data_dir: Path) -> bool:
    return all((data_dir / n).exists() and _md5(data_dir / n) == m for n, m in MNIST_FILES.values())


def _download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in MNIST_FILES.values():
        target = data_dir / name
        if target.exists() and _md5(target) == md5:
            continue
        with urllib.request.urlopen(MIRROR_URL + name, timeout=60) as resp:
            target.write_bytes(resp.read())
        if _md5(target) != md5:
            target.unlink(missing_ok=True)
            raise DatasetError(f"Checksum mismatch for {name}: the download was corrupted.")


def load_mnist(data_dir: Path = DATA_DIR) -> RawMNIST:
    """Load the genuine MNIST train/test partitions as raw uint8 arrays."""
    data_dir = Path(data_dir)
    source = "official MNIST idx files in data/ (MD5 verified)"
    if not _files_ok(data_dir):
        try:
            _download(data_dir)
            source = "official MNIST idx files downloaded to data/ (MD5 verified)"
        except Exception as exc:  # reported, then the Keras copy is tried
            log.warning("IDX download failed (%s); trying keras.datasets.mnist.", exc)
            try:
                from tensorflow.keras.datasets import mnist
            except ImportError as imp:
                raise DatasetError("MNIST is unavailable: download failed and TensorFlow is not installed.") from imp
            try:
                (xtr, ytr), (xte, yte) = mnist.load_data()
            except Exception as kexc:
                raise DatasetError(
                    "Could not obtain MNIST. Check your internet connection, or place the four official "
                    f"*-idx*-ubyte.gz files in {data_dir}.") from kexc
            return RawMNIST(xtr, ytr, xte, yte, "keras.datasets.mnist")
    return RawMNIST(
        _read_idx_images(data_dir / MNIST_FILES["x_train"][0]), _read_idx_labels(data_dir / MNIST_FILES["y_train"][0]),
        _read_idx_images(data_dir / MNIST_FILES["x_test"][0]), _read_idx_labels(data_dir / MNIST_FILES["y_test"][0]),
        source)


# ------------------------------------------------------------------ validation / preparation
def validate_dataset(x_train, y_train, x_test, y_test, strict_size: bool = True) -> None:
    """Raise DatasetError on unexpected shapes, dtypes, label ranges or missing classes."""
    for name, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
        if x.ndim != 3 or x.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise DatasetError(f"{name} images must have shape (N, {IMAGE_SIZE}, {IMAGE_SIZE}); got {x.shape}.")
        if x.dtype != np.uint8:
            raise DatasetError(f"{name} images must be uint8 (0-255); got {x.dtype}.")
        if y.ndim != 1 or len(y) != len(x):
            raise DatasetError(f"{name} labels {y.shape} do not match the {len(x)} images.")
        if y.min() < 0 or y.max() >= NUM_CLASSES:
            raise DatasetError(f"{name} labels must be in 0-{NUM_CLASSES - 1}; found {y.min()}-{y.max()}.")
        if len(np.unique(y)) != NUM_CLASSES:
            raise DatasetError(f"{name} labels must cover all {NUM_CLASSES} classes.")
        if strict_size and len(x) != STANDARD_SIZES[name]:
            raise DatasetError(f"Standard MNIST {name} split has {STANDARD_SIZES[name]} images (found {len(x)}).")


def normalize_images(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 (N,28,28) -> float32 (N,28,28,1) in [0,1]; the same scaling as the upload pipeline."""
    if not isinstance(x_uint8, np.ndarray) or x_uint8.dtype != np.uint8:
        raise DatasetError("Images must be a uint8 NumPy array before normalisation.")
    return (x_uint8.astype("float32") / 255.0).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def prepare_labels(y: np.ndarray) -> np.ndarray:
    """Integer class labels (int32) for sparse categorical cross-entropy."""
    y = np.asarray(y)
    if y.ndim != 1 or y.min() < 0 or y.max() >= NUM_CLASSES:
        raise DatasetError(f"Labels must be a 1-D array with values in 0-{NUM_CLASSES - 1}.")
    return y.astype("int32")


def class_distribution(y: np.ndarray) -> dict:
    counts = np.bincount(np.asarray(y).astype(int), minlength=NUM_CLASSES)
    return {i: int(c) for i, c in enumerate(counts)}


def split_indices(y: np.ndarray, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Stratified, seeded train/validation indices (taken from the training partition only)."""
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=val_fraction, random_state=seed, stratify=y)
    return train_idx, val_idx


def split_train_validation(x, y, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    tr, va = split_indices(y, val_fraction, seed)
    return x[tr], x[va], y[tr], y[va]


def select_sample_pool(x, y, per_class: int = 20, seed: int = SEED):
    """Seeded, class-balanced subset used by the app's sample gallery. Returns (images, labels, indices)."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(NUM_CLASSES)])
    return x[idx], y[idx], idx
