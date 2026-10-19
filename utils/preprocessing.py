"""The single image-preprocessing pipeline used for every uploaded image.

Mirrors how MNIST digits were made (light digit on black, 20x20 box centred by centre of mass in a
28x28 frame): validate/decode -> grayscale (alpha used as ink for transparent PNGs) -> polarity
detection from the border -> contrast stretch -> Otsu foreground + removal of tiny specks -> tight
crop -> aspect-preserving resize to 20 px -> centre by centre of mass in 28x28 -> scale to [0, 1].
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from utils.config import IMAGE_SIZE, MAX_IMAGE_SIDE, MAX_UPLOAD_BYTES, MIN_IMAGE_SIDE

POLARITY_AUTO = "auto"
POLARITY_DARK_ON_LIGHT = "dark_on_light"
POLARITY_LIGHT_ON_DARK = "light_on_dark"
_POLARITIES = (POLARITY_AUTO, POLARITY_DARK_ON_LIGHT, POLARITY_LIGHT_ON_DARK)

DIGIT_BOX = 20  # MNIST digits fit a 20x20 box inside the 28x28 frame
MIN_CONTRAST = 30  # grey levels between darkest and brightest pixel
WORK_SIDE = 1024  # working resolution used to locate the digit
_FORMATS = {"PNG", "JPEG", "MPO"}  # MPO = JPEG variant written by some phones


class ImageValidationError(ValueError):
    """The uploaded file is not a usable image (type, size, corruption)."""


class PreprocessingError(ValueError):
    """The image is valid but no digit can be extracted (blank, no contrast, bad tensor)."""


@dataclass
class PreprocessResult:
    tensor: np.ndarray  # (1, 28, 28, 1) float32 in [0, 1]
    preview: np.ndarray  # (28, 28) uint8 view of the same data
    info: dict


# ------------------------------------------------------------------ loading
def _read_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        try:
            with open(source, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise ImageValidationError(f"The file could not be opened ({exc}).") from exc
    if hasattr(source, "read"):
        return source.read()
    raise ImageValidationError(f"Unsupported input type {type(source).__name__}; expected bytes, a path, a file or a PIL image.")


def load_image(source) -> Image.Image:
    """Validate and fully decode bytes / path / file-like / PIL image. Raises ImageValidationError."""
    if isinstance(source, Image.Image):
        img = source.copy()
    else:
        data = _read_bytes(source)
        if not data:
            raise ImageValidationError("The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ImageValidationError(f"The file is {len(data) / 2**20:.1f} MB, above the {MAX_UPLOAD_BYTES // 2**20} MB limit.")
        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            img.load()  # forces a full decode, so truncated files fail here
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            raise ImageValidationError(f"The file could not be read as an image ({exc}). It may be corrupted.") from exc
        if fmt not in _FORMATS:
            raise ImageValidationError(f"Unsupported image format '{fmt}'. Please upload a PNG, JPG or JPEG file.")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if min(w, h) < MIN_IMAGE_SIDE:
        raise ImageValidationError(f"The image is too small ({w}x{h} px); each side needs at least {MIN_IMAGE_SIDE} px.")
    if max(w, h) > MAX_IMAGE_SIDE:
        raise ImageValidationError(f"The image is too large ({w}x{h} px); the longest side may be at most {MAX_IMAGE_SIDE} px.")
    return img


# ------------------------------------------------------------------ helpers
def otsu_threshold(values) -> int:
    """Otsu's threshold (0-255) separating a dark and a bright population."""
    v = np.clip(np.asarray(values, dtype=np.float64).ravel(), 0, 255).astype(np.uint8)
    hist = np.bincount(v, minlength=256).astype(np.float64)
    total, sum_all = hist.sum(), float(np.dot(np.arange(256), hist))
    w_b = sum_b = 0.0
    best_var, best_t = -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        var = w_b * w_f * (sum_b / w_b - (sum_all - sum_b) / w_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return int(best_t)


def validate_tensor(x) -> None:
    """The exact tensor contract the CNN receives: (1,28,28,1) float32, finite, within [0,1]."""
    if not isinstance(x, np.ndarray) or x.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 1):
        raise PreprocessingError(f"Tensor must have shape (1, {IMAGE_SIZE}, {IMAGE_SIZE}, 1); got {getattr(x, 'shape', type(x))}.")
    if x.dtype != np.float32:
        raise PreprocessingError(f"Tensor must be float32; got {x.dtype}.")
    if not np.isfinite(x).all():
        raise PreprocessingError("Tensor contains NaN or infinite values.")
    if x.min() < 0.0 or x.max() > 1.0:
        raise PreprocessingError(f"Tensor values must lie in [0, 1]; got {x.min():.3f}-{x.max():.3f}.")


def _border(g: np.ndarray) -> np.ndarray:
    k = max(1, int(0.04 * min(g.shape)))
    return np.concatenate([g[:k].ravel(), g[-k:].ravel(), g[:, :k].ravel(), g[:, -k:].ravel()])


def _ink_channel(img: Image.Image):
    """Grayscale array; for genuinely transparent PNGs the alpha channel is the ink."""
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        alpha = np.asarray(img.convert("RGBA"))[..., 3].astype(np.float32)
        if (alpha < 250).mean() > 0.02:
            return alpha, True
    return np.asarray(img.convert("RGB").convert("L"), dtype=np.float32), False


# ------------------------------------------------------------------ main entry point
def preprocess_image(source, polarity: str = POLARITY_AUTO, auto_crop: bool = True) -> PreprocessResult:
    """Convert an image into the (1, 28, 28, 1) float32 tensor expected by the CNN."""
    if polarity not in _POLARITIES:
        raise ValueError(f"polarity must be one of {_POLARITIES}; got '{polarity}'.")
    img = load_image(source)
    original_size = img.size
    gray, alpha_used = _ink_channel(img)

    scale = 1.0
    if max(gray.shape) > WORK_SIDE:
        scale = WORK_SIDE / max(gray.shape)
        small = Image.fromarray(gray.astype(np.uint8)).resize(
            (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))), Image.LANCZOS)
        gray = np.asarray(small, dtype=np.float32)

    if alpha_used:
        detected = used = POLARITY_LIGHT_ON_DARK  # alpha already marks the ink as bright
    else:
        detected = POLARITY_DARK_ON_LIGHT if float(np.median(_border(gray))) > 127 else POLARITY_LIGHT_ON_DARK
        used = detected if polarity == POLARITY_AUTO else polarity
    g = 255.0 - gray if used == POLARITY_DARK_ON_LIGHT else gray

    lo, hi = float(g.min()), float(g.max())
    if hi - lo < MIN_CONTRAST:
        raise PreprocessingError("No digit found: the image looks blank or has too little contrast between ink and background.")
    g = (g - lo) / (hi - lo) * 255.0
    g[g < 0.12 * 255.0] = 0  # suppress paper / JPEG noise

    thr = otsu_threshold(g)
    if auto_crop:
        mask = g > max(thr * 0.6, 20)
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            raise PreprocessingError("No digit found after thresholding; try a clearer, higher-contrast image.")
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.10 * sizes.max()))  # drops dust / specks
        rows, cols = np.flatnonzero(keep.any(axis=1)), np.flatnonzero(keep.any(axis=0))
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if (y1 - y0) < 3 and (x1 - x0) < 3:
            raise PreprocessingError("The detected mark is too small to be a digit; crop closer to the digit.")
    else:
        y0, y1, x0, x1 = 0, g.shape[0], 0, g.shape[1]

    crop = Image.fromarray(np.clip(g[y0:y1, x0:x1], 0, 255).astype(np.uint8))
    f = DIGIT_BOX / max(crop.size)
    new = (max(1, round(crop.size[0] * f)), max(1, round(crop.size[1] * f)))
    crop = crop.resize(new, Image.LANCZOS)  # aspect ratio preserved
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(crop, ((IMAGE_SIZE - new[0]) // 2, (IMAGE_SIZE - new[1]) // 2))
    arr = np.asarray(canvas, dtype=np.float32)
    if arr.sum() <= 0:
        raise PreprocessingError("The digit vanished while resizing; try a larger or darker digit.")

    cy = float((arr.sum(axis=1) * np.arange(IMAGE_SIZE)).sum() / arr.sum())
    cx = float((arr.sum(axis=0) * np.arange(IMAGE_SIZE)).sum() / arr.sum())
    centre = IMAGE_SIZE / 2 - 0.5
    dy, dx = (int(np.clip(round(centre - c), -4, 4)) for c in (cy, cx))
    arr = np.roll(arr, (dy, dx), axis=(0, 1))  # margin >= 4 px, so nothing wraps around

    arr = arr / arr.max()
    tensor = arr.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    validate_tensor(tensor)
    info = {
        "original_size": list(original_size),
        "polarity_detected": detected,
        "polarity_used": used,
        "alpha_channel_used": bool(alpha_used),
        "auto_crop": bool(auto_crop),
        "crop_box_xyxy": [int(v / scale) for v in (x0, y0, x1, y1)],
        "otsu_threshold": int(thr),
        "foreground_fraction": float((g > 20).mean()),
    }
    return PreprocessResult(tensor=tensor, preview=np.rint(arr * 255).astype(np.uint8), info=info)
