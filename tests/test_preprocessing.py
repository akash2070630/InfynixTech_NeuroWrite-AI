"""Tests for the inference-time preprocessing pipeline (no TensorFlow needed)."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from utils.config import MAX_UPLOAD_BYTES
from utils.preprocessing import (
    ImageValidationError,
    PreprocessingError,
    load_image,
    otsu_threshold,
    preprocess_image,
    validate_tensor,
)


def draw_digit(kind="zero", canvas=300, offset=(0, 0), ink=255, bg=0, mode="L"):
    """Draw a simple synthetic digit-like glyph (a ring or a vertical bar)."""
    img = Image.new("L", (canvas, canvas), bg)
    d = ImageDraw.Draw(img)
    cx, cy = canvas // 2 + offset[0], canvas // 2 + offset[1]
    if kind == "zero":
        r = canvas // 5
        d.ellipse([cx - r, cy - int(r * 1.4), cx + r, cy + int(r * 1.4)], outline=ink, width=canvas // 15)
    else:  # "one": tall thin bar
        d.line([cx, cy - canvas // 4, cx, cy + canvas // 4], fill=ink, width=canvas // 15)
    return img.convert(mode)


def to_bytes(img, fmt="PNG", **kwargs):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def center_of_mass(preview):
    return ndimage.center_of_mass(preview.astype(float))


# ---------------------------------------------------------------- output contract
def test_output_shape_dtype_and_range():
    res = preprocess_image(to_bytes(draw_digit("zero")))
    assert res.tensor.shape == (1, 28, 28, 1)
    assert res.tensor.dtype == np.float32
    assert np.isfinite(res.tensor).all()
    assert 0.0 <= res.tensor.min() and res.tensor.max() <= 1.0
    assert res.tensor.max() == pytest.approx(1.0)
    assert res.preview.shape == (28, 28) and res.preview.dtype == np.uint8


def test_accepts_pil_image_path_and_filelike(tmp_path):
    img = draw_digit("zero")
    path = tmp_path / "d.png"
    img.save(path)
    from_pil = preprocess_image(img)          # in-memory PIL images are accepted
    from_path = preprocess_image(str(path))
    from_file = preprocess_image(io.BytesIO(to_bytes(img)))
    for result in (from_pil, from_path, from_file):
        assert result.tensor.shape == (1, 28, 28, 1)
    assert np.allclose(from_pil.tensor, from_path.tensor, atol=1e-5)


# ---------------------------------------------------------------- polarity
def test_black_on_white_matches_white_on_black():
    white_on_black = preprocess_image(to_bytes(draw_digit("zero", ink=255, bg=0))).tensor
    black_on_white = preprocess_image(to_bytes(draw_digit("zero", ink=0, bg=255))).tensor
    assert np.abs(white_on_black - black_on_white).mean() < 0.05


def test_auto_polarity_is_reported():
    dark = preprocess_image(to_bytes(draw_digit("zero", ink=0, bg=255)))
    light = preprocess_image(to_bytes(draw_digit("zero", ink=255, bg=0)))
    assert dark.info["polarity_detected"] == "dark_on_light"
    assert light.info["polarity_detected"] == "light_on_dark"


def test_manual_polarity_override_is_applied():
    data = to_bytes(draw_digit("zero", ink=255, bg=0))
    forced = preprocess_image(data, polarity="dark_on_light")
    assert forced.info["polarity_used"] == "dark_on_light"
    assert forced.info["polarity_detected"] == "light_on_dark"


def test_invalid_polarity_argument():
    with pytest.raises(ValueError):
        preprocess_image(to_bytes(draw_digit()), polarity="sideways")


# ---------------------------------------------------------------- geometry
def test_offcentre_digit_is_centred_and_scaled():
    res = preprocess_image(to_bytes(draw_digit("zero", canvas=400, offset=(90, -70))))
    cy, cx = center_of_mass(res.preview)
    assert abs(cy - 13.5) < 1.5 and abs(cx - 13.5) < 1.5
    rows = np.flatnonzero((res.preview > 25).any(axis=1))
    assert 17 <= rows[-1] - rows[0] + 1 <= 22          # fits the ~20 px MNIST box


def test_aspect_ratio_is_preserved():
    res = preprocess_image(to_bytes(draw_digit("one", canvas=400)))
    rows = np.flatnonzero((res.preview > 25).any(axis=1))
    cols = np.flatnonzero((res.preview > 25).any(axis=0))
    height, width = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
    assert height >= 17 and width <= height / 2         # a tall bar stays tall and thin


def test_specks_far_from_digit_do_not_shift_the_crop():
    clean = draw_digit("zero", canvas=400)
    noisy = clean.copy()
    ImageDraw.Draw(noisy).rectangle([5, 5, 12, 12], fill=255)  # dust in the corner
    a = preprocess_image(to_bytes(clean)).tensor
    b = preprocess_image(to_bytes(noisy)).tensor
    assert np.abs(a - b).mean() < 0.05


# ---------------------------------------------------------------- formats
def test_jpeg_input():
    res = preprocess_image(to_bytes(draw_digit("zero", ink=20, bg=235, mode="RGB"), "JPEG", quality=70))
    assert res.tensor.shape == (1, 28, 28, 1) and res.preview.max() > 200


def test_transparent_png_uses_alpha_as_ink():
    alpha = np.asarray(draw_digit("zero", ink=255, bg=0))
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)           # black ink, transparent background
    rgba[..., 3] = alpha
    res = preprocess_image(to_bytes(Image.fromarray(rgba, "RGBA")))
    assert res.info["alpha_channel_used"] is True
    assert res.preview.max() > 200
    reference = preprocess_image(to_bytes(draw_digit("zero"))).tensor
    assert np.abs(res.tensor - reference).mean() < 0.05


# ---------------------------------------------------------------- rejection paths
def test_blank_image_is_rejected():
    with pytest.raises(PreprocessingError, match="blank|contrast"):
        preprocess_image(to_bytes(Image.new("L", (200, 200), 255)))


def test_constant_gray_image_is_rejected():
    with pytest.raises(PreprocessingError):
        preprocess_image(to_bytes(Image.new("RGB", (120, 120), (128, 128, 128))))


def test_corrupted_bytes_are_rejected():
    with pytest.raises(ImageValidationError, match="could not be read"):
        load_image(b"this is definitely not an image")


def test_truncated_png_is_rejected():
    data = to_bytes(draw_digit("zero"))
    with pytest.raises(ImageValidationError):
        load_image(data[: len(data) // 2])


def test_empty_file_is_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        load_image(b"")


def test_unsupported_format_is_rejected():
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        load_image(to_bytes(draw_digit("zero"), "GIF"))


def test_too_small_image_is_rejected():
    with pytest.raises(ImageValidationError, match="too small"):
        load_image(to_bytes(Image.new("L", (4, 4), 0)))


def test_oversized_file_is_rejected():
    with pytest.raises(ImageValidationError, match="limit"):
        load_image(b"0" * (MAX_UPLOAD_BYTES + 1))


# ---------------------------------------------------------------- helpers
def test_validate_tensor_rejects_bad_inputs():
    good = np.zeros((1, 28, 28, 1), dtype=np.float32)
    validate_tensor(good)
    with pytest.raises(PreprocessingError):
        validate_tensor(np.zeros((28, 28), dtype=np.float32))
    with pytest.raises(PreprocessingError):
        validate_tensor(good.astype(np.float64))
    with pytest.raises(PreprocessingError):
        validate_tensor(good + 1.5)
    bad = good.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(PreprocessingError):
        validate_tensor(bad)


def test_otsu_separates_two_levels():
    values = np.concatenate([np.full(500, 20), np.full(500, 220)]).astype(np.float32)
    assert 20 <= otsu_threshold(values) < 220
