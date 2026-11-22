"""Grayscale conversion and binarization for the detection pipeline.

This module is on the main detection path, so it uses only NumPy and SciPy —
it must never import OpenCV.  OpenCV stays confined to synthetic image
generation (`qr_gen`, `synth`) and to verification tooling (scripts, tests).
The rule is enforced by `tests/test_no_opencv_in_pipeline.py`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# cv2.cvtColor(..., COLOR_BGR2GRAY) coefficients, in BGR channel order.
_BGR_TO_GRAY = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return *img* as an 8-bit grayscale array.

    Colour input is assumed to be BGR (the ``cv2.imread`` convention) or BGRA;
    grayscale input is returned unchanged.  Note that RGB input gives slightly
    wrong weights — convert explicitly before calling if that matters.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {img.shape}")

    gray = img[..., :3].astype(np.float32) @ _BGR_TO_GRAY
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def otsu_threshold(img: np.ndarray) -> int:
    """Otsu's threshold: the level maximising between-class variance."""
    values = np.asarray(img)
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)

    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)

    weight_below = np.cumsum(hist)
    weight_above = hist.sum() - weight_below
    sum_below = np.cumsum(hist * levels)
    sum_above = sum_below[-1] - sum_below

    mean_below = np.divide(
        sum_below, weight_below, out=np.zeros_like(sum_below), where=weight_below > 0
    )
    mean_above = np.divide(
        sum_above, weight_above, out=np.zeros_like(sum_above), where=weight_above > 0
    )
    between_class_variance = (
        weight_below * weight_above * (mean_below - mean_above) ** 2
    )
    return int(np.argmax(between_class_variance))


def binarize_image(img: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Threshold to a boolean image (``True`` = white).

    If *threshold* is ``None`` (default), Otsu's method is used.
    """
    values = np.asarray(img)
    level = otsu_threshold(values) if threshold is None else threshold
    return values > level


_ILLUMINATION_DOWNSAMPLE = 4


def estimate_illumination(
    img: np.ndarray, blur_size: int = 101, step: int = _ILLUMINATION_DOWNSAMPLE
) -> np.ndarray:
    """Estimate the slowly-varying illumination field with a box blur.

    The window is much larger than a QR module, so the dark/light pattern
    averages out and only the local brightness remains.  The field is smooth
    by construction, so it is estimated on a 1/*step* subsample and expanded
    back with nearest-neighbour repetition — a large speed-up at a negligible
    accuracy cost.  Returns ``float32`` gray levels in the same shape as *img*.
    """
    img = np.asarray(img, dtype=np.float32)
    window = max(3, int(round(blur_size / step)))
    small = uniform_filter(img[::step, ::step], size=window, mode="reflect")
    expanded = np.repeat(np.repeat(small, step, axis=0), step, axis=1)
    return expanded[: img.shape[0], : img.shape[1]]


def illumination_corrected_image(img: np.ndarray, blur_size: int = 101) -> np.ndarray:
    """Divide out the estimated illumination and rescale to 8-bit."""
    img = np.asarray(img)
    background = estimate_illumination(img, blur_size)
    normalized = img.astype(np.float32) / np.maximum(background, 1.0) * 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _erode_cross(mask: np.ndarray) -> np.ndarray:
    """Erode a boolean mask with a plus-shaped 3×3 element.

    Equivalent to ``cv2.erode(mask, cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)))``: the image border is left untouched rather
    than treated as background.  Written with shifted boolean ops because
    ``scipy.ndimage.binary_erosion`` is ~25× slower on megapixel images.
    """
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def _dilate_cross(mask: np.ndarray) -> np.ndarray:
    """Dilate a boolean mask with a plus-shaped 3×3 element (see `_erode_cross`)."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def binarize_image_illumination_corrected(
    img: np.ndarray,
    blur_size: int = 101,
) -> np.ndarray:
    """Threshold to a boolean image after removing uneven illumination.

    Estimates the illumination field with a large box blur, divides it out,
    applies Otsu's method to the normalized image, and removes isolated noise
    with a 3×3 morphological opening.  Returns ``True`` for white pixels.
    """
    corrected = illumination_corrected_image(img, blur_size)
    binary = binarize_image(corrected)
    return _dilate_cross(_erode_cross(binary))
