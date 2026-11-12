"""Tests for the illumination-corrected binarizer."""

from __future__ import annotations

import numpy as np

from qr_reader.qr_gen import (
    binarize_image,
    binarize_image_illumination_corrected,
    make_qr_image,
)


def _smooth_gradient(img: np.ndarray) -> np.ndarray:
    """Apply a strong left-to-right multiplicative illumination gradient."""
    width = img.shape[1]
    gain = np.linspace(0.30, 1.30, width, dtype=np.float32)
    return np.clip(
        img.astype(np.float32) * gain[None, :], 0, 255
    ).astype(np.uint8)


def test_illumination_correction_is_invariant_to_smooth_gradient():
    """The corrected binarizer should ignore a smooth multiplicative gradient."""
    clean = make_qr_image(
        content="illumination test", version=4, box_size=8, border=4
    )
    shaded = _smooth_gradient(clean)

    flat_binary = binarize_image_illumination_corrected(clean)
    shaded_binary = binarize_image_illumination_corrected(shaded)

    assert flat_binary.dtype == bool
    assert flat_binary.any()
    assert (~flat_binary).any()
    assert np.mean(flat_binary != shaded_binary) < 0.01


def test_plain_otsu_is_not_invariant_to_the_same_gradient():
    """The old global Otsu path should change substantially under the gradient."""
    clean = make_qr_image(
        content="illumination test", version=4, box_size=8, border=4
    )
    shaded = _smooth_gradient(clean)

    flat_binary = binarize_image(clean)
    shaded_binary = binarize_image(shaded)

    assert np.mean(flat_binary != shaded_binary) > 0.05
