"""QR code generation with randomized transforms, noise, and blur.

All randomness is controlled by a single seed via ``numpy.random.default_rng(seed)``.
Each step is a small, testable function.
"""

from __future__ import annotations

import cv2
import numpy as np
import qrcode


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------


def make_qr_image(
    content: str = "Some data",
    *,
    version: int = 1,
    error_correction: int = qrcode.constants.ERROR_CORRECT_L,
    box_size: int = 10,
    border: int = 4,
) -> np.ndarray:
    """Generate a clean binary QR code image (dtype ``np.uint8``, values 0 or 255)."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image()
    return np.array(img).astype(np.uint8) * 255


def rotate_image(
    img: np.ndarray,
    angle_deg: float,
    border_value: int = 255,
) -> np.ndarray:
    """Rotate *img* around its centre by *angle_deg* degrees."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(border_value,) * img.shape[-1]
                          if img.ndim == 3 else border_value)


def random_perspective_warp(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    max_shift: float = 50.0,
    border_value: int = 255,
) -> np.ndarray:
    """Apply a random perspective warp to *img*.

    Each corner is shifted by a uniform random offset in [0, *max_shift*] in both
    x and y (clamped to image bounds).
    """
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    shifts = rng.uniform(0, max_shift, size=(4, 2)).astype(np.float32)
    dst_pts = src_pts + shifts
    # Clamp to image bounds
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, w - 1)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, h - 1)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        img, M, (w, h),
        borderValue=(border_value,) * img.shape[-1] if img.ndim == 3 else border_value,
    )


def add_gaussian_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    std: float = 50.0,
    blur_kernel: int = 3,
    intensity_scale: float = 0.8,
) -> np.ndarray:
    """Add spatially smoothed Gaussian noise to *img*, scaled and clamped to [0, 255]."""
    noise = rng.normal(0, std, img.shape).astype(np.float32)
    spatial_noise = cv2.GaussianBlur(noise, (blur_kernel, blur_kernel), 0)
    return np.clip(img.astype(np.float32) * intensity_scale + spatial_noise, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def binarize_image(img: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Threshold to a boolean image (True = white).

    If *threshold* is ``None`` (default), Otsu's method is used.
    """
    if threshold is None:
        threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary.astype(bool)
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1].astype(bool)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_test_image(
    seed: int = 0,
    *,
    # QR code
    content: str = "Some data",
    version: int = 1,
    error_correction: int = qrcode.constants.ERROR_CORRECT_L,
    box_size: int = 10,
    border: int = 4,
    # Transforms
    rotation_angle_deg: float = 20.0,
    perspective_max_shift: float = 50.0,
    # Noise / blur
    noise_std: float = 50.0,
    noise_blur_kernel: int = 3,
    intensity_scale: float = 0.8,
    final_blur_kernel: int = 5,
    # Border fill colour
    border_value: int = 255,
) -> np.ndarray:
    """Full pipeline: QR code → rotate → random perspective → noise → blur.

    Returns a grayscale ``uint8`` image ready for thresholding.
    All randomness derives from *seed*.
    """
    rng = np.random.default_rng(seed)

    img = make_qr_image(
        content=content,
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )

    img = rotate_image(img, rotation_angle_deg, border_value=border_value)

    img = random_perspective_warp(
        img, rng, max_shift=perspective_max_shift, border_value=border_value,
    )

    img = add_gaussian_noise(
        img, rng,
        std=noise_std,
        blur_kernel=noise_blur_kernel,
        intensity_scale=intensity_scale,
    )

    img = gaussian_blur(img, kernel_size=final_blur_kernel)
    return img


