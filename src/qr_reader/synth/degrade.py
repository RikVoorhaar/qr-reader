"""Phase 5 — Global Degradation.

Apply post-compositing degradation effects to the final composited image:
Gaussian blur, additive Gaussian noise, JPEG compression, and
brightness/contrast adjustment.

Functions
---------
apply_gaussian_blur
    Apply Gaussian blur to an image.
apply_gaussian_noise
    Add Gaussian noise to an image.
apply_jpeg_compression
    Simulate JPEG compression artifacts.
apply_brightness_contrast
    Adjust brightness and contrast of an image.
apply_global_degradation
    Sample parameters from config ranges and apply degradation in order.
"""

from __future__ import annotations

import cv2
import numpy as np

from qr_reader.synth.config import AugmentationConfig

__all__ = [
    "apply_gaussian_blur",
    "apply_gaussian_noise",
    "apply_jpeg_compression",
    "apply_brightness_contrast",
    "apply_global_degradation",
]

# ---------------------------------------------------------------------------
# 5.1  Individual degradation functions
# ---------------------------------------------------------------------------


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to an image.

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Input image (RGB).
    sigma : float
        Gaussian blur sigma in pixels.  ``sigma <= 0`` returns a copy of the
        input unchanged.

    Returns
    -------
    np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Blurred image.
    """
    if sigma <= 0.0:
        return image.copy()

    # OpenCV computes the kernel size automatically when ksize = (0, 0)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    """Add Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Input image (RGB).
    rng : numpy.random.Generator
        Seeded random number generator.
    sigma : float
        Standard deviation of the Gaussian noise.  ``sigma <= 0`` returns a
        copy of the input unchanged.

    Returns
    -------
    np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Noisy image, clipped to ``[0, 255]``.
    """
    if sigma <= 0.0:
        return image.copy()

    noise = rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    result = image.astype(np.float32) + noise
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)
    return result


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Simulate JPEG compression artifacts.

    At ``quality=100`` the loss is very small but not quite zero (the JPEG
    codec is always lossy).  The function does **not** shortcut at ``quality >=
    100`` — it always encodes and decodes — so the output may differ from the
    input by a few intensity levels.

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Input image (RGB).
    quality : int
        JPEG quality in ``[0, 100]``.

    Returns
    -------
    np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        JPEG-compressed image (RGB).

    Raises
    ------
    RuntimeError
        If the JPEG encoder fails (unlikely for normal images).
    """
    # OpenCV's JPEG encoder uses BGR; convert RGB→BGR before encode and
    # BGR→RGB after decode so the function remains colour-order agnostic.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    success, encoded = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        raise RuntimeError("JPEG encoding failed")

    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: int,
    contrast: float,
) -> np.ndarray:
    """Adjust brightness and contrast of an image.

    The transformation per pixel is::

        result = contrast * pixel + brightness

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Input image (RGB).
    brightness : int
        Additive brightness offset (e.g. ``-50`` darkens, ``+50`` lightens).
    contrast : float
        Multiplicative contrast factor (``1.0`` is identity, ``0.5`` reduces
        contrast, ``1.5`` increases contrast).

    Returns
    -------
    np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Adjusted image, clipped to ``[0, 255]``.
    """
    if brightness == 0 and contrast == 1.0:
        return image.copy()

    result = image.astype(np.float32) * contrast + brightness
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# 5.2  apply_global_degradation
# ---------------------------------------------------------------------------


def _sample_float(rng: np.random.Generator, r: tuple[float, float]) -> float:
    """Uniform sample from a float range ``(lo, hi)``."""
    return rng.uniform(r[0], r[1])


def apply_global_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Sample degradation parameters from *config* ranges and apply them in order.

    The degradation pipeline (applied only when the sampled parameter is
    non-identity):

    1. Gaussian blur (if ``blur_sigma > 0``).
    2. Gaussian noise (if ``noise_sigma > 0``).
    3. JPEG compression (if ``jpeg_quality < 100``).

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Input image (RGB).
    rng : numpy.random.Generator
        Seeded random number generator.
    config : AugmentationConfig
        Pipeline configuration (uses ``blur_sigma_range``,
        ``noise_sigma_range``, ``jpeg_quality_range``).

    Returns
    -------
    degraded : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Degraded image.
    params : dict[str, float | int]
        The actual sampled degradation parameters (for metadata recording).
        Keys: ``"blur_sigma"``, ``"noise_sigma"``, ``"jpeg_quality"``.
    """
    result = image.copy()

    # 1. Gaussian blur
    blur_sigma = _sample_float(rng, config.blur_sigma_range)
    if blur_sigma > 0.0:
        result = apply_gaussian_blur(result, blur_sigma)

    # 2. Gaussian noise
    noise_sigma = _sample_float(rng, config.noise_sigma_range)
    if noise_sigma > 0.0:
        result = apply_gaussian_noise(result, rng, noise_sigma)

    # 3. JPEG compression
    jpeg_quality = int(
        rng.integers(config.jpeg_quality_range[0], config.jpeg_quality_range[1] + 1)
    )
    if jpeg_quality < 100:
        result = apply_jpeg_compression(result, jpeg_quality)

    params = {
        "blur_sigma": blur_sigma,
        "noise_sigma": noise_sigma,
        "jpeg_quality": jpeg_quality,
    }
    return result, params
