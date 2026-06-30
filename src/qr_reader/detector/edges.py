"""Thin edge extraction via Gaussian blur → Sobel → L2 magnitude → Non-Maximum Suppression.


Returns thinned edge magnitude and gradient angle images for downstream use
(e.g., gradient-guided Hough voting).
"""

import numpy as np
from scipy import ndimage


def extract_thin_edges(
    roi: np.ndarray, blur_sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract thin edges from a grayscale ROI.

    Pipeline: Gaussian blur → Sobel gradients → L2 magnitude →
    atan2 orientation → interpolated non-maximum suppression.

    Border pixels (1px margin) are zeroed in the NMS output since their
    gradient-direction neighbors fall outside the image.

    Parameters
    ----------
    roi : np.ndarray
        2-D grayscale image (float or uint8).
    blur_sigma : float
        Gaussian blur sigma passed to ``scipy.ndimage.gaussian_filter`` (default 1.0).

    Returns
    -------
    nms : np.ndarray
        L2 gradient magnitude after non-maximum suppression (same shape as *roi*,
        float64). Non-edge pixels are 0.
    angle : np.ndarray
        ``atan2(gy, gx)`` in radians, range [-π, π], same shape as *roi* (float64).
        Non-edge pixels are 0.
    """
    roi_f = roi.astype(np.float64, copy=False)

    # 1. Gaussian blur
    blurred = ndimage.gaussian_filter(roi_f, sigma=blur_sigma, mode="reflect")

    # 2. Sobel gradients
    gx = ndimage.sobel(blurred, axis=1, mode="constant")
    gy = ndimage.sobel(blurred, axis=0, mode="constant")

    # 3. L2 magnitude
    mag = np.hypot(gx, gy)

    # 4. Gradient direction (atan2), zero where no edge exists
    angle = np.arctan2(gy, gx, out=np.zeros_like(mag), where=mag > 0)

    # 5. Interpolated non-maximum suppression
    nms = _non_maximum_suppression_interpolated(mag, gx, gy)

    # Zero out angle where NMS suppressed (for clean visualization)
    angle = np.where(nms > 0, angle, 0.0)

    return nms, angle


def _non_maximum_suppression_interpolated(
    mag: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    """Per-pixel NMS with linear interpolation along the exact gradient direction.

    For each interior pixel, the two neighbors along the gradient normal are
    linearly interpolated from their nearest four neighbours.  The center pixel
    survives only when its magnitude is >= both interpolated neighbours.

    Border pixels (one-pixel margin) are always suppressed.
    """
    h, w = mag.shape
    nms = np.zeros_like(mag)

    # Classify each pixel as horizontal-dominant (|gx| >= |gy|) or
    # vertical-dominant (|gy| > |gx|) to avoid dividing by the smaller
    # component and keep the fractional step ≤ 1.
    hdom = np.abs(gx) >= np.abs(gy)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = mag[y, x]
            if m == 0:
                continue

            gx_val = gx[y, x]
            gy_val = gy[y, x]

            if hdom[y, x]:
                # Horizontal-dominant: step along x is ±1, step along y is gy/gx.
                dy = gy_val / gx_val
                sx = np.sign(gx_val)
                n1 = _bilinear_sample(mag, y - dy, x - sx)
                n2 = _bilinear_sample(mag, y + dy, x + sx)
            else:
                # Vertical-dominant: step along y is ±1, step along x is gx/gy.
                dx = gx_val / gy_val
                sy = np.sign(gy_val)
                n1 = _bilinear_sample(mag, y - sy, x - dx)
                n2 = _bilinear_sample(mag, y + sy, x + dx)

            if m >= n1 and m >= n2:
                nms[y, x] = m

    return nms


def _bilinear_sample(img: np.ndarray, y: float, x: float) -> float:
    """Bilinear interpolation at sub-pixel position (y, x), clamped to image bounds."""
    h, w = img.shape

    y0 = int(np.clip(np.floor(y), 0, h - 1))
    x0 = int(np.clip(np.floor(x), 0, w - 1))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)

    fy = np.clip(y - y0, 0.0, 1.0)
    fx = np.clip(x - x0, 0.0, 1.0)

    return float(
        (1.0 - fy) * ((1.0 - fx) * img[y0, x0] + fx * img[y0, x1])
        + fy * ((1.0 - fx) * img[y1, x0] + fx * img[y1, x1])
    )
