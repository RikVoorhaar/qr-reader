"""Phase 2 — Perspective augmentation on an isolated QR patch.

Applies a random perspective transform (rotation + aspect scale + corner
jitter) to a clean QR patch, producing a warped patch, a warped mask, and
the transformed QR-code corners in warped-patch space.

Functions
---------
sample_patch_ppm
    Sample pixels-per-module uniformly from ``config.ppm_range``.
jitter_corners
    Add uniform random offsets to each of the 4 corners of a rectangle.
perspective_warp
    Warp image + mask by the homography mapping source → destination corners.
apply_augmentation
    Orchestrate the full augmentation step (rotate + aspect + jitter + warp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from qr_reader.synth.config import AugmentationConfig

__all__ = [
    "AugmentedPatch",
    "sample_patch_ppm",
    "jitter_corners",
    "perspective_warp",
    "apply_augmentation",
]

# ---------------------------------------------------------------------------
# AugmentedPatch (data container)
# ---------------------------------------------------------------------------


@dataclass
class AugmentedPatch:
    """Result of applying perspective augmentation to a single QR patch.

    Attributes
    ----------
    warped_patch : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Warped RGB patch image.
    warped_mask : np.ndarray, shape ``(H, W)``, dtype ``float32``
        Warped mask (values in ``[0, 1]``).
    warped_corners_qr : np.ndarray, shape ``(4, 2)``, dtype ``float64``
        Four corners of the QR code proper (TL, TR, BR, BL) in warped-patch
        space coordinates.
    rotation_deg : float
        The rotation angle sampled and applied during augmentation.
    aspect_scale : float
        The aspect scale sampled and applied during augmentation.
    """

    warped_patch: np.ndarray
    warped_mask: np.ndarray
    warped_corners_qr: np.ndarray
    rotation_deg: float = 0.0
    aspect_scale: float = 1.0


# ---------------------------------------------------------------------------
# 2.1  sample_patch_ppm
# ---------------------------------------------------------------------------


def sample_patch_ppm(
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> float:
    """Sample the number of pixels-per-module for the clean QR patch.

    The value is drawn uniformly from ``config.ppm_range``.  It is a
    ``float`` — the caller should round/truncate when passing it to
    :func:`~qr_reader.synth.patch.generate_qr_patch` which requires an
    ``int`` for its ``box_size`` parameter.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random number generator.
    config : AugmentationConfig
        Pipeline configuration (uses ``ppm_range``).

    Returns
    -------
    float
        Sampled pixels-per-module.
    """
    lo, hi = config.ppm_range
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# 2.2  jitter_corners
# ---------------------------------------------------------------------------


def jitter_corners(
    corners_4x2: np.ndarray,
    rng: np.random.Generator,
    jitter_fraction: float,
) -> np.ndarray:
    """Add uniform random offsets to each corner of a rectangle.

    For each corner, the offset in x and y is drawn from
    ``Uniform(-jitter_fraction * side, +jitter_fraction * side)``, where
    *side* is half of the average of the rectangle's width and height.

    No validity checks are performed — the caller is responsible for
    ensuring the result produces a reasonable perspective transform.

    Parameters
    ----------
    corners_4x2 : np.ndarray, shape ``(4, 2)``
        Input corners in TL, TR, BR, BL order.  Should describe a rectangle
        (but any convex quad will work).
    rng : numpy.random.Generator
        Seeded random number generator.
    jitter_fraction : float
        Fraction of the mean side length used as the maximum per-axis offset.

    Returns
    -------
    np.ndarray, shape ``(4, 2)``
        Jittered corners (same order as input).
    """
    # Compute side lengths: width = average of top and bottom edges,
    # height = average of left and right edges.
    top = np.linalg.norm(corners_4x2[1] - corners_4x2[0])  # TL→TR
    bottom = np.linalg.norm(corners_4x2[2] - corners_4x2[3])  # BL→BR
    left = np.linalg.norm(corners_4x2[3] - corners_4x2[0])  # TL→BL
    right = np.linalg.norm(corners_4x2[2] - corners_4x2[1])  # TR→BR

    w = (top + bottom) / 2.0
    h = (left + right) / 2.0
    side = (w + h) / 2.0  # average side length

    max_offset = jitter_fraction * side
    offsets = rng.uniform(-max_offset, max_offset, size=(4, 2))
    return corners_4x2 + offsets


# ---------------------------------------------------------------------------
# 2.3  perspective_warp
# ---------------------------------------------------------------------------


def perspective_warp(
    image: np.ndarray,
    mask: np.ndarray,
    src_corners: np.ndarray,
    dst_corners: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image and its mask by the homography ``src → dst``.

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W, C)``, dtype ``uint8``
        Source image to warp.
    mask : np.ndarray, shape ``(H, W)``, dtype ``float32``
        Source mask to warp.
    src_corners : np.ndarray, shape ``(4, 2)``
        Four source corners in (x, y) order (OpenCV convention).
    dst_corners : np.ndarray, shape ``(4, 2)``
        Four destination corners in (x, y) order.
    output_size : tuple[int, int]
        Output image size as ``(width, height)`` (OpenCV convention).

    Returns
    -------
    warped_image : np.ndarray, shape ``(output_height, output_width, C)``, dtype ``uint8``
    warped_mask : np.ndarray, shape ``(output_height, output_width)``, dtype ``float32``
    """
    H_mat = cv2.getPerspectiveTransform(
        src_corners.astype(np.float32),
        dst_corners.astype(np.float32),
    )

    warped_image = cv2.warpPerspective(
        image,
        H_mat,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_mask = cv2.warpPerspective(
        mask,
        H_mat,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    return warped_image, warped_mask


# ---------------------------------------------------------------------------
# 2.4  apply_augmentation
# ---------------------------------------------------------------------------


def _rotate_point(
    pt: np.ndarray,
    center: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate a single 2D point about a *center* by *angle_rad* radians."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    x_new = cos_a * dx - sin_a * dy + center[0]
    y_new = sin_a * dx + cos_a * dy + center[1]
    return np.array([x_new, y_new], dtype=np.float64)


def _build_target_quad(
    src_corners: np.ndarray,
    center: np.ndarray,
    rotation_deg: float,
    aspect_scale: float,
    rng: np.random.Generator,
    jitter_fraction: float,
) -> np.ndarray:
    """Build the target quad from the source quad through rotate → aspect → jitter."""
    rotation_rad = math.radians(rotation_deg)

    # 1. Rotate each corner around center
    rotated = np.array([_rotate_point(c, center, rotation_rad) for c in src_corners])

    # 2. Aspect scale (independent x/y) around centre.
    #    Scale x by *aspect_scale* and y by its reciprocal so the overall
    #    area is roughly preserved and the aspect ratio changes.
    inv_aspect = 1.0 / aspect_scale
    scaled = np.empty_like(rotated)
    for i, c in enumerate(rotated):
        dx = c[0] - center[0]
        dy = c[1] - center[1]
        scaled[i] = [
            center[0] + dx * aspect_scale,
            center[1] + dy * inv_aspect,
        ]

    # 3. Corner jitter
    jittered = jitter_corners(scaled, rng, jitter_fraction)

    return jittered


def apply_augmentation(
    patch: np.ndarray,
    mask: np.ndarray,
    qr_corners_patch: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> AugmentedPatch:
    """Apply a random perspective augmentation to a clean QR patch.

    The augmentation consists of:
    1. Sample a rotation angle from ``config.rotation_deg_range``.
    2. Sample an aspect scale from ``config.aspect_scale_range``.
    3. Build a target quad by rotating the source quad around its centre,
       applying the aspect scale, and jittering each corner.
    4. Compute the output size as the bounding box of the target quad padded
       by one module-width of pixels.
    5. Warp the patch and mask via :func:`perspective_warp`.
    6. Transform the QR code corners through the same homography.

    Parameters
    ----------
    patch : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Clean QR patch from :func:`~qr_reader.synth.patch.generate_qr_patch`.
    mask : np.ndarray, shape ``(H, W)``, dtype ``float32``
        Mask corresponding to *patch*.
    qr_corners_patch : np.ndarray, shape ``(4, 2)``, dtype ``float64``
        QR code proper corners in patch-space (from
        :func:`~qr_reader.synth.patch.compute_qr_corners_patch_space`).
    rng : numpy.random.Generator
        Seeded random number generator.
    config : AugmentationConfig
        Pipeline configuration (uses ``rotation_deg_range``,
        ``aspect_scale_range``, ``jitter_fraction``).

    Returns
    -------
    AugmentedPatch
        Warped patch, mask, and QR corners in warped-patch space.
    """
    # 1. Sample parameters
    rot_lo, rot_hi = config.rotation_deg_range
    rotation_deg = rng.uniform(rot_lo, rot_hi)

    asp_lo, asp_hi = config.aspect_scale_range
    aspect_scale = rng.uniform(asp_lo, asp_hi)

    # 2. Source quad: the 4 corners of the *patch* rectangle.
    H, W = patch.shape[:2]
    src_quad = np.array(
        [
            [0.0, 0.0],  # TL
            [W, 0.0],  # TR
            [W, H],  # BR
            [0.0, H],  # BL
        ],
        dtype=np.float64,
    )
    center = np.array([W / 2.0, H / 2.0], dtype=np.float64)

    # 3. Build target quad
    dst_quad = _build_target_quad(
        src_quad,
        center,
        rotation_deg,
        aspect_scale,
        rng,
        config.jitter_fraction,
    )

    # 4. Output size: bounding box of target quad with a small padding
    #    to prevent clipping during subsequent feathering (Phase 4).
    #    Since we don't have the exact ppm value here, use 5 % of the
    #    bounding box extent, with a minimum of 4 px.
    xs = dst_quad[:, 0]
    ys = dst_quad[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    pad_x = max(4, (x_max - x_min) * 0.05)
    pad_y = max(4, (y_max - y_min) * 0.05)

    x_min_p = int(math.floor(x_min - pad_x))
    y_min_p = int(math.floor(y_min - pad_y))
    x_max_p = int(math.ceil(x_max + pad_x))
    y_max_p = int(math.ceil(y_max + pad_y))

    out_w = x_max_p - x_min_p
    out_h = y_max_p - y_min_p

    # Shift destination quad into the output image coordinate frame
    dst_quad_shifted = dst_quad - np.array([[x_min_p, y_min_p]], dtype=np.float64)

    # 5. Warp
    warped_patch, warped_mask = perspective_warp(
        patch,
        mask,
        src_quad,
        dst_quad_shifted,
        (out_w, out_h),
    )

    # 6. Warp the QR corners through the same homography
    H_mat = cv2.getPerspectiveTransform(
        src_quad.astype(np.float32),
        dst_quad_shifted.astype(np.float32),
    )

    # Transform each QR corner (needs shape (N, 1, 2) for cv2.perspectiveTransform)
    qr_corners_2d = qr_corners_patch.reshape(1, -1, 2).astype(np.float32)
    warped_qr = cv2.perspectiveTransform(qr_corners_2d, H_mat)
    warped_corners_qr = warped_qr.reshape(4, 2).astype(np.float64)

    return AugmentedPatch(
        warped_patch=warped_patch,
        warped_mask=warped_mask,
        warped_corners_qr=warped_corners_qr,
        rotation_deg=rotation_deg,
        aspect_scale=aspect_scale,
    )
