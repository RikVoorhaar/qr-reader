"""Phase 3 — Placement & Scale.

Scale a warped QR patch (from Phase 2) and place it on a background canvas
so that the QR code modules have approximately the desired pixels-per-module
in the final image.

Functions
---------
sample_placement_scale
    Determine scale and translation to achieve a target PPM in the final image.
place_patch
    Scale and translate an :class:`AugmentedPatch` onto a background-sized canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qr_reader.synth.augment import AugmentedPatch
from qr_reader.synth.config import AugmentationConfig

__all__ = [
    "PlacedPatch",
    "sample_placement_scale",
    "place_patch",
]

# ---------------------------------------------------------------------------
# PlacedPatch (data container)
# ---------------------------------------------------------------------------


@dataclass
class PlacedPatch:
    """Result of placing a warped QR patch onto a background canvas.

    Attributes
    ----------
    full_image : np.ndarray, shape ``(bg_H, bg_W, 3)``, dtype ``uint8``
        The scaled warped patch composited onto a black background canvas of
        the requested background size.
    full_mask : np.ndarray, shape ``(bg_H, bg_W)``, dtype ``float32``
        The scaled warped mask on a black background canvas (zeros outside the
        patch region).
    image_corners_qr : np.ndarray, shape ``(4, 2)``, dtype ``float64``
        Four corners of the QR code proper (TL, TR, BR, BL) in full-image
        space coordinates **(x, y)**.
    """

    full_image: np.ndarray
    full_mask: np.ndarray
    image_corners_qr: np.ndarray


# ---------------------------------------------------------------------------
# 3.1  sample_placement_scale
# ---------------------------------------------------------------------------


def sample_placement_scale(
    rng: np.random.Generator,
    warped_patch_shape: tuple[int, ...],
    N: int,
    config: AugmentationConfig,
    bg_shape: tuple[int, int],
) -> tuple[float, float, float]:
    """Sample a scale factor and translation to place a warped QR patch.

    The scale is chosen so that the QR code modules have approximately
    ``config.target_ppm_range`` pixels per module in the final image.  A
    translation is then sampled uniformly so that the scaled patch is fully
    visible within the background canvas.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random number generator.
    warped_patch_shape : tuple of int
        Shape ``(H, W)`` or ``(H, W, C)`` of the warped patch from Phase 2.
    N : int
        Number of modules per side of the QR code (``N = 17 + 4 * version``).
    config : AugmentationConfig
        Pipeline configuration (uses ``target_ppm_range`` and
        ``quiet_zone_modules``).
    bg_shape : tuple[int, int]
        Background canvas shape as ``(height, width)``.

    Returns
    -------
    scale : float
        Scaling factor (always > 0).
    tx : float
        Horizontal translation in pixels.
    ty : float
        Vertical translation in pixels.
    """
    # 1. Sample target ppm
    lo, hi = config.target_ppm_range
    target_ppm = rng.uniform(lo, hi)

    warped_W = warped_patch_shape[1]
    warped_H = warped_patch_shape[0]

    # 2. Estimate the fraction of warped patch width occupied by QR code proper
    qz = config.quiet_zone_modules
    qr_fraction = N / (N + 2 * qz)

    # 3. Estimate QR width inside the warped patch
    qr_width_in_warped = warped_W * qr_fraction

    # 4. Desired QR width in image space
    target_qr_width = N * target_ppm

    # 5. Scale factor
    scale = target_qr_width / qr_width_in_warped

    # 6. Clamp scale so the patch fits within the background
    bg_H, bg_W = bg_shape
    max_scale_x = bg_W / warped_W
    max_scale_y = bg_H / warped_H
    scale = min(scale, max_scale_x, max_scale_y)

    # 7. Compute translation bounds so patch is fully within background
    scaled_W = warped_W * scale
    scaled_H = warped_H * scale
    max_tx = bg_W - scaled_W
    max_ty = bg_H - scaled_H

    # If bounds are still negative (shouldn't happen after clamping), clamp to 0
    max_tx = max(max_tx, 0.0)
    max_ty = max(max_ty, 0.0)

    # 8. Sample translation
    tx = rng.uniform(0.0, max_tx) if max_tx > 0 else 0.0
    ty = rng.uniform(0.0, max_ty) if max_ty > 0 else 0.0

    return scale, tx, ty


# ---------------------------------------------------------------------------
# 3.2  place_patch
# ---------------------------------------------------------------------------


def place_patch(
    augmented_patch: AugmentedPatch,
    scale: float,
    tx: float,
    ty: float,
    bg_shape: tuple[int, int],
) -> PlacedPatch:
    """Scale and translate a warped QR patch onto a background canvas.

    Parameters
    ----------
    augmented_patch : AugmentedPatch
        Result from :func:`~qr_reader.synth.augment.apply_augmentation`.
    scale : float
        Scaling factor (e.g. from :func:`sample_placement_scale`).
    tx : float
        Horizontal translation in pixels.
    ty : float
        Vertical translation in pixels.
    bg_shape : tuple[int, int]
        Background canvas shape as ``(height, width)``.

    Returns
    -------
    PlacedPatch
        Full-image-sized arrays for the patch, mask, and QR corners.
    """
    H, W = bg_shape  # OpenCV: (output_width, output_height) = (W, H)

    # 1. Build affine matrix (2×3)
    M = np.array(
        [[scale, 0.0, tx], [0.0, scale, ty]],
        dtype=np.float32,
    )

    # 2. Warp patch onto full background canvas
    full_image = cv2.warpAffine(
        augmented_patch.warped_patch,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 3. Warp mask onto full background canvas
    full_mask = cv2.warpAffine(
        augmented_patch.warped_mask,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # 4. Transform QR corners through the same affine transform
    #    cv2.transform expects shape (N, 1, 2)
    corners_input = augmented_patch.warped_corners_qr.reshape(1, -1, 2).astype(
        np.float32
    )
    #    For affine transforms we only need the 2×3 part
    transformed = cv2.transform(corners_input, M)
    image_corners_qr = transformed.reshape(4, 2).astype(np.float64)

    return PlacedPatch(
        full_image=full_image,
        full_mask=full_mask,
        image_corners_qr=image_corners_qr,
    )
