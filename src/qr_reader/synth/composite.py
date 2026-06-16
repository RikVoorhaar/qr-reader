"""Phase 4 — Compositing.

Feather a placed QR patch's mask and composite it onto a real background image.

Functions
---------
feather_mask
    Apply Gaussian blur to the outer boundary of the mask.
alpha_composite
    Standard alpha compositing (over operation).
composite_patch
    Orchestrate compositing of a placed patch onto a background.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qr_reader.synth.placement import PlacedPatch

__all__ = [
    "CompositeResult",
    "feather_mask",
    "alpha_composite",
    "composite_patch",
]

# ---------------------------------------------------------------------------
# CompositeResult (data container)
# ---------------------------------------------------------------------------


@dataclass
class CompositeResult:
    """Result of compositing a placed QR patch onto a background image.

    Attributes
    ----------
    composited_image : np.ndarray, shape ``(bg_H, bg_W, 3)``, dtype ``uint8``
        The final composited image — the feathered QR patch alpha-blended
        onto the background.
    image_corners_qr : np.ndarray, shape ``(4, 2)``, dtype ``float64``
        Four corners of the QR code proper (TL, TR, BR, BL) in image-space
        coordinates **(x, y)**.  Passed through unchanged from
        :attr:`PlacedPatch.image_corners_qr`.
    """

    composited_image: np.ndarray
    image_corners_qr: np.ndarray


# ---------------------------------------------------------------------------
# 4.1  feather_mask
# ---------------------------------------------------------------------------


def feather_mask(full_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to the outer boundary of a mask to create a feathered alpha.

    Parameters
    ----------
    full_mask : np.ndarray, shape ``(H, W)``, dtype ``float32``
        Binary or soft mask with values in ``[0, 1]``.
    sigma : float
        Gaussian blur sigma in pixels.  ``sigma=0`` returns the mask unchanged.
        OpenCV computes the kernel size automatically when *ksize* is ``(0, 0)``.

    Returns
    -------
    np.ndarray, shape ``(H, W)``, dtype ``float32``
        Feathered alpha map with values clipped to ``[0, 1]``.
    """
    if sigma <= 0.0:
        return full_mask.astype(np.float32, copy=False)

    alpha = cv2.GaussianBlur(
        full_mask,
        (0, 0),  # automatic kernel size
        sigmaX=sigma,
        sigmaY=sigma,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


# ---------------------------------------------------------------------------
# 4.2  alpha_composite
# ---------------------------------------------------------------------------


def alpha_composite(
    background: np.ndarray,
    patch_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Standard alpha compositing (over operation).

    .. math::

        \\text{result} = \\alpha \\cdot \\text{patch} + (1 - \\alpha) \\cdot \\text{background}

    All arrays must have the same spatial dimensions.  The *background* and
    *patch_rgb* may be ``uint8`` or ``float32``; the result is always ``uint8``.

    Parameters
    ----------
    background : np.ndarray, shape ``(H, W, 3)``
        Background image.
    patch_rgb : np.ndarray, shape ``(H, W, 3)``
        Foreground patch image (already sized and positioned to match the
        background, e.g. via :class:`~qr_reader.synth.placement.PlacedPatch`).
    alpha : np.ndarray, shape ``(H, W)``, dtype ``float32``
        Alpha mask — values in ``[0, 1]`` (e.g. from :func:`feather_mask`).

    Returns
    -------
    np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Composited image clipped to ``[0, 255]``.
    """
    # Convert inputs to float32 for computation
    bg_f = background.astype(np.float32, copy=False)
    patch_f = patch_rgb.astype(np.float32, copy=False)

    # Alpha compositing
    alpha_3d = alpha[..., np.newaxis]  # (H, W, 1) — broadcasts over 3 channels
    result_f = alpha_3d * patch_f + (1.0 - alpha_3d) * bg_f

    # Clip and convert back to uint8
    result = np.clip(result_f, 0.0, 255.0).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# 4.3  composite_patch
# ---------------------------------------------------------------------------


def composite_patch(
    background: np.ndarray,
    placed_patch: PlacedPatch,
    feather_sigma: float,
) -> CompositeResult:
    """Orchestrate compositing of a placed QR patch onto a background image.

    Parameters
    ----------
    background : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Background image (same size as the background used during placement).
    placed_patch : PlacedPatch
        Result from :func:`~qr_reader.synth.placement.place_patch`.
    feather_sigma : float
        Gaussian blur sigma for feathering the mask edge.
        Passed to :func:`feather_mask`.

    Returns
    -------
    CompositeResult
        Composited image and pass-through QR corners.
    """
    alpha = feather_mask(placed_patch.full_mask, feather_sigma)
    composited_image = alpha_composite(background, placed_patch.full_image, alpha)
    return CompositeResult(
        composited_image=composited_image,
        image_corners_qr=placed_patch.image_corners_qr,
    )
