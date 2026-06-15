"""Augmentation pipeline configuration.

This module defines the single configuration dataclass used by every phase of
the augmentation pipeline.  It is deliberately kept as a plain ``dataclass``
rather than a pydantic model to avoid an extra dependency; if serialisation or
schema validation are needed later, swapping to pydantic is a localised change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AugmentationConfig",
]

# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for the full synthetic QR augmentation pipeline.

    Every parameter has a sensible default so that most callers only need to
    override the fields that differ from the defaults.

    Parameters
    ----------
    version : int
        QR code version (1–40).  Default ``5``.
    global_seed : int
        Base seed for dataset generation; each sample uses ``global_seed + index``.
        Default ``42``.
    content : str
        Text payload to encode.  Default ``"QR Reader v1"``.
    error_correction : str
        Error correction level — one of ``"L"``, ``"M"``, ``"Q"``, ``"H"``.
        Default ``"M"``.
    quiet_zone_modules : int
        Width of the quiet zone in modules (QR spec minimum is 4).
        Default ``4``.
    ppm_range : tuple[float, float]
        Range for sampling *pixels-per-module* when generating the clean patch.
        Default ``(3.0, 20.0)``.
    rotation_deg_range : tuple[float, float]
        Range for the perspective rotation angle in degrees.
        Default ``(0.0, 360.0)``.
    jitter_fraction : float
        Fraction of a side length used as the max offset when jittering
        perspective corners.  Default ``0.15``.
    aspect_scale_range : tuple[float, float]
        Range for the independent x/y aspect scale before jitter.
        Default ``(0.8, 1.2)``.
    target_ppm_range : tuple[float, float]
        Desired *pixels-per-module* in the final composited image (used by
        Phase 3 placement).  Default ``(4.0, 12.0)``.
    feather_sigma_range : tuple[float, float]
        Range for Gaussian feather (blur) sigma on the mask edge (Phase 4).
        Default ``(0.5, 2.5)``.
    global_seed : int
        Base seed for dataset generation; each sample uses ``global_seed + index``.
        Default ``42``.
    blur_sigma_range : tuple[float, float]
        Range for post-composite Gaussian blur sigma (Phase 5).
        Default ``(0.0, 1.5)``.
    noise_sigma_range : tuple[float, float]
        Range for post-composite additive Gaussian noise sigma (Phase 5).
        Default ``(0.0, 8.0)``.
    jpeg_quality_range : tuple[int, int]
        Range for post-composite JPEG compression quality (Phase 5).
        Default ``(50, 100)``.
    """

    # QR code parameters
    version: int = 5
    content: str = "QR Reader v1"
    error_correction: str = "M"

    # Patch generation
    quiet_zone_modules: int = 4
    ppm_range: tuple[float, float] = (3.0, 20.0)

    # Augmentation — perspective jitter
    rotation_deg_range: tuple[float, float] = (0.0, 360.0)
    jitter_fraction: float = 0.15
    aspect_scale_range: tuple[float, float] = (0.8, 1.2)

    # Placement
    target_ppm_range: tuple[float, float] = (4.0, 12.0)

    # Feathering
    feather_sigma_range: tuple[float, float] = (0.5, 2.5)  # px

    # Dataset generation
    global_seed: int = 42  # base seed for dataset iteration

    # Global degradation (post-composite)
    blur_sigma_range: tuple[float, float] = (0.0, 1.5)
    noise_sigma_range: tuple[float, float] = (0.0, 8.0)
    jpeg_quality_range: tuple[int, int] = (50, 100)
