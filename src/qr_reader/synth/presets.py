"""Phase 7.1 — Difficulty presets for the augmentation pipeline.

Provides three named :class:`AugmentationConfig` presets (EASY, MEDIUM, HARD)
that control the difficulty of the generated synthetic QR samples.  These are
sensible defaults; callers can override individual fields after selecting a
preset.
"""

from __future__ import annotations

from qr_reader.synth.config import AugmentationConfig

__all__ = [
    "EASY",
    "MEDIUM",
    "HARD",
    "PRESET_MAP",
]

# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

EASY = AugmentationConfig(
    ppm_range=(8.0, 16.0),
    target_ppm_range=(8.0, 16.0),
    jitter_fraction=0.05,
    feather_sigma_range=(0.5, 1.5),
    blur_sigma_range=(0.0, 0.4),
    noise_sigma_range=(0.0, 2.0),
    jpeg_quality_range=(85, 100),
)

MEDIUM = AugmentationConfig(
    ppm_range=(5.0, 12.0),
    target_ppm_range=(4.0, 10.0),
    jitter_fraction=0.15,
    feather_sigma_range=(0.5, 2.0),
    blur_sigma_range=(0.2, 1.0),
    noise_sigma_range=(1.0, 5.0),
    jpeg_quality_range=(65, 95),
)

HARD = AugmentationConfig(
    ppm_range=(3.0, 8.0),
    target_ppm_range=(2.5, 6.0),
    jitter_fraction=0.25,
    feather_sigma_range=(0.5, 2.5),
    blur_sigma_range=(0.5, 1.5),
    noise_sigma_range=(3.0, 10.0),
    jpeg_quality_range=(45, 85),
)

# ---------------------------------------------------------------------------
# Lookup map for CLI use
# ---------------------------------------------------------------------------

PRESET_MAP: dict[str, AugmentationConfig] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
}
