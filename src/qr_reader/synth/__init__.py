"""Synthetic QR augmentation pipeline.

This subpackage provides all phases of the synthetic data generation pipeline:
generating QR patches, applying perspective augmentation, placing patches on
backgrounds, compositing with feathering, applying global degradation, and
orchestrating the full pipeline.

Top-level convenience imports for the most common entry points.
"""

from __future__ import annotations

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
from qr_reader.synth.composite import CompositeResult, composite_patch
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.degrade import (
    apply_global_degradation,
)
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)
from qr_reader.synth.pipeline import generate_dataset, generate_sample
from qr_reader.synth.placement import PlacedPatch, place_patch
from qr_reader.synth.presets import EASY, HARD, MEDIUM, PRESET_MAP

__all__ = [
    "AugmentedPatch",
    "AugmentationConfig",
    "CompositeResult",
    "EASY",
    "HARD",
    "MEDIUM",
    "PlacedPatch",
    "PRESET_MAP",
    "apply_augmentation",
    "apply_global_degradation",
    "composite_patch",
    "compute_qr_corners_patch_space",
    "generate_dataset",
    "generate_qr_patch",
    "generate_sample",
    "place_patch",
]
