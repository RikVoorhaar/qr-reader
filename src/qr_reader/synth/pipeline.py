"""Phase 6 — Pipeline Orchestrator.

Orchestrates the full augmentation pipeline from QR generation through to
global degradation, returning both the final image and a metadata dictionary
ready for serialisation.

Functions
---------
generate_sample
    Generate a single augmented QR sample on a given background image.
generate_dataset
    Batch-generate multiple samples, saving images and metadata to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from qr_reader.synth.augment import apply_augmentation
from qr_reader.synth.composite import composite_patch
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.degrade import apply_global_degradation
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)
from qr_reader.synth.placement import place_patch, sample_placement_scale

__all__ = [
    "generate_sample",
    "generate_dataset",
]

# ---------------------------------------------------------------------------
# 6.1  generate_sample
# ---------------------------------------------------------------------------


def generate_sample(
    rng: np.random.Generator,
    config: AugmentationConfig,
    background: np.ndarray,
    sample_index: int = 0,
    background_path: str = "",
) -> tuple[np.ndarray, dict]:
    """Generate a single augmented QR sample on a given background image.

    The pipeline runs all five phases in sequence:

    1. Generate a clean QR patch and mask (Phase 1).
    2. Apply perspective augmentation (Phase 2).
    3. Place the warped patch on the background canvas (Phase 3).
    4. Alpha-composite with feathered mask (Phase 4).
    5. Apply global degradation (Phase 5).

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random number generator for all stochastic steps.
    config : AugmentationConfig
        Pipeline configuration.
    background : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Background image (RGB).  The composited patch is placed onto this
        canvas at its native resolution.
    sample_index : int
        Zero-based sample index (included in the metadata dict).
    background_path : str
        Optional filesystem path to the background image (included in the
        metadata dict for provenance).

    Returns
    -------
    image : np.ndarray, shape ``(H, W, 3)``, dtype ``uint8``
        Final composited and degraded RGB image (same shape as *background*).
    metadata : dict
        Serializable dictionary with all generation parameters.
    """
    bg_shape = background.shape[:2]
    N = 17 + 4 * config.version  # modules per side

    # --- Phase 1: QR patch & mask ---
    ppm_int = int(round(sample_patch_ppm(rng, config)))
    patch, mask = generate_qr_patch(
        version=config.version,
        content=config.content,
        ecl_str=config.error_correction,
        ppm=ppm_int,
        quiet_zone_modules=config.quiet_zone_modules,
    )
    qr_corners_patch = compute_qr_corners_patch_space(
        quiet_zone_modules=config.quiet_zone_modules,
        N=N,
        ppm=ppm_int,
    )

    # --- Phase 2: Perspective augmentation ---
    augmented = apply_augmentation(patch, mask, qr_corners_patch, rng, config)

    # --- Phase 3: Placement ---
    scale, tx, ty = sample_placement_scale(
        rng, augmented.warped_patch.shape[:2], N, config, bg_shape
    )
    placed = place_patch(augmented, scale, tx, ty, bg_shape)

    # --- Phase 4: Compositing ---
    feather_sigma = rng.uniform(
        config.feather_sigma_range[0], config.feather_sigma_range[1]
    )
    composited = composite_patch(background, placed, feather_sigma)

    # --- Phase 5: Global degradation ---
    degraded, deg_params = apply_global_degradation(
        composited.composited_image, rng, config
    )

    # --- Build metadata ---
    corners = placed.image_corners_qr
    metadata = {
        "sample_index": sample_index,
        "seed": config.global_seed + sample_index,
        "background_path": background_path,
        "payload": config.content,
        "version": config.version,
        "N": N,
        "ecl": config.error_correction,
        "pixels_per_module": ppm_int,
        "corners_qr": {
            "TL": [float(corners[0, 0]), float(corners[0, 1])],
            "TR": [float(corners[1, 0]), float(corners[1, 1])],
            "BR": [float(corners[2, 0]), float(corners[2, 1])],
            "BL": [float(corners[3, 0]), float(corners[3, 1])],
        },
        "augmentations": {
            "rotation_deg": float(augmented.rotation_deg),
            "jitter_fraction": float(config.jitter_fraction),
            "aspect_scale": float(augmented.aspect_scale),
            "feather_sigma": float(feather_sigma),
            "blur_sigma": float(deg_params["blur_sigma"]),
            "noise_sigma": float(deg_params["noise_sigma"]),
            "jpeg_quality": int(deg_params["jpeg_quality"]),
        },
    }

    return degraded, metadata


# ---------------------------------------------------------------------------
# 6.2  generate_dataset
# ---------------------------------------------------------------------------


def generate_dataset(
    config: AugmentationConfig,
    background_dir: str | Path,
    output_dir: str | Path,
    num_samples: int,
) -> None:
    """Batch-generate QR samples over backgrounds and save to disk.

    For each sample:
    1. Pick a background image (cycled round-robin from the background
       directory).
    2. Seed ``rng`` from ``config.global_seed + sample_index``.
    3. Call :func:`generate_sample`.
    4. Save the image as ``output_dir/images/{sample_index:06d}.jpg``.
    5. Append the metadata line to ``output_dir/metadata.jsonl``.

    Parameters
    ----------
    config : AugmentationConfig
        Pipeline configuration.
    background_dir : str or Path
        Directory containing background images (``.jpg``, ``.png``, ``.jpeg``).
    output_dir : str or Path
        Output directory.  Subdirectories ``images/`` will be created.
    num_samples : int
        Number of samples to generate.

    Raises
    ------
    FileNotFoundError
        If *background_dir* contains no image files.
    """
    background_dir = Path(background_dir)
    output_dir = Path(output_dir)

    # Collect background paths
    bg_extensions = {".jpg", ".jpeg", ".png"}
    bg_paths = sorted(
        p for p in background_dir.iterdir() if p.suffix.lower() in bg_extensions
    )
    if not bg_paths:
        raise FileNotFoundError(
            f"No image files found in {background_dir} (supported: {bg_extensions})"
        )

    # Create output directories
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"

    print(f"Generating {num_samples} samples → {output_dir}")

    for sample_index in range(num_samples):
        if sample_index > 0 and sample_index % 100 == 0:
            print(f"  {sample_index}/{num_samples}...")

        # Pick background (round-robin)
        bg_path = bg_paths[sample_index % len(bg_paths)]

        # Load background
        bg_bgr = cv2.imread(str(bg_path))
        if bg_bgr is None:
            continue  # skip unreadable images silently
        background = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

        # Seed rng
        rng = np.random.default_rng(config.global_seed + sample_index)

        # Generate sample
        image, metadata = generate_sample(
            rng=rng,
            config=config,
            background=background,
            sample_index=sample_index,
            background_path=str(bg_path),
        )

        # Save image
        image_path = images_dir / f"{sample_index:06d}.jpg"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Append metadata
        with open(metadata_path, "a") as f:
            f.write(json.dumps(metadata) + "\n")

    print(f"Done — {num_samples} samples written to {output_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def sample_patch_ppm(
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> float:
    """Sample pixels-per-module for the clean QR patch as a float.

    Returns a float; the caller should round/truncate to int when passing
    to :func:`~qr_reader.synth.patch.generate_qr_patch`.
    """
    lo, hi = config.ppm_range
    return rng.uniform(lo, hi)
