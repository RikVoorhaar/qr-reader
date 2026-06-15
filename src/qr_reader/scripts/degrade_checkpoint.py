"""Phase 5 Deliverable Checkpoint — Global Degradation visual inspection.

Takes 5 composited images (generated on-the-fly) and applies degradation at
easy / medium / hard settings.  Produces side-by-side comparisons so the user
can visually confirm that the QR is still readable and the degradation looks
realistic.

Usage::

    python src/qr_reader/scripts/degrade_checkpoint.py \\
        [--outdir /tmp/degrade_checkpoint] \\
        [--backgrounds-dir data/images/train]

Human inspection criteria:
- Easy: almost indistinguishable from the original composited image.
- Medium: visible blur, noise, and compression but QR is easily readable.
- Hard: significant degradation but QR modules are still discernible.
- All settings should look realistic (not obviously synthetic).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
from qr_reader.synth.composite import CompositeResult, composite_patch
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.degrade import apply_global_degradation
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)
from qr_reader.synth.placement import PlacedPatch, place_patch, sample_placement_scale


def _N(version: int) -> int:
    return 17 + 4 * version


def _make_composited(
    seed: int,
    version: int,
    content: str,
    bg_path: str,
    bg_shape: tuple[int, int],
) -> CompositeResult:
    """Helper: generate a composited image from scratch (reuses full pipeline)."""
    rng = np.random.default_rng(seed)

    config = AugmentationConfig(
        version=version,
        content=content,
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(10.0, 10.0),
        rotation_deg_range=(15.0, 15.0),
        jitter_fraction=0.05,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(6.0, 6.0),
        feather_sigma_range=(1.5, 1.5),
    )

    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(f"Cannot read background: {bg_path}")
    bg_H, bg_W = bg_shape

    ppm_int = int(config.ppm_range[0])
    N = _N(config.version)

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

    augmented: AugmentedPatch = apply_augmentation(
        patch, mask, qr_corners_patch, rng, config
    )
    scale, tx, ty = sample_placement_scale(
        rng,
        augmented.warped_patch.shape[:2],
        N,
        config,
        (bg_H, bg_W),
    )
    placed: PlacedPatch = place_patch(augmented, scale, tx, ty, (bg_H, bg_W))
    return composite_patch(bg, placed, config.feather_sigma_range[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/tmp/degrade_checkpoint",
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--backgrounds-dir",
        default="data/images/train",
        help="Directory containing background JPEG images",
    )
    args = parser.parse_args()

    # Resolve backgrounds directory relative to repo root
    bg_dir = Path(args.backgrounds_dir)
    if not bg_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        bg_dir = repo_root / bg_dir

    if not bg_dir.is_dir():
        print(f"Error: backgrounds directory not found: {bg_dir}", file=sys.stderr)
        sys.exit(1)

    bg_paths = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.jpeg"))
    if not bg_paths:
        print(f"Error: no .jpg images found in {bg_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(bg_paths)} background images in {bg_dir}")
    os.makedirs(args.outdir, exist_ok=True)

    # --- Difficulty presets ---
    presets = {
        "easy": AugmentationConfig(
            blur_sigma_range=(0.3, 0.3),
            noise_sigma_range=(1.0, 1.0),
            jpeg_quality_range=(95, 95),
        ),
        "medium": AugmentationConfig(
            blur_sigma_range=(0.8, 0.8),
            noise_sigma_range=(3.0, 3.0),
            jpeg_quality_range=(75, 75),
        ),
        "hard": AugmentationConfig(
            blur_sigma_range=(1.4, 1.4),
            noise_sigma_range=(7.0, 7.0),
            jpeg_quality_range=(50, 50),
        ),
    }

    # 5 samples: varied versions, content, backgrounds
    samples_config = [
        (42, 3, "Hello QR", bg_paths[0]),
        (142, 5, "Medium QR", bg_paths[1 % len(bg_paths)]),
        (242, 7, "Longer text", bg_paths[2 % len(bg_paths)]),
        (342, 10, "Version 10", bg_paths[3 % len(bg_paths)]),
        (442, 1, "v1 small", bg_paths[4 % len(bg_paths)]),
    ]

    print()
    header = (
        f"{'#':<4} {'Version':<8} {'Content':<16} {'Difficulty':<10} {'Background':<20}"
    )
    print(header)
    print("-" * len(header))

    for sample_idx, (seed, version, content, bg_rel_path) in enumerate(samples_config):
        bg_path = str(bg_rel_path)

        # Read background to get shape
        bg = cv2.imread(bg_path)
        if bg is None:
            print(f"  Skipping sample {sample_idx}: cannot read {bg_path}")
            continue
        bg_H, bg_W = bg.shape[:2]
        bg_label = os.path.splitext(os.path.basename(bg_rel_path))[0]

        # Generate the composited image
        composite_result = _make_composited(
            seed, version, content, bg_path, (bg_H, bg_W)
        )
        base_image = composite_result.composited_image

        # Build a row for this sample: original + easy + medium + hard
        row_parts: list[np.ndarray] = [base_image]

        for diff_name in ("easy", "medium", "hard"):
            diff_cfg = presets[diff_name]
            degraded = apply_global_degradation(
                base_image, np.random.default_rng(seed), diff_cfg
            )

            # Add label
            labeled = degraded.copy()
            cv2.putText(
                labeled,
                diff_name.upper(),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            row_parts.append(labeled)

        # Horizontal stack: original | easy | medium | hard
        comparison = np.hstack(row_parts)

        out_path = os.path.join(
            args.outdir, f"sample_{sample_idx}_v{version}_{bg_label}.png"
        )
        cv2.imwrite(out_path, comparison)

        print(f"{sample_idx:<4} {version:<8} {content:<16} {'all':<10} {bg_label:<20}")

    print()
    print(f"Generated {len(samples_config)} comparison panels in {args.outdir}")
    print()
    print("Open each panel to inspect.")
    print()
    print("Checklist for each sample:")
    print("  - EASY:  barely any visible degradation")
    print("  - MEDIUM: blur + noise + compression visible, QR still clear")
    print("  - HARD:  significant degradation, QR modules still discernible")
    print("  - All levels should look realistic (not obviously synthetic)")


if __name__ == "__main__":
    main()
