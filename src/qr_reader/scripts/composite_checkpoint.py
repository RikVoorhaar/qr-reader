"""Phase 4 Deliverable Checkpoint — Compositing visual inspection.

Generates 10 composited QR-on-background images across versions 1–15, using
real background images from the HomeObjects-3K dataset.  For each sample:
1. Generates and augments a QR patch.
2. Places it on a background-sized canvas.
3. Feathers the mask and composites onto a real background image.
4. Also saves a version with corner markers for inspection.
5. Saves a side-by-side comparison (feathered vs hard edge) for the first sample.

Usage::

    python src/qr_reader/scripts/composite_checkpoint.py \\
        [--outdir /tmp/composite_checkpoint] \\
        [--backgrounds-dir data/images/train] \\
        [--num-samples 10]

Human inspection criteria:
- QR code is visible and readable.
- Edges blend smoothly into the background (no hard rectangular border).
- Corner markers (TL/TR/BR/BL) sit at the QR code boundary.
- Different backgrounds, QR versions, and scales produce varied outputs.
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
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)
from qr_reader.synth.placement import PlacedPatch, place_patch, sample_placement_scale


def _N(version: int) -> int:
    return 17 + 4 * version


def draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw the 4 QR corners as coloured dots on a copy of *img*."""
    out = img.copy()
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
    labels = ["TL", "TR", "BR", "BL"]
    for pt, colour, label in zip(corners, colours, labels):
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (cx, cy), 8, colour, -1)
        cv2.putText(
            out,
            label,
            (cx + 10, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/tmp/composite_checkpoint",
        help="Output directory for result images",
    )
    parser.add_argument(
        "--backgrounds-dir",
        default="data/images/train",
        help="Directory containing background JPEG images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of composited samples to generate",
    )
    args = parser.parse_args()

    # Resolve backgrounds directory relative to repo root
    bg_dir = Path(args.backgrounds_dir)
    if not bg_dir.is_absolute():
        # Assume it's relative to repo root
        repo_root = Path(__file__).resolve().parents[3]
        bg_dir = repo_root / bg_dir

    if not bg_dir.is_dir():
        print(
            f"Error: backgrounds directory not found: {bg_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect background image paths
    bg_paths = sorted(bg_dir.glob("*.jpg")) + sorted(bg_dir.glob("*.jpeg"))
    if not bg_paths:
        print(f"Error: no .jpg images found in {bg_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(bg_paths)} background images in {bg_dir}")

    os.makedirs(args.outdir, exist_ok=True)

    # QR versions to use (spread across 1–15)
    versions = [1, 3, 5, 7, 10, 12, 15]

    # Varying feather sigmas
    feather_sigmas = [0.0, 1.0, 3.0]

    print()
    print(
        f"{'#':<4} {'Version':<8} {'Feather σ':<10} {'Background':<20} {'Target PPM':<12}"
    )
    print("-" * 60)

    rng = np.random.default_rng(42)

    for idx in range(args.num_samples):
        seed = idx * 100 + 42
        sample_rng = np.random.default_rng(seed)

        version = versions[idx % len(versions)]
        feather_sigma = feather_sigmas[idx % len(feather_sigmas)]
        bg_rel_path = bg_paths[idx % len(bg_paths)]
        bg_path = str(bg_rel_path)

        # Parse background dimensions for the config
        bg = cv2.imread(bg_path)
        if bg is None:
            print(f"  Warning: couldn't read {bg_path}, skipping")
            continue
        bg_H, bg_W = bg.shape[:2]

        # Config with slight randomness to produce variety
        config = AugmentationConfig(
            version=version,
            content=f"v{version} sample {idx}",
            error_correction="M",
            quiet_zone_modules=4,
            ppm_range=(10.0, 10.0),
            rotation_deg_range=(float(sample_rng.uniform(0, 45)),) * 2,
            jitter_fraction=0.05,
            aspect_scale_range=(float(sample_rng.uniform(0.9, 1.1)),) * 2,
            target_ppm_range=(float(sample_rng.uniform(4.0, 12.0)),) * 2,
        )

        ppm_int = int(config.ppm_range[0])
        N = _N(config.version)

        # 1. Generate patch
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

        # 2. Augment
        augmented: AugmentedPatch = apply_augmentation(
            patch, mask, qr_corners_patch, sample_rng, config
        )

        # 3. Place
        scale, tx, ty = sample_placement_scale(
            sample_rng,
            augmented.warped_patch.shape[:2],
            N,
            config,
            (bg_H, bg_W),
        )
        placed: PlacedPatch = place_patch(augmented, scale, tx, ty, (bg_H, bg_W))

        # 4. Composite
        composite_result: CompositeResult = composite_patch(bg, placed, feather_sigma)

        # 5. Make corner overlay
        overlay = draw_corners(
            composite_result.composited_image,
            composite_result.image_corners_qr,
        )

        # 6. Save three views
        bg_label = os.path.splitext(os.path.basename(bg_rel_path))[0]
        stem = f"sample_{idx:02d}_v{version}_feather{feather_sigma:.1f}_{bg_label}"

        # 6a. Composite result
        cv2.imwrite(
            os.path.join(args.outdir, f"{stem}_composited.png"),
            composite_result.composited_image,
        )

        # 6b. Corner overlay
        cv2.imwrite(os.path.join(args.outdir, f"{stem}_corners.png"), overlay)

        # 6c. Hard-edge comparison (feather_sigma=0) — only if feather_sigma > 0
        if feather_sigma > 0:
            hard_result = composite_patch(bg, placed, 0.0)
            comparison = np.hstack(
                [
                    hard_result.composited_image,
                    composite_result.composited_image,
                ]
            )
            # Add labels
            cv2.putText(
                comparison,
                "Hard edge (σ=0)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                comparison,
                f"Feathered (σ={feather_sigma})",
                (bg_W + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                os.path.join(args.outdir, f"{stem}_comparison.png"),
                comparison,
            )

        print(
            f"{idx:<4} {version:<8} {feather_sigma:<10.1f} {bg_label:<20} "
            f"{config.target_ppm_range[0]:<12.1f}"
        )

    print()
    print(f"Generated {args.num_samples} composite samples in {args.outdir}")
    print()
    print("Open these files to visually inspect:")
    print("  - _composited.png: QR composited onto real background")
    print("  - _corners.png: overlay with TL/TR/BR/BL dots at module corners")
    print("  - _comparison.png (if feather_sigma > 0): hard edge vs feathered edge")
    print()
    print("Check:")
    print("  - QR code blends into the background naturally")
    print("  - No hard rectangular border around the QR")
    print("  - Corner markers sit on the QR code boundary")
    print("  - Different backgrounds/versions/scales produce variety")


if __name__ == "__main__":
    main()
