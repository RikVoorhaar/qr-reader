"""Phase 3 Deliverable Checkpoint — Placement & Scale visual inspection.

Generates 9 placement samples: 3 background resolutions × 3 scale/translation
configurations per background.  For each sample it:
1. Generates and augments a QR patch.
2. Places it on a black canvas of the given background size.
3. Overlays the 4 QR corners (TL/TR/BR/BL) as coloured dots.
4. Saves the placed image, the mask, and the corner overlay.

Usage::

    python src/qr_reader/scripts/placement_checkpoint.py [--outdir /tmp/placement_checkpoint]

Human inspection criteria:
- QR code is fully visible (no clipping at edges).
- Patches appear at different positions and scales per background.
- Corners are coloured dots on the QR code boundary.
- Masks are properly scaled and positioned.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
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
        cv2.circle(out, (cx, cy), 6, colour, -1)
        cv2.putText(
            out,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/tmp/placement_checkpoint",
        help="Output directory for result images",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Background resolutions (height, width)
    bg_shapes = [
        (1920, 1280),  # landscape
        (1280, 1920),  # portrait
        (1920, 1440),  # near-square (1080p + bars)
    ]

    # A single QR configuration with some variety
    # Use 3 different scale/translation configs by varying target_ppm_range
    ppm_patterns = [
        (4.0, 4.0),  # small in image
        (8.0, 8.0),  # medium
        (16.0, 16.0),  # large (should nearly fill bg)
    ]
    scale_labels = ["small", "medium", "large"]

    qr_config = AugmentationConfig(
        version=5,
        content="Placement Checkpoint v5",
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(10.0, 10.0),
        rotation_deg_range=(25.0, 25.0),
        jitter_fraction=0.05,
        aspect_scale_range=(1.1, 1.1),
    )

    ppm_int = int(qr_config.ppm_range[0])
    N = _N(qr_config.version)

    print(f"Output directory: {args.outdir}")
    print()
    print(f"{'#':<4} {'Background':<20} {'Scale':<8} {'Target PPM':<12}")
    print("-" * 50)

    sample_idx = 0
    for bg_idx, bg_shape in enumerate(bg_shapes):
        for ppm_idx, target_ppm_range in enumerate(ppm_patterns):
            config = AugmentationConfig(
                version=qr_config.version,
                content=qr_config.content,
                ppm_range=qr_config.ppm_range,
                rotation_deg_range=qr_config.rotation_deg_range,
                jitter_fraction=qr_config.jitter_fraction,
                aspect_scale_range=qr_config.aspect_scale_range,
                target_ppm_range=target_ppm_range,
            )

            rng = np.random.default_rng(seed=sample_idx * 100 + 42)

            # Generate patch
            patch, mask = generate_qr_patch(
                version=config.version,
                content=config.content,
                ecl_str=config.error_correction,
                ppm=ppm_int,
                quiet_zone_modules=config.quiet_zone_modules,
            )
            qr_corners = compute_qr_corners_patch_space(
                quiet_zone_modules=config.quiet_zone_modules,
                N=N,
                ppm=ppm_int,
            )

            # Augment
            augmented: AugmentedPatch = apply_augmentation(
                patch, mask, qr_corners, rng, config
            )

            # Sample placement
            scale, tx, ty = sample_placement_scale(
                rng,
                augmented.warped_patch.shape[:2],
                N,
                config,
                bg_shape,
            )

            # Place
            placed: PlacedPatch = place_patch(augmented, scale, tx, ty, bg_shape)

            # Overlay corners
            overlay = draw_corners(placed.full_image, placed.image_corners_qr)

            # Save
            bg_label = f"{bg_shape[1]}x{bg_shape[0]}"
            stem = f"sample_{sample_idx:02d}_{bg_label}_{scale_labels[ppm_idx]}"
            cv2.imwrite(os.path.join(args.outdir, f"{stem}_rgb.png"), placed.full_image)
            cv2.imwrite(
                os.path.join(args.outdir, f"{stem}_mask.png"),
                placed.full_mask * 255,
            )
            cv2.imwrite(os.path.join(args.outdir, f"{stem}_corners.png"), overlay)

            print(
                f"{sample_idx:<4} {bg_label:<20} {scale:<8.3f} "
                f"{target_ppm_range[0]:<12.1f}"
            )

            sample_idx += 1

    print(f"\nGenerated {sample_idx} placement samples in {args.outdir}")
    print("Open these files to visually inspect:")
    print("  - _rgb.png: QR patch placed on black canvas")
    print("  - _mask.png: grayscale mask showing patch region")
    print("  - _corners.png: overlay with TL/TR/BR/BL dots at module corners")
    print()
    print("Check:")
    print("  - QR code is fully visible (no clipping)")
    print("  - Patches appear at different positions")
    print("  - 'small' patches leave lots of black background")
    print("  - 'large' patches fill most of the canvas")
    print("  - Corner markers sit on the QR code boundary")


if __name__ == "__main__":
    main()
