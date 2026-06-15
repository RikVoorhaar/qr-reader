"""Phase 2 Deliverable Checkpoint — Perspective Augmentation visual inspection.

Generates 5 QR patches at different versions/ppms, applies augmentation with
varying rotation and jitter, and saves the warped patch RGB, warped mask, and
an overlay showing the 4 QR corners as coloured dots.

Usage::

    python src/qr_reader/scripts/augment_checkpoint.py [--outdir /tmp/augment_checkpoint]

Human inspection criteria:
- QR code is readable (the warped patch still looks like a QR code).
- Corners align to the QR code proper boundary.
- Mask aligns to the patch (same shape, covers the QR region).
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from qr_reader.synth.augment import apply_augmentation
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)


def _N(version: int) -> int:
    return 17 + 4 * version


def draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw the 4 QR corners as coloured dots on a copy of *img*."""
    out = img.copy()
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
    labels = ["TL", "TR", "BR", "BL"]
    for pt, colour, label in zip(corners, colours, labels):
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (cx, cy), 4, colour, -1)
        cv2.putText(
            out,
            label,
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/tmp/augment_checkpoint",
        help="Output directory for result images",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 5 test cases with varying versions, ppms, rotation and jitter
    test_cases = [
        AugmentationConfig(
            version=1,
            content="Hello v1",
            rotation_deg_range=(20.0, 20.0),
            jitter_fraction=0.05,
            aspect_scale_range=(1.0, 1.0),
            ppm_range=(10.0, 10.0),
        ),
        AugmentationConfig(
            version=5,
            content="QR Reader checkpoint v5",
            rotation_deg_range=(45.0, 45.0),
            jitter_fraction=0.1,
            aspect_scale_range=(1.15, 1.15),
            ppm_range=(8.0, 8.0),
        ),
        AugmentationConfig(
            version=10,
            content="Version 10 augmentation test with longer payload",
            rotation_deg_range=(90.0, 90.0),
            jitter_fraction=0.08,
            aspect_scale_range=(0.85, 0.85),
            ppm_range=(6.0, 6.0),
        ),
        AugmentationConfig(
            version=3,
            content="Jitter + rotation combo test",
            rotation_deg_range=(30.0, 30.0),
            jitter_fraction=0.15,
            aspect_scale_range=(1.0, 1.0),
            ppm_range=(12.0, 12.0),
        ),
        AugmentationConfig(
            version=7,
            content="Mixed v7 with aspect warp and jitter",
            rotation_deg_range=(15.0, 15.0),
            jitter_fraction=0.12,
            aspect_scale_range=(0.9, 0.9),
            ppm_range=(7.0, 7.0),
        ),
    ]

    print(f"Output directory: {args.outdir}")
    print(
        f"{'#':<5} {'Version':<10} {'PPM':<6} {'Rotation':<10} {'Jitter':<10} {'Aspect':<8}"
    )
    print("-" * 60)

    for i, cfg in enumerate(test_cases):
        ppm_int = int(cfg.ppm_range[0])
        N = _N(cfg.version)
        rng = np.random.default_rng(seed=i * 100 + 42)

        # Generate clean patch
        patch, mask = generate_qr_patch(
            version=cfg.version,
            content=cfg.content,
            ecl_str=cfg.error_correction,
            ppm=ppm_int,
            quiet_zone_modules=cfg.quiet_zone_modules,
        )

        # Compute QR corners in patch space
        qr_corners = compute_qr_corners_patch_space(
            quiet_zone_modules=cfg.quiet_zone_modules,
            N=N,
            ppm=ppm_int,
        )

        # Apply augmentation
        result = apply_augmentation(patch, mask, qr_corners, rng, cfg)

        # Overlay corners
        overlay = draw_corners(result.warped_patch, result.warped_corners_qr)

        # Save
        stem = f"sample_{i:02d}_v{cfg.version}"
        cv2.imwrite(os.path.join(args.outdir, f"{stem}_rgb.png"), result.warped_patch)
        cv2.imwrite(
            os.path.join(args.outdir, f"{stem}_mask.png"), result.warped_mask * 255
        )
        cv2.imwrite(os.path.join(args.outdir, f"{stem}_corners.png"), overlay)

        print(
            f"{i:<5} {cfg.version:<10} {ppm_int:<6} "
            f"{cfg.rotation_deg_range[0]:<10} {cfg.jitter_fraction:<10} "
            f"{cfg.aspect_scale_range[0]:<8}"
        )

    print(f"\nGenerated {len(test_cases)} samples in {args.outdir}")
    print("Open these files to visually inspect:")
    print("  - _rgb.png: warped patch (should look like a QR code)")
    print("  - _mask.png: warped mask (bright region matches the patch)")
    print("  - _corners.png: overlay with TL/TR/BR/BL dots at module corners")


if __name__ == "__main__":
    main()
