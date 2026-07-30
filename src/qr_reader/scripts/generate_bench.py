"""Generate composited benchmark images with adjustable contrast and presets.

Usage:
    python generate_bench.py              # generates all 300 images
    python generate_bench.py --count 5    # generates 5 images per config

Images saved to data/bench/synth/images/ with metadata.jsonl.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("data/bench/synth")
BG_DIR = Path("data/images/train")

# ── Versions ──────────────────────────────────────────────────────────────────
VERSIONS = [1, 3, 5, 8, 12, 20]


# ── Updated presets ───────────────────────────────────────────────────────────


@dataclass
class BenchPreset:
    """Benchmark preset with version-aware PPM ranges."""

    name: str
    jitter_fraction: float
    feather_sigma_range: tuple[float, float]
    blur_sigma_range: tuple[float, float]
    noise_sigma_range: tuple[float, float]
    jpeg_quality_range: tuple[int, int]
    min_qr_px: int  # minimum QR side in pixels on final image
    max_qr_px: int  # maximum QR side in pixels on final image

    def target_ppm_range(self, version: int) -> tuple[float, float]:
        N = 17 + 4 * version
        return (self.min_qr_px / N, self.max_qr_px / N)


PRESETS = {
    "easy": BenchPreset(
        name="easy",
        jitter_fraction=0.02,
        feather_sigma_range=(0.5, 1.5),
        blur_sigma_range=(0.0, 0.4),
        noise_sigma_range=(0.0, 2.0),
        jpeg_quality_range=(85, 100),
        min_qr_px=500,
        max_qr_px=1000,
    ),
    "medium": BenchPreset(
        name="medium",
        jitter_fraction=0.08,
        feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.2, 1.0),
        noise_sigma_range=(1.0, 5.0),
        jpeg_quality_range=(65, 95),
        min_qr_px=350,
        max_qr_px=800,
    ),
    "hard": BenchPreset(
        name="hard",
        jitter_fraction=0.15,
        feather_sigma_range=(0.5, 2.5),
        blur_sigma_range=(0.5, 1.5),
        noise_sigma_range=(3.0, 10.0),
        jpeg_quality_range=(45, 85),
        min_qr_px=250,
        max_qr_px=600,
    ),
}

# ── Contrast adjustment ──────────────────────────────────────────────────────


def apply_contrast_adjustment(
    patch: np.ndarray,
    rng: np.random.Generator,
    black_max: float = 0.3,
    white_min: float = 0.7,
) -> np.ndarray:
    """Randomize black/white levels on a clean binary patch.

    Black modules (value 0) become a random gray in [0, black_max].
    White modules (value 255) become a random gray in [white_min, 1.0].

    *patch* is (H, W, 3) uint8 RGB. Modified in place.
    """
    black_val = float(rng.uniform(0, black_max)) * 255.0
    white_val = float(rng.uniform(white_min, 1.0)) * 255.0

    mask_black = (patch == 0).all(axis=2) if patch.ndim == 3 else (patch == 0)
    mask_white = (patch == 255).all(axis=2) if patch.ndim == 3 else (patch == 255)

    if patch.ndim == 3:
        patch[mask_black] = black_val
        patch[mask_white] = white_val
    else:
        patch[mask_black] = black_val
        patch[mask_white] = white_val

    return patch


# ── Main pipeline ────────────────────────────────────────────────────────────


def generate_bench_image(
    rng: np.random.Generator,
    preset: BenchPreset,
    version: int,
    background: np.ndarray,
    seed: int,
    bg_path: str,
) -> tuple[np.ndarray, dict]:
    """Generate one composited QR image with contrast-adjusted patch.

    Same stages as generate_sample, but with contrast adjustment between
    Phase 1 (patch) and Phase 2 (augmentation).
    """
    bg_shape = background.shape[:2]
    N = 17 + 4 * version
    content = f"v{version}"

    # Phase 1: QR patch & mask
    ppm_lo, ppm_hi = preset.target_ppm_range(version)
    ppm_int = int(round(rng.uniform(ppm_lo, ppm_hi)))
    config = AugmentationConfig(
        version=version,
        content=content,
        error_correction="M",
        quiet_zone_modules=4,
    )
    patch, mask = generate_qr_patch(
        version=version, content=content, ecl_str="M",
        ppm=ppm_int, quiet_zone_modules=4,
    )
    qr_corners_patch = compute_qr_corners_patch_space(
        quiet_zone_modules=4, N=N, ppm=ppm_int,
    )

    # Contrast adjustment (between Phase 1 and Phase 2)
    patch = apply_contrast_adjustment(patch, rng)

    # Phase 2: Perspective augmentation
    config.jitter_fraction = preset.jitter_fraction
    augmented = apply_augmentation(patch, mask, qr_corners_patch, rng, config)

    # Phase 3: Placement
    config.target_ppm_range = (ppm_lo, ppm_hi)
    scale, tx, ty = sample_placement_scale(
        rng, augmented.warped_patch.shape[:2], N, config, bg_shape,
    )
    placed = place_patch(augmented, scale, tx, ty, bg_shape)

    # Phase 4: Compositing
    feather_sigma = float(rng.uniform(
        preset.feather_sigma_range[0], preset.feather_sigma_range[1],
    ))
    composited = composite_patch(background, placed, feather_sigma)

    # Phase 5: Global degradation
    config.blur_sigma_range = preset.blur_sigma_range
    config.noise_sigma_range = preset.noise_sigma_range
    config.jpeg_quality_range = preset.jpeg_quality_range
    degraded, deg_params = apply_global_degradation(
        composited.composited_image, rng, config,
    )

    corners = placed.image_corners_qr
    metadata = {
        "seed": seed,
        "preset": preset.name,
        "version": version,
        "background_path": bg_path,
        "payload": content,
        "N": N,
        "pixels_per_module": ppm_int,
        "corners_qr": {
            "TL": [float(corners[0, 0]), float(corners[0, 1])],
            "TR": [float(corners[1, 0]), float(corners[1, 1])],
            "BR": [float(corners[2, 0]), float(corners[2, 1])],
            "BL": [float(corners[3, 0]), float(corners[3, 1])],
        },
        "augmentations": {
            "rotation_deg": float(augmented.rotation_deg),
            "jitter_fraction": preset.jitter_fraction,
            "feather_sigma": feather_sigma,
            "blur_sigma": float(deg_params["blur_sigma"]),
            "noise_sigma": float(deg_params["noise_sigma"]),
            "jpeg_quality": int(deg_params["jpeg_quality"]),
        },
    }

    return degraded, metadata


# ── Dataset generation ──────────────────────────────────────────────────────


def main():
    count_per = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--count" else None

    bg_paths = sorted(BG_DIR.glob("*.jpg"))
    if not bg_paths:
        print(f"ERROR: No background images in {BG_DIR}")
        sys.exit(1)

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    total_per_config = count_per if count_per is not None else 100
    total_images = 0
    meta_lines: list[str] = []

    for preset_name, preset in PRESETS.items():
        n_ver = len(VERSIONS)
        base = total_per_config // n_ver
        rem = total_per_config % n_ver
        counts = [base + (1 if i < rem else 0) for i in range(n_ver)]
        for vi, version in enumerate(VERSIONS):
            n_img = counts[vi]
            for si in range(n_img):
                seed = total_images  # deterministic, reproducible
                bg_path = bg_paths[seed % len(bg_paths)]
                bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
                rng = np.random.default_rng(seed)

                try:
                    img, meta = generate_bench_image(
                        rng, preset, version, bg, seed, bg_path.name,
                    )
                    fname = f"{preset_name}_v{version:02d}_s{si:03d}.jpg"
                    out_path = img_dir / fname
                    cv2.imwrite(
                        str(out_path),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    meta["image_path"] = str(out_path)
                    meta_lines.append(json.dumps(meta))
                    total_images += 1
                except Exception as e:
                    print(f"  ERROR {preset_name} V={version} s={si}: {e}")

    # Write metadata
    meta_path = OUT_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for line in meta_lines:
            f.write(line + "\n")

    print(f"Generated {total_images} images → {img_dir}")
    print(f"Metadata → {meta_path}")
    for preset_name in PRESETS:
        count = sum(1 for l in meta_lines if json.loads(l)["preset"] == preset_name)
        print(f"  {preset_name}: {count} images")


def _compute_per_version(preset: str, total: int, n_versions: int) -> int:
    """Distribute images uniformly across versions."""
    base = total // n_versions
    remainder = total % n_versions
    return base + (1 if remainder > 0 else 0)


if __name__ == "__main__":
    main()
