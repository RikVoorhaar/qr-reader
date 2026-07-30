"""QR detection/decoding benchmark with preset-based parameter sweep.

Sweeps 3 presets (easy, medium, hard) × 6 versions (1, 3, 5, 8, 12, 20) × 10 seeds
= 180 test images generated via generate_test_image.

Outputs benchmark_results.json with per-image metrics and prints a summary table.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from qr_reader.detector.detector import detect_corners, detect_homography, detect_sample
from qr_reader.decoder.decoder import decode as decode_qr
from qr_reader.qr_gen import generate_test_image, make_qr_image

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
# Each preset defines the distortion parameters passed to generate_test_image.
# All presets use random rotation (rotation_angle_deg=None).
@dataclass
class Preset:
    name: str
    noise_std: float
    perspective_max_shift: float
    final_blur_kernel: int
    intensity_scale: float = 0.8
    noise_blur_kernel: int = 3


PRESETS = {
    "easy": Preset("easy", noise_std=5, perspective_max_shift=5, final_blur_kernel=3),
    "medium": Preset("medium", noise_std=25, perspective_max_shift=20, final_blur_kernel=5),
    "hard": Preset("hard", noise_std=50, perspective_max_shift=50, final_blur_kernel=7),
}

VERSIONS = [1, 3, 5, 8, 12, 20]
NUM_SEEDS = 10
CONTENT = "https://www.rikvoorhaar.com"
BOX_SIZE = 10
BORDER = 4

# ---------------------------------------------------------------------------
# Image result container
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    preset: str
    version: int
    seed: int
    detection_time_s: float = 0.0
    detection_success: bool = False
    detection_error: str = ""
    corner_error_px: float = -1.0
    decode_time_s: float = 0.0
    decode_success: bool = False
    decode_text: str = ""
    total_time_s: float = 0.0
    detected_version: int = -1


@dataclass
class BenchmarkRun:
    config: dict[str, Any] = field(default_factory=dict)
    results: list[ImageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ground-truth corner extraction
# ---------------------------------------------------------------------------


def _compute_gt_corners(
    version: int,
    content: str,
    box_size: int,
    border: int,
    seed: int,
    perspective_max_shift: float,
) -> np.ndarray:
    """Return (4, 2) GT QR corners in (x, y) for the image produced by
    generate_test_image with the given seed and params.

    Reproduces exactly the same random transforms (rotation + perspective)
    that generate_test_image applies, then carries the clean QR corners
    through the transform chain.
    """
    rng = np.random.default_rng(seed)

    clean = make_qr_image(content=content, version=version, box_size=box_size, border=border)
    h, w = clean.shape

    # Corners of the QR code region in the clean image (before transforms)
    corner_px = border * box_size
    corners_clean = np.array(
        [
            [corner_px, corner_px],
            [w - corner_px - 1, corner_px],
            [w - corner_px - 1, h - corner_px - 1],
            [corner_px, h - corner_px - 1],
        ],
        dtype=np.float64,
    )

    # --- rotation (same random draw as in generate_test_image) ---
    rotation_amount = rng.uniform(0, 2 * np.pi)
    angle_deg = float(np.rad2deg(rotation_amount))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    pts = corners_clean.reshape(1, -1, 2)
    corners_rot = cv2.transform(pts.astype(np.float32), M_rot).reshape(4, 2).astype(np.float64)

    # --- perspective warp (same per-corner shifts as in generate_test_image) ---
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    shifts = rng.uniform(0, perspective_max_shift, size=(4, 2)).astype(np.float32)
    dst_pts = src_pts + shifts
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, w - 1)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, h - 1)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    pts2 = corners_rot.reshape(1, -1, 2).astype(np.float32)
    corners_final = cv2.perspectiveTransform(pts2, M).reshape(4, 2).astype(np.float64)

    return corners_final


def _corner_error(detected: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-corner Euclidean distance between detected and GT corners."""
    # Match corners by nearest neighbour (handles ordering differences)
    dists = np.full(4, np.inf)
    used = set()
    for i in range(4):
        for j in range(4):
            if j in used:
                continue
            d = float(np.linalg.norm(detected[i] - gt[j]))
            if d < dists[i]:
                dists[i] = d
                used.add(j)
    # Reset and do it properly with minimum sum assignment
    best = np.inf
    from itertools import permutations

    for perm in permutations(range(4)):
        d = sum(float(np.linalg.norm(detected[p] - gt[q])) for p, q in enumerate(perm))
        if d < best:
            best = d
    return best / 4.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(output_json: str) -> None:
    """Run the full sweep and write results to *output_json*."""
    results: list[ImageResult] = []
    total_configs = len(PRESETS) * len(VERSIONS) * NUM_SEEDS

    print(f"Benchmarking {total_configs} configs (3 presets × {len(VERSIONS)} versions × {NUM_SEEDS} seeds)...")
    print()

    n = 0
    for preset_name, preset in PRESETS.items():
        for version in VERSIONS:
            for seed in range(NUM_SEEDS):
                n += 1
                result = ImageResult(preset=preset_name, version=version, seed=seed)

                try:
                    # Generate test image
                    t0 = perf_counter()
                    img = generate_test_image(
                        seed=seed,
                        content=CONTENT,
                        version=version,
                        box_size=BOX_SIZE,
                        border=BORDER,
                        noise_std=preset.noise_std,
                        perspective_max_shift=preset.perspective_max_shift,
                        final_blur_kernel=preset.final_blur_kernel,
                        intensity_scale=preset.intensity_scale,
                        noise_blur_kernel=preset.noise_blur_kernel,
                    )

                    # Compute ground truth corners
                    gt_corners = _compute_gt_corners(
                        version=version,
                        content=CONTENT,
                        box_size=BOX_SIZE,
                        border=BORDER,
                        seed=seed,
                        perspective_max_shift=preset.perspective_max_shift,
                    )

                    # Detection
                    t0 = perf_counter()
                    detected_corners, detected_version = detect_corners(img)
                    t1 = perf_counter()
                    result.detection_time_s = t1 - t0
                    result.detection_success = True
                    result.detected_version = detected_version
                    result.corner_error_px = _corner_error(detected_corners, gt_corners)

                    # Decoding
                    try:
                        bits = detect_sample(img)
                        t2 = perf_counter()
                        result.decode_time_s = t2 - t1
                        decoded = decode_qr(bits)
                        result.decode_success = (decoded == CONTENT)
                        result.decode_text = decoded
                    except Exception as dec_err:
                        result.decode_time_s = perf_counter() - t1
                        result.decode_success = False
                        result.decode_text = str(dec_err)

                    result.total_time_s = perf_counter() - t0

                except Exception as det_err:
                    result.detection_success = False
                    result.detection_error = str(det_err)
                    result.total_time_s = perf_counter() - t0

                results.append(result)

                # Progress indicator
                pct = 100 * n / total_configs
                status = "✓" if result.detection_success else "✗"
                decode_status = "✓" if result.decode_success else "✗"
                print(
                    f"[{n:3d}/{total_configs}] {preset_name:>6s} V{version:2d} s{seed:02d}  "
                    f"det={status} v={result.detected_version:2d} err={result.corner_error_px:5.1f}px  "
                    f"dec={decode_status}  {result.total_time_s:5.2f}s"
                )

    # Serialize
    output: dict[str, Any] = {
        "config": {
            "presets": {name: asdict(p) for name, p in PRESETS.items()},
            "versions": VERSIONS,
            "seeds": NUM_SEEDS,
            "content": CONTENT,
            "box_size": BOX_SIZE,
            "border": BORDER,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {output_json}")
    print_summary(results)


def print_summary(results: list[ImageResult]) -> None:
    """Print a markdown summary table."""
    from collections import defaultdict

    # Group by (preset, version)
    groups = defaultdict(list)
    for r in results:
        groups[(r.preset, r.version)].append(r)

    print("\n## Benchmark Summary\n")
    print("| Preset | Version | Detect % | Corner Err (px) | Decode % | Time (ms) |")
    print("|--------|---------|----------|-----------------|----------|-----------|")

    summary_by_preset: dict[str, list[float]] = defaultdict(list)

    for preset_name in ["easy", "medium", "hard"]:
        for version in VERSIONS:
            grp = groups[(preset_name, version)]
            n = len(grp)
            det_ok = sum(1 for r in grp if r.detection_success)
            dec_ok = sum(1 for r in grp if r.decode_success)
            corner_errs = [r.corner_error_px for r in grp if r.detection_success]
            times = [r.total_time_s for r in grp if r.detection_success]

            mean_err = np.mean(corner_errs) if corner_errs else float("nan")
            mean_time = np.mean(times) * 1000 if times else float("nan")

            det_pct = det_ok / n * 100
            dec_pct = dec_ok / n * 100

            print(
                f"| {preset_name:>6s} | {version:7d} | "
                f"{det_pct:5.1f}% ({det_ok}/{n}) | "
                f"{mean_err:15.1f} | "
                f"{dec_pct:5.1f}% ({dec_ok}/{n}) | "
                f"{mean_time:9.1f} |"
            )
            summary_by_preset[preset_name].append(det_pct)

    # Aggregate by preset
    print("|--------|---------|----------|-----------------|----------|-----------|")
    for preset_name in ["easy", "medium", "hard"]:
        avg_det = np.mean(summary_by_preset[preset_name])
        print(f"| **{preset_name:>6s}** | **avg** | **{avg_det:5.1f}%** | | | |")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "benchmark_current.json"
    run_benchmark(output)
