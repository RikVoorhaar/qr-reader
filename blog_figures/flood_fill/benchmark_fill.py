#!/usr/bin/env python3
"""Time the old flood-fill corner pipeline on the two blog-figure cutouts.

They are exactly the success and failure crops produced by
``blog_figures/flood_fill/make_figures.py``: a 432x432 crop of
``qr_code_ - 6.jpeg`` cluster 0 and a 333x333 crop of cluster 2.

The timings compare the current NumPy wave-front fill with two SciPy
equivalents -- mask-based ``binary_propagation`` from a seed and whole-crop
``label`` plus the seed component -- then measure boundary extraction,
boundary connected components, angular-NMS corner selection, and Otsu
binarization for context.

Usage (from the repository root):

    .venv/bin/python blog_figures/flood_fill/benchmark_fill.py

This file only measures; it does not modify the figure script, the GIFs, or
anything outside ``blog_figures/flood_fill/``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import make_figures as fig  # noqa: E402

from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.detector.corner import angular_nms_top_radial_indices  # noqa: E402
from qr_reader.detector.region import (  # noqa: E402
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.qr_gen import binarize_image  # noqa: E402


def scipy_fill(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """8-connected equivalent of ``region_fill_wave_front``."""
    structure = ndimage.generate_binary_structure(2, 2)
    target = bool(binary[seed])
    labels, _ = ndimage.label(binary == target, structure=structure)
    return labels == labels[seed]


def propagation_fill(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """8-connected fill via ``scipy.ndimage.binary_propagation``.

    This is the mask-based propagation the project actually used before
    falling back to the hand-written wave front: start from a one-pixel seed
    mask, restrict growth to pixels with the same binary value, and let
    ``binary_propagation`` iterate to convergence.
    """
    structure = ndimage.generate_binary_structure(2, 2)
    seed_mask = np.zeros_like(binary, dtype=bool)
    seed_mask[seed] = True
    target_mask = binary == binary[seed]
    return ndimage.binary_propagation(seed_mask, structure=structure, mask=target_mask)


def timed_trials(fn, *, trials: int, target_trial_s: float = 0.03) -> dict:
    """Warm up, choose an inner-loop count, then time ``trials`` batches.

    Each batch is divided by its inner-loop count, so the recorded values are
    per-call milliseconds.  The median/IQR are across batches.
    """
    for _ in range(5):
        fn()

    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    inner = max(1, int(math.ceil(target_trial_s / max(single, 1e-9))))

    values = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        values.append((time.perf_counter() - t0) / inner * 1000.0)

    values = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "median_ms": float(np.median(values)),
        "iqr_ms": float(q3 - q1),
        "trials": trials,
        "inner": inner,
    }


def corner_inputs(components: list[list[tuple[int, int]]]) -> list[tuple]:
    """Precompute radial/angle arrays for the pure NMS timing."""
    inputs = []
    for comp in components:
        arr = np.asarray(comp, dtype=np.float64)
        if arr.shape[0] < 4:
            continue
        centroid = arr.mean(axis=0)
        radial = np.linalg.norm(arr - centroid, axis=1)
        angles = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        inputs.append((radial, angles))
    return inputs


def measure_example(ex: fig.Example, binary: np.ndarray) -> dict:
    r0, r1, c0, c1 = ex.crop_box
    crop = np.ascontiguousarray(binary[r0:r1, c0:c1])
    seed = ex.seed
    truth = ex.region

    results = {}

    # 1. Current hand-coded NumPy wave front.
    results["wave_fill"] = timed_trials(
        lambda: region_fill_wave_front(crop, seed[0], seed[1]),
        trials=61,
    )

    # 2. SciPy whole-crop label + seed component.
    results["scipy_fill"] = timed_trials(
        lambda: scipy_fill(crop, seed),
        trials=61,
    )

    # 3. SciPy mask-based propagation from a one-pixel seed.
    results["propagation_fill"] = timed_trials(
        lambda: propagation_fill(crop, seed),
        trials=61,
    )

    wave_mask = region_fill_wave_front(crop, seed[0], seed[1])
    scipy_mask = scipy_fill(crop, seed)
    propagation_mask = propagation_fill(crop, seed)
    results["equivalence"] = {
        "wave_matches_prepared": bool(np.array_equal(wave_mask, truth)),
        "scipy_matches_wave": bool(np.array_equal(scipy_mask, wave_mask)),
        "propagation_matches_wave": bool(np.array_equal(propagation_mask, wave_mask)),
        "mask_pixels": int(wave_mask.sum()),
    }

    # 3. Boundary trace of the wave-front mask.
    results["boundary_8"] = timed_trials(
        lambda: region_boundary_8(wave_mask),
        trials=101,
    )

    boundary = region_boundary_8(wave_mask)
    components = boundary_connected_components_ndimage(np.asarray(boundary))

    # 4. Boundary connected components (includes Python list construction).
    results["boundary_components"] = timed_trials(
        lambda: boundary_connected_components_ndimage(np.asarray(boundary)),
        trials=101,
    )

    # 5. Angular NMS on the corner inputs (preprocessing excluded).
    nms_inputs = corner_inputs(components)

    def run_nms():
        for radial, angles in nms_inputs:
            angular_nms_top_radial_indices(
                radial,
                angles,
                angular_nms_rad=fig.ANGULAR_NMS_RAD,
                k=4,
            )

    results["angular_nms"] = timed_trials(run_nms, trials=101)

    def run_corner_selection():
        for comp in components:
            arr = np.asarray(comp, dtype=np.float64)
            centroid = arr.mean(axis=0)
            radial = np.linalg.norm(arr - centroid, axis=1)
            angles = np.arctan2(
                arr[:, 1] - centroid[1], arr[:, 0] - centroid[0]
            )
            angular_nms_top_radial_indices(
                radial,
                angles,
                angular_nms_rad=fig.ANGULAR_NMS_RAD,
                k=4,
            )

    results["corner_selection_total"] = timed_trials(
        run_corner_selection, trials=101
    )

    # 6. Otsu binarization on the same crop.
    results["binarize"] = timed_trials(
        lambda: binarize_image(ex.gray),
        trials=101,
    )

    results["_meta"] = {
        "shape": ex.gray.shape,
        "seed": seed,
        "mask_pixels": int(wave_mask.sum()),
        "n_components": len(components),
        "n_nms_components": len(nms_inputs),
    }
    return results


def main() -> None:
    image_path = fig.DEFAULT_DATA_DIR / fig.IMAGE_NAME
    gray_full = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
    binary_full = binarize_image(gray_full)
    rows, cols = find_alignment_patterns_2d(binary_full, np.log(1.3))
    clusters = cluster_candidates(rows, cols)

    examples = [
        fig.prepare_example(
            gray_full,
            binary_full,
            fig.IMAGE_NAME,
            fig.SUCCESS_CLUSTER,
            clusters[fig.SUCCESS_CLUSTER],
            name="success",
        ),
        fig.prepare_example(
            gray_full,
            binary_full,
            fig.IMAGE_NAME,
            fig.FAILURE_CLUSTER,
            clusters[fig.FAILURE_CLUSTER],
            name="failure",
        ),
    ]

    print(f"Image: {image_path}")
    print("Per-call milliseconds; median and IQR across timed batches.")
    print(f"Fill operations use 61 trials; fast operations use 101 trials.")

    for ex in examples:
        res = measure_example(ex, binary_full)
        meta = res["_meta"]
        print(
            f"\n## {ex.name} cutout {meta['shape'][1]}x{meta['shape'][0]} "
            f"(cluster {ex.cluster_index}, crop {ex.crop_box})"
        )
        eq = res["equivalence"]
        print(
            f"Equivalence: wave==prepared {eq['wave_matches_prepared']}, "
            f"scipy==wave {eq['scipy_matches_wave']}, "
            f"propagation==wave {eq['propagation_matches_wave']}, "
            f"mask pixels {eq['mask_pixels']}; "
            f"boundary components {meta['n_components']}."
        )
        print()
        print("| operation | median ms | IQR ms | trials | inner |")
        print("|---|---:|---:|---:|---:|")
        for key, label in [
            ("wave_fill", "region_fill_wave_front"),
            ("propagation_fill", "ndimage.binary_propagation fill"),
            ("scipy_fill", "ndimage.label fill"),
            ("boundary_8", "region_boundary_8"),
            ("boundary_components", "boundary_connected_components_ndimage"),
            ("angular_nms", "angular_nms_top_radial_indices (all comps)"),
            ("corner_selection_total", "corner selection incl. centroid/radial/angle"),
            ("binarize", "binarize_image"),
        ]:
            r = res[key]
            print(
                f"| {label} | {r['median_ms']:.4f} | {r['iqr_ms']:.4f} "
                f"| {r['trials']} | {r['inner']} |"
            )
        ratio = res["propagation_fill"]["median_ms"] / res["wave_fill"]["median_ms"]
        label_ratio = res["scipy_fill"]["median_ms"] / res["wave_fill"]["median_ms"]
        print(
            f"\nbinary_propagation / wave fill median ratio: {ratio:.3f}x "
            f"(~{1 / ratio:.0f}x faster)"
        )
        print(
            f"label / wave fill median ratio: {label_ratio:.3f}x "
            f"(~{1 / label_ratio:.0f}x faster)"
        )


if __name__ == "__main__":
    main()
