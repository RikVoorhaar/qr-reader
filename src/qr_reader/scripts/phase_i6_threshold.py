"""I6 — Measure threshold_rel sensitivity for D failures.

For each D-failure cluster, this script extracts peaks at multiple
threshold_rel values and reports:
1. Number of peaks extracted at each threshold.
2. Whether any new peak matches a D-failure GT edge.
3. Whether any new peak is a B phantom (wrong-angle, non-finder cluster).
4. Also runs on v12-clean to check for phantom regressions.

Decision: If any threshold surfaces D edges without new B phantoms,
Phase 11 is worth trying.
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.hough import hough_vote_peaks, refine_line
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from debug_hough_failures import _compute_accumulator, _simulate_peak_suppression
from test_hough_harness import (
    _angular_distance_deg,
    _compute_finder_edges,
    _make_background,
    _match_peak,
    _run_pipeline_to_rois,
)

CONFIG = AugmentationConfig(
    version=12,
    content="https://www.rikvoorhaar.com",
    error_correction="M",
    ppm_range=(5.0, 12.0),
    target_ppm_range=(4.0, 10.0),
    jitter_fraction=0.15,
    feather_sigma_range=(0.5, 2.0),
    blur_sigma_range=(0.2, 1.0),
    noise_sigma_range=(1.0, 5.0),
    jpeg_quality_range=(65, 95),
    global_seed=42,
)

CONFIG_CLEAN = AugmentationConfig(
    version=12,
    content="https://www.rikvoorhaar.com",
    error_correction="M",
    ppm_range=(10.0, 10.0),
    rotation_deg_range=(0.0, 0.0),
    jitter_fraction=0.0,
    aspect_scale_range=(1.0, 1.0),
    target_ppm_range=(10.0, 10.0),
    feather_sigma_range=(0.5, 0.5),
    blur_sigma_range=(0.0, 0.0),
    noise_sigma_range=(0.0, 0.0),
    jpeg_quality_range=(100, 100),
    global_seed=42,
)

THRESHOLDS = [0.25, 0.20, 0.15, 0.10]


def _count_phantoms(
    normals: np.ndarray,
    rhos: np.ndarray,
    scores: np.ndarray,
    nms: np.ndarray,
    angle: np.ndarray,
    gt_edges: list[dict],
) -> int:
    """Count how many peaks are B phantoms."""
    gt_normals = np.array(
        [e["normal"] for e in gt_edges if e["segment"] is not None]
    )
    phantom_count = 0
    for i in range(len(normals)):
        matched = any(
            gt["segment"] is not None
            and _match_peak(gt, normals[[i]], rhos[[i]]) >= 0
            for gt in gt_edges
        )
        if matched:
            continue
        if len(gt_normals) > 0:
            min_ang = min(
                _angular_distance_deg(normals[i], gn) for gn in gt_normals
            )
            if min_ang < 12.0:
                continue
        seg = refine_line(
            normals[i], float(rhos[i]), float(scores[i]), nms, angle,
            gap_tolerance=2.0, distance_thresh=1.5,
        )
        if np.all(seg.endpoints == 0):
            continue
        ys, xs = np.nonzero(np.asarray(nms))
        strengths = nms[ys, xs]
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        dists = np.abs(points @ seg.normal - seg.rho)
        mask = dists < 1.5
        mean_str = float(strengths[mask].mean()) if np.sum(mask) > 0 else 0.0
        if mean_str > 400:
            phantom_count += 1
    return phantom_count


def analyze_config(config: AugmentationConfig, name: str) -> None:
    print(f"=== {name} ===")
    bg = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, config, bg)

    roi_results = _run_pipeline_to_rois(image)

    if len(roi_results) == 0:
        print("  No clusters found.")
        return

    # Row: threshold. Column: per-cluster stats
    print(f"  {'thresh':>7}  {'cluster':>7}  {'peaks':>5}  {'GT_in':>5}  "
          f"{'D_r6':>4}  {'D_new':>4}  {'phantoms':>7}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*7}")

    for thr in THRESHOLDS:
        for roi, nms, angle, bbox, ci in roi_results:
            normals, rhos, scores = hough_vote_peaks(
                nms, angle, threshold_rel=thr
            )
            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),
                roi_shape=roi.shape,
            )
            n_gt = sum(1 for gt in gt_edges if gt["segment"] is not None)
            n_d_r6 = sum(
                1 for gt in gt_edges
                if gt["segment"] is not None and _match_peak(gt, normals, rhos) < 0
            )

            # Count new matches vs baseline (threshold_rel=0.25)
            if thr == 0.25:
                n_d_new = 0
            else:
                # Compare to baseline normals+rhos
                normals_25, rhos_25, _ = hough_vote_peaks(nms, angle, threshold_rel=0.25)
                # New D matches = D failures in baseline that are now matched
                n_d_new = 0
                for gt in gt_edges:
                    if gt["segment"] is None:
                        continue
                    if _match_peak(gt, normals_25, rhos_25) < 0 and _match_peak(gt, normals, rhos) >= 0:
                        n_d_new += 1

            n_phantom = _count_phantoms(normals, rhos, scores, nms, angle, gt_edges)

            print(f"  {thr:>7.2f}  C{ci:<6}  {len(normals):>5}  {n_gt:>5}  "
                  f"{n_d_r6:>4}  {n_d_new:>4}  {n_phantom:>7}")
        print()

    # Summary
    print(f"  Summary for {name}:")
    for thr in THRESHOLDS:
        total_peaks = 0
        total_phantoms = 0
        total_d = 0
        for roi, nms, angle, bbox, ci in roi_results:
            normals, rhos, scores = hough_vote_peaks(
                nms, angle, threshold_rel=thr
            )
            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),
                roi_shape=roi.shape,
            )
            total_peaks += len(normals)
            total_phantoms += _count_phantoms(
                normals, rhos, scores, nms, angle, gt_edges
            )
            total_d += sum(
                1 for gt in gt_edges
                if gt["segment"] is not None and _match_peak(gt, normals, rhos) < 0
            )
        print(f"    threshold_rel={thr:.2f}: {total_peaks:>3} peaks, "
              f"{total_d:>2} D failures, {total_phantoms:>2} phantoms")
    print()


def main() -> None:
    print("=" * 70)
    print("I6 — Threshold_rel sensitivity measurement")
    print("=" * 70)
    print()

    analyze_config(CONFIG, "v12-default")
    analyze_config(CONFIG_CLEAN, "v12-clean")

    print("=" * 70)
    print("Decision guidance:")
    thresholds = [0.25, 0.20, 0.15, 0.10]
    print(f"  If any threshold >0.10 surfaces new D matches "
          f"without new B phantoms → Phase 11 is worth trying.")
    print(f"  If lowering threshold only surfaces phantoms → skip Phase 11.")
    print("=" * 70)


if __name__ == "__main__":
    main()
