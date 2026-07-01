"""I7 — Measure TLS-normal drift from Hough-normal in C failures vs. successes.

For each GT edge with a matching Hough peak, this script records:
- Hough peak normal angle (degrees).
- TLS-refined normal angle (degrees).
- Angular drift (degrees, mod π).
- Whether the edge exhibits a C failure (span too long).

Key question: Do C-failure edges have systematically larger TLS drift
than C-success edges?  If yes, Phase 12 (Hough-normal-based support
collection) is worth trying.
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.hough import hough_vote_peaks, refine_line
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
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


def main() -> None:
    print("=" * 70)
    print("I7 — TLS drift measurement")
    print("=" * 70)
    print()

    bg = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, CONFIG, bg)

    roi_results = _run_pipeline_to_rois(image)

    if len(roi_results) == 0:
        print("No clusters found.")
        return

    all_drifts_c_fail: list[float] = []
    all_drifts_c_pass: list[float] = []

    for roi, nms, angle, bbox, ci in roi_results:
        normals, rhos, scores = hough_vote_peaks(nms, angle)

        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        print(f"--- Cluster {ci} ---")
        print(f"  {'GT edge':>12s}  {'Hough °':>8s}  {'TLS °':>8s}  "
              f"{'drift°':>7s}  {'C fail?':>8s}  {'span_ratio':>10s}  {'notes'}")

        for gt in gt_edges:
            if gt["segment"] is None:
                continue

            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                # D failure — no peak to drift from
                continue

            # Hough peak normal angle
            hough_n = normals[match_idx]
            hough_deg = np.rad2deg(np.arctan2(hough_n[1], hough_n[0])) % 180

            # Refine line to get TLS normal
            seg = refine_line(
                hough_n, float(rhos[match_idx]), float(scores[match_idx]),
                nms, angle, gap_tolerance=2.0, distance_thresh=1.5,
            )
            tls_n = seg.normal
            tls_deg = np.rad2deg(np.arctan2(tls_n[1], tls_n[0])) % 180

            drift = _angular_distance_deg(hough_n, tls_n)

            # Check C failure
            gt_seg = gt["segment"]
            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
            gt_proj = gt_seg @ direction
            gt_span = abs(float(gt_proj[1] - gt_proj[0]))

            degenerate = np.all(seg.endpoints == 0)
            if degenerate:
                c_fail = "DEGEN"
                span_ratio = 0.0
            else:
                ep_proj = seg.endpoints @ direction
                seg_span = abs(float(ep_proj[1] - ep_proj[0]))
                span_ratio = seg_span / gt_span if gt_span > 0 else 0.0

                # C failure check
                c_fail = False
                for gt_ep in gt_seg:
                    dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
                    if dists.min() > 5.0:
                        c_fail = True
                        break

            notes = ""
            if drift > 2.0:
                notes = f"← large drift (>2°)"
            if degenerate:
                notes = "degenerate"
            elif c_fail:
                notes = f"C FAIL  {notes}"
            else:
                notes = f"pass    {notes}"

            print(f"  {gt['label']:>12s}  {hough_deg:>8.2f}  {tls_deg:>8.2f}  "
                  f"{drift:>7.3f}  {str(c_fail) if not degenerate else 'DEGEN':>8s}  "
                  f"{span_ratio:>10.2f}  {notes}")

            if degenerate:
                continue
            if c_fail:
                all_drifts_c_fail.append(drift)
            else:
                all_drifts_c_pass.append(drift)

        print()

    # Summary statistics
    print("=" * 70)
    print("DRIFT STATISTICS")
    print("=" * 70)

    def describe(arr: list[float], label: str) -> None:
        if len(arr) == 0:
            print(f"  {label}: (no samples)")
            return
        a = np.array(arr)
        print(f"  {label}: n={len(a)}, mean={a.mean():.3f}°, "
              f"median={np.median(a):.3f}°, max={a.max():.3f}°, "
              f"min={a.min():.3f}°")

    describe(all_drifts_c_fail, "C-failure drifts")
    describe(all_drifts_c_pass, "C-success drifts")

    print()
    if len(all_drifts_c_fail) > 0 and len(all_drifts_c_pass) > 0:
        c_fail_arr = np.array(all_drifts_c_fail)
        c_pass_arr = np.array(all_drifts_c_pass)
        if c_fail_arr.mean() > c_pass_arr.mean() + 0.5:
            print("  ✓ C-failure drifts are systematically larger than C-success drifts.")
            print("  → Phase 12 (Hough-normal-based support collection) is worth trying.")
        else:
            print("  ✗ C-failure drifts are NOT systematically larger.")
            print("  → TLS drift is not the root cause. Skip Phase 12.")
    else:
        print("  Insufficient data for comparison.")
    print()

    # Distribution by cluster
    print("=" * 70)
    print("DRIFT DISTRIBUTION BY EDGE TYPE")
    print("=" * 70)
    print()
    # Group by edge orientation
    for gt in gt_edges:
        if gt["segment"] is None:
            continue
        edge_type = gt["label"].split("_")[1]  # "top", "left", "right", "bottom"
        # Just do a final summary
    print("(see per-edge data above for detailed drift values)")
    print()
    print("End I7 — TLS drift measurement")


if __name__ == "__main__":
    main()
