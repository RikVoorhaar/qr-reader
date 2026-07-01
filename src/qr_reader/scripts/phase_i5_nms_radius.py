"""I5 — Measure NMS radius sensitivity for D failures.

For each D-failure GT edge, this script measures:
1. The GT rho bin and its accumulator score.
2. The peak accumulator score within ±6 rho bins (nms_radius_rho=6).
3. The peak accumulator score within ±3 rho bins (nms_radius_rho=3, proposed).
4. Whether reducing nms_radius_rho to 3 would surface the true-edge peak.

Key question: Is the D failure caused by the true-edge bin being within
nms_radius_rho=6 of a stronger competitor and getting suppressed?
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.hough import hough_vote_peaks
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from debug_hough_failures import _compute_accumulator, _match_peak
from test_hough_harness import (
    _compute_finder_edges,
    _make_background,
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

RHO_STEP = 1.0


def main() -> None:
    print("=" * 70)
    print("I5 — NMS radius sensitivity measurement")
    print("=" * 70)
    print()

    bg = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, CONFIG, bg)

    roi_results = _run_pipeline_to_rois(image)

    if len(roi_results) == 0:
        print("No clusters found.")
        return

    for roi, nms, angle, bbox, ci in roi_results:
        print(f"--- Cluster {ci} (ROI shape={nms.shape}) ---")

        acc, theta_step, n_theta, n_rho = _compute_accumulator(nms, angle)
        normals, rhos, scores = hough_vote_peaks(nms, angle)

        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        for gt in gt_edges:
            if gt["segment"] is None:
                continue

            match_idx = _match_peak(gt, normals, rhos)
            if match_idx >= 0:
                continue  # Not a D failure

            # D-failure edge
            gt_rho_bin = int(round(gt["rho"] / RHO_STEP))
            gt_theta_rad = np.arctan2(gt["normal"][1], gt["normal"][0]) % np.pi
            gt_t_idx = int(np.round(gt_theta_rad / theta_step)) % n_theta

            # Scores in the GT theta bin
            gt_bin_score = float(acc[gt_t_idx, gt_rho_bin]) if 0 <= gt_rho_bin < n_rho else 0.0

            # Window of ±6 bins (current nms_radius_rho)
            r0_6 = max(0, gt_rho_bin - 6)
            r1_6 = min(n_rho, gt_rho_bin + 7)
            band_6 = acc[gt_t_idx, r0_6:r1_6]
            peak_bin_6 = int(np.argmax(band_6)) + r0_6
            peak_score_6 = float(band_6.max())
            competitor_dist_6 = abs(peak_bin_6 - gt_rho_bin) if peak_bin_6 != gt_rho_bin else 0

            # Window of ±3 bins (proposed nms_radius_rho=3)
            r0_3 = max(0, gt_rho_bin - 3)
            r1_3 = min(n_rho, gt_rho_bin + 4)
            band_3 = acc[gt_t_idx, r0_3:r1_3]
            peak_bin_3 = int(np.argmax(band_3)) + r0_3
            peak_score_3 = float(band_3.max())
            competitor_dist_3 = abs(peak_bin_3 - gt_rho_bin) if peak_bin_3 != gt_rho_bin else 0

            # Is the GT bin itself the strongest in window ±3?
            gt_wins_in_3 = abs(peak_bin_3 - gt_rho_bin) == 0

            print(f"  {gt['label']}:")
            print(f"    GT rho={gt['rho']:.1f} px (bin {gt_rho_bin})")
            print(f"    GT bin score: {gt_bin_score:.0f}")
            print(f"    ±6 window: peak bin {peak_bin_6} (ρ={peak_bin_6*RHO_STEP:.0f}) "
                  f"score={peak_score_6:.0f}  dist from GT={competitor_dist_6} bins")
            print(f"    ±3 window: peak bin {peak_bin_3} (ρ={peak_bin_3*RHO_STEP:.0f}) "
                  f"score={peak_score_3:.0f}  dist from GT={competitor_dist_3} bins")
            print(f"    GT wins in ±3 window: {'YES' if gt_wins_in_3 else 'NO'}")
            print(f"    → {'nms_radius_rho=3 would NOT help' if not gt_wins_in_3 else 'nms_radius_rho=3 would surface this edge'}")

            # Show vote distribution for diagnostic
            distro = acc[gt_t_idx, max(0, gt_rho_bin - 8):min(n_rho, gt_rho_bin + 9)]
            non_zero = np.where(distro > 0)[0]
            print(f"    Vote distribution (rho bins {max(0, gt_rho_bin - 8)}–{min(n_rho-1, gt_rho_bin + 8)}):")
            for idx in range(len(distro)):
                rb = max(0, gt_rho_bin - 8) + idx
                sc = distro[idx]
                if sc > 0:
                    marker = " ← GT" if rb == gt_rho_bin else ""
                    print(f"      bin {rb}: score={sc:.0f}{marker}")
            print()

        # Also scan for potential 2nd-best peak that could be a D-surfaced edge
        # when nms_radius_rho is reduced.
        # Simulate nms_radius_rho=3 by modifying the work copy.
        work_3 = acc.copy()
        acc_max = float(work_3.max())
        threshold = 0.25 * acc_max

        peaks_3_theta = []
        peaks_3_rho = []
        peaks_3_score = []

        for _ in range(20):
            idx = int(np.argmax(work_3.ravel()))
            value = float(work_3.ravel()[idx])
            if value < threshold:
                break
            t_idx, r_idx = map(int, np.unravel_index(idx, work_3.shape))
            peaks_3_theta.append(t_idx)
            peaks_3_rho.append(r_idx)
            peaks_3_score.append(value)
            r0 = max(0, r_idx - 3)
            r1 = min(n_rho, r_idx + 4)
            for dt in range(-3, 4):
                tt = (t_idx + dt) % n_theta
                work_3[tt, r0:r1] = 0.0

        # How many D failures get a match with nms_radius_rho=3 vs 6?
        n_d_r6 = 0
        n_d_r3 = 0
        for gt in gt_edges:
            if gt["segment"] is None:
                continue
            gt_rho_bin = int(round(gt["rho"] / RHO_STEP))
            gt_theta_rad = np.arctan2(gt["normal"][1], gt["normal"][0]) % np.pi
            gt_t_idx = int(np.round(gt_theta_rad / theta_step)) % n_theta

            # Check if any r3 peak matches
            matched_r3 = False
            for t, r, s in zip(peaks_3_theta, peaks_3_rho, peaks_3_score):
                ang_d = abs((t - gt_t_idx) * theta_step)
                ang_d = min(ang_d, np.pi - ang_d)
                rho_d = abs(r - gt_rho_bin) * RHO_STEP
                if np.rad2deg(ang_d) <= 5.0 and rho_d <= 5.0:
                    matched_r3 = True
                    break

            if _match_peak(gt, normals, rhos) < 0:
                n_d_r6 += 1
            if not matched_r3 and _match_peak(gt, normals, rhos) < 0:
                n_d_r3 += 1

        print(f"  Cluster {ci} D failures: nms_radius_rho=6 → {n_d_r6}, "
              f"nms_radius_rho=3 → {n_d_r3}")
        print(f"  D improvement: {n_d_r6 - n_d_r3} edge(s) gained")
        print()

    print("=" * 70)
    print("End I5 — NMS radius sensitivity measurement")


if __name__ == "__main__":
    main()
