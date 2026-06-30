"""Diagnostic script — Hough failure modes deep dive.

Uses the same synth-pipeline setup as the Phase II fixture tests, but
dumps per-cluster, per-GT-edge diagnostics to help root-cause failures.

Output (per cluster):
  - GT edges with (normal, rho, span)
  - All Hough peaks with angular/rho distances to each GT edge
  - For GT edges with matching peaks: _describe_support dump
  - Summary of which failure modes manifest
"""

from __future__ import annotations

# Re-use helpers from the test harness
import sys

import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.hough import hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from test_hough_harness import (
    _angular_distance_deg,
    _compute_finder_edges,
    _describe_support,
    _make_background,
    _match_peak,
    _run_pipeline_to_rois,
)

# ===========================================================================
# Config
# ===========================================================================

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

SEED = 42
BACKGROUND_SIZE = 640

# ===========================================================================
# Generate
# ===========================================================================


def main() -> None:
    print("=" * 70)
    print("Hough failure diagnostics — v12 default difficulty")
    print("=" * 70)
    print(f"  seed={SEED}, image size={BACKGROUND_SIZE}x{BACKGROUND_SIZE}")
    print(f"  jitter_fraction={CONFIG.jitter_fraction}")
    print(f"  noise_sigma={CONFIG.noise_sigma_range}")
    print(f"  blur_sigma={CONFIG.blur_sigma_range}")
    print()

    background = _make_background(BACKGROUND_SIZE, BACKGROUND_SIZE)
    rng = np.random.default_rng(SEED)
    image, metadata = generate_sample(rng, CONFIG, background)

    print(f"  version={metadata['version']}, N={metadata['N']}")
    print(
        f"  corners_qr: TL=({metadata['corners_qr']['TL'][0]:.0f},{metadata['corners_qr']['TL'][1]:.0f}) "
        f"TR=({metadata['corners_qr']['TR'][0]:.0f},{metadata['corners_qr']['TR'][1]:.0f}) "
        f"BR=({metadata['corners_qr']['BR'][0]:.0f},{metadata['corners_qr']['BR'][1]:.0f}) "
        f"BL=({metadata['corners_qr']['BL'][0]:.0f},{metadata['corners_qr']['BL'][1]:.0f})"
    )
    print()

    roi_results = _run_pipeline_to_rois(image)

    if len(roi_results) == 0:
        print("No clusters found — aborting.")
        return

    all_mode_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for roi, nms, angle, bbox, ci in roi_results:
        H_roi, W_roi = nms.shape
        print("=" * 70)
        print(f"Cluster {ci} — ROI shape=({H_roi}, {W_roi})")
        print(
            f"  bbox (image coords): rows=[{bbox[0]},{bbox[1]}) cols=[{bbox[2]},{bbox[3]})"
        )
        n_edge = np.count_nonzero(nms)
        print(
            f"  NMS edges: {n_edge} nonzero pixels (density={n_edge / (H_roi * W_roi) * 100:.1f}%)"
        )
        print(f"  NMS max={nms.max():.1f}, mean nonzero={nms[nms > 0].mean():.1f}")
        print()

        # --- Hough peaks ---
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        print(f"  Hough peaks: {len(normals)}")
        for i, (n, r, s) in enumerate(zip(normals, rhos, scores)):
            ang = np.rad2deg(np.arctan2(n[1], n[0])) % 180
            print(
                f"    P{i:2d}: θ={ang:6.1f}°  ρ={r:6.1f}  score={s:.0f}  n=({n[0]:.4f},{n[1]:.4f})"
            )
        print()

        # --- GT edges ---
        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        print("  GT edges:")
        for gt in gt_edges:
            if gt["segment"] is None:
                status = "OUTSIDE ROI"
            else:
                seg = gt["segment"]
                span = np.linalg.norm(seg[1] - seg[0])
                status = (
                    f"({seg[0][0]:.0f},{seg[0][1]:.0f})→({seg[1][0]:.0f},{seg[1][1]:.0f}) "
                    f"span={span:.1f}px"
                )
            ang = np.rad2deg(np.arctan2(gt["normal"][1], gt["normal"][0])) % 180
            print(f"    {gt['label']:12s}: θ={ang:6.1f}°  ρ={gt['rho']:6.1f}  {status}")
        print()

        # --- Match peaks to GT edges ---
        print("  Peak ↔ GT edge distance matrix (angular ° | rho px):")
        # Header
        header = "        " + "  ".join(f"{gt['label']:>12s}" for gt in gt_edges)
        print(header)
        for i, (n, r) in enumerate(zip(normals, rhos)):
            row = f"    P{i:2d}:"
            for gt in gt_edges:
                ang_d = _angular_distance_deg(n, gt["normal"])
                rho_d = abs(r - gt["rho"])
                row += f"  {ang_d:4.1f}°|{rho_d:5.1f}"
            print(row)
        print()

        # --- Per-GT-edge diagnostics ---
        print("  Per-edge diagnostics:")
        print()

        for gt in gt_edges:
            if gt["segment"] is None:
                continue

            match_idx = _match_peak(gt, normals, rhos)
            gt_seg = gt["segment"]
            gt_span = float(np.linalg.norm(gt_seg[1] - gt_seg[0]))

            if match_idx < 0:
                # No peak found — Failure D
                all_mode_counts["D"] += 1
                print(f"  ### {gt['label']}: FAILURE D — no peak matches")
                print(
                    f"      GT: θ={np.rad2deg(np.arctan2(gt['normal'][1], gt['normal'][0])) % 180:.1f}° ρ={gt['rho']:.1f} span={gt_span:.1f}px"
                )
                print(f"      Closest Hough peak distances:")
                if len(normals) > 0:
                    dists = [
                        (_angular_distance_deg(gt["normal"], n), abs(gt["rho"] - r), i)
                        for i, (n, r) in enumerate(zip(normals, rhos))
                    ]
                    dists.sort(key=lambda x: x[0] + x[1])
                    for ang_d, rho_d, pi in dists[:3]:
                        print(f"        P{pi}: {ang_d:.1f}° / {rho_d:.1f}px")
                print()
                continue

            # Peak found — check span / excessive / degeneracy
            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                float(scores[match_idx]),
                nms,
                angle,
                gap_tolerance=2.0,
                distance_thresh=1.5,
            )

            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
            gt_proj = gt_seg @ direction
            gt_span_dir = abs(float(gt_proj[1] - gt_proj[0]))

            degenerate = np.all(seg.endpoints == 0)
            if degenerate:
                ep_proj = np.array([0.0, 0.0])
                seg_span = 0.0
            else:
                ep_proj = seg.endpoints @ direction
                seg_span = abs(float(ep_proj[1] - ep_proj[0]))

            failures = []
            if degenerate:
                failures.append("D (degenerate)")
                all_mode_counts["D"] += 1
            else:
                if seg_span < 0.8 * gt_span_dir:
                    failures.append("A (span too short)")
                    all_mode_counts["A"] += 1

                # Check endpoint overflow (C)
                overflow = False
                for gt_ep in gt_seg:
                    dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
                    if dists.min() > 5.0:
                        overflow = True
                        break
                if overflow:
                    failures.append("C (span too long)")
                    all_mode_counts["C"] += 1

            status = "PASS" if not failures else f"FAIL {'+'.join(failures)}"
            print(f"  ### {gt['label']}: {status}")
            print(
                f"      GT: θ={np.rad2deg(np.arctan2(gt['normal'][1], gt['normal'][0])) % 180:.1f}° ρ={gt['rho']:.1f} span={gt_span_dir:.1f}px"
            )
            n_hough = normals[match_idx]
            print(
                f"      Peak P{match_idx}: θ={np.rad2deg(np.arctan2(n_hough[1], n_hough[0])) % 180:.1f}° ρ={rhos[match_idx]:.1f} score={scores[match_idx]:.0f}"
            )
            if not degenerate:
                print(
                    f"      Segment span={seg_span:.1f}px (GT={gt_span_dir:.1f}px, ratio={seg_span / gt_span_dir * 100:.0f}%)"
                )
                print(
                    f"      Endpoints: ({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f})"
                )
                print(
                    f"      GT endpoints: ({gt_seg[0][0]:.1f},{gt_seg[0][1]:.1f})→({gt_seg[1][0]:.1f},{gt_seg[1][1]:.1f})"
                )
            print()

            # Full support diagnostics
            print(_describe_support(seg, nms, angle, distance_thresh=1.5))
            print()

        # --- Phantom scan ---
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
                    continue  # parallel to a GT edge → internal QR structure

            seg = refine_line(
                normals[i],
                float(rhos[i]),
                float(scores[i]),
                nms,
                angle,
                gap_tolerance=2.0,
                distance_thresh=1.5,
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
                all_mode_counts["B"] += 1
                ang = np.rad2deg(np.arctan2(normals[i][1], normals[i][0])) % 180
                print(f"  ### Phantom P{i}: FAILURE B")
                print(f"      θ={ang:.1f}° ρ={rhos[i]:.1f} score={scores[i]:.0f}")
                print(f"      support={np.sum(mask)} pixels, mean NMS={mean_str:.1f}")
                print()

        print(
            f"  Cluster {ci}: {phantom_count} phantoms, "
            f"{sum(1 for gt in gt_edges if gt['segment'] is not None and _match_peak(gt, normals, rhos) < 0)} missing edges"
        )
        print()

    # --- Global summary ---
    print("=" * 70)
    print("FAILURE MODE SUMMARY")
    print("=" * 70)
    for mode in ("A", "B", "C", "D"):
        label = {
            "A": "Span too short",
            "B": "Phantom in blank",
            "C": "Span too long",
            "D": "Edge missing",
        }[mode]
        print(f"  Failure {mode} ({label}): {all_mode_counts[mode]}")

    total = sum(all_mode_counts.values())
    print(f"  Total failures: {total}")
    print()
    print("(end)")


if __name__ == "__main__":
    main()
