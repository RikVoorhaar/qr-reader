"""I1 — Verify D displacement: widen rho tolerance diagnostic.

Tests whether D-failing edges' displaced votes (11-15 px from GT) are
finder-boundary pixels or unrelated internal QR edges.

For each D edge, tries widened rho matching (5, 10, 15, 20 px) and
reports refine_line segment quality vs GT.
"""

from __future__ import annotations

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


def _match_peak_rho(gt_edge, normals, rhos, rho_tol=20.0, angle_tol_deg=5.0):
    """Like _match_peak but with configurable rho_tol."""
    best_i = -1
    best_dist = float("inf")
    for i in range(len(normals)):
        ang_dist = _angular_distance_deg(gt_edge["normal"], normals[i])
        rho_dist = abs(gt_edge["rho"] - rhos[i])
        if ang_dist <= angle_tol_deg and rho_dist <= rho_tol:
            score = ang_dist + rho_dist
            if score < best_dist:
                best_dist = score
                best_i = i
    return best_i


def _compute_accumulator(nms, angle, theta_step_deg=2.0, rho_step=1.0):
    H, W = nms.shape
    ys, xs = np.nonzero(nms)
    strengths = nms[ys, xs].astype(np.float64)
    thetas = np.fmod(angle[ys, xs], np.pi)
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)
    theta_step = np.deg2rad(theta_step_deg)
    n_theta = int(np.ceil(np.pi / theta_step))
    rho_max = np.hypot(W, H)
    n_rho = int(np.ceil(rho_max / rho_step)) + 1
    theta_idx = np.round(thetas / theta_step).astype(np.int32) % n_theta
    theta_q = theta_idx.astype(np.float64) * theta_step
    rho_vals = xs.astype(np.float64) * np.cos(theta_q) + ys.astype(np.float64) * np.sin(theta_q)
    rho_idx = np.round(rho_vals / rho_step).astype(np.int32)
    valid = (rho_idx >= 0) & (rho_idx < n_rho)
    flat_idx = theta_idx[valid] * n_rho + rho_idx[valid]
    acc_flat = np.bincount(flat_idx, weights=strengths[valid], minlength=n_theta * n_rho)
    return acc_flat.reshape(n_theta, n_rho).astype(np.float64), theta_step, n_theta, n_rho


def _theta_bin_of(normal, theta_step, n_theta):
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    return int(np.round(theta / theta_step)) % n_theta


def main():
    print("=" * 70)
    print("I1: D displacement — widened rho tolerance diagnostic")
    print("=" * 70)

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

    background = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, CONFIG, background)

    roi_results = _run_pipeline_to_rois(image)

    for roi, nms, angle, bbox, ci in roi_results:
        normals, rhos, scores = hough_vote_peaks(nms, angle)

        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        acc, theta_step, n_theta, n_rho = _compute_accumulator(nms, angle)

        for gt in gt_edges:
            if gt["segment"] is None:
                continue

            gt_seg = gt["segment"]
            gt_span = float(np.linalg.norm(gt_seg[1] - gt_seg[0]))

            # Standard match (5 px)
            match5 = _match_peak(gt, normals, rhos)

            if match5 >= 0:
                print(f"\n  {gt['label']}: matched at 5px (NOT a D failure)")
                continue

            # D failure — try widening
            print(f"\n  {'='*60}")
            print(f"  {gt['label']}: FAILURE D — trying widened rho matching")
            print(f"  {'='*60}")
            
            ang = np.rad2deg(np.arctan2(gt['normal'][1], gt['normal'][0])) % 180
            print(f"  GT: θ={ang:.1f}° ρ={gt['rho']:.1f} span={gt_span:.1f}px")
            
            print(f"\n  Closest Hough peaks (angle + rho distances):")
            dists = [
                (_angular_distance_deg(gt["normal"], n), abs(gt["rho"] - r), i)
                for i, (n, r) in enumerate(zip(normals, rhos))
            ]
            dists.sort(key=lambda x: x[0] + x[1])
            for ang_d, rho_d, pi in dists[:5]:
                n = normals[pi]
                ang_p = np.rad2deg(np.arctan2(n[1], n[0])) % 180
                print(f"    P{pi}: θ={ang_p:.1f}° ρ={rhos[pi]:.1f}  ang_dist={ang_d:.1f}° rho_dist={rho_d:.1f}px  score={scores[pi]:.0f}")

            # Try widening match with rho_tol = 10, 15, 20, 25, 30
            for rho_tol in [10, 15, 20, 25, 30]:
                match_idx = _match_peak_rho(gt, normals, rhos, rho_tol=rho_tol)
                if match_idx < 0:
                    print(f"\n  rho_tol={rho_tol}: NO MATCH")
                    continue

                matched_r = rhos[match_idx]
                matched_rho_dist = abs(gt["rho"] - matched_r)
                matched_n = normals[match_idx]
                matched_ang = np.rad2deg(np.arctan2(matched_n[1], matched_n[0])) % 180
                matched_ang_dist = _angular_distance_deg(gt["normal"], matched_n)

                seg = refine_line(
                    matched_n, float(matched_r), float(scores[match_idx]),
                    nms, angle, gap_tolerance=2.0, distance_thresh=1.5,
                )

                degenerate = np.all(seg.endpoints == 0)

                if degenerate:
                    print(f"\n  rho_tol={rho_tol}: MATCHED P{match_idx} (θ={matched_ang:.1f}° ρ={matched_r:.1f}, "
                          f"ang_d={matched_ang_dist:.1f}° rho_d={matched_rho_dist:.1f}px, score={scores[match_idx]:.0f})"
                          f" → DEGENERATE segment")
                    continue

                # Compute span along direction
                direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
                gt_proj = gt_seg @ direction
                gt_span_dir = abs(float(gt_proj[1] - gt_proj[0]))
                ep_proj = seg.endpoints @ direction
                seg_span = abs(float(ep_proj[1] - ep_proj[0]))
                span_ratio = seg_span / gt_span_dir * 100

                # Check endpoint proximity
                ep_dists = []
                for gt_ep in gt_seg:
                    d = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
                    ep_dists.append(float(d.min()))

                print(f"\n  rho_tol={rho_tol}: MATCHED P{match_idx} (θ={matched_ang:.1f}° ρ={matched_r:.1f}, "
                      f"ang_d={matched_ang_dist:.1f}° rho_d={matched_rho_dist:.1f}px, score={scores[match_idx]:.0f})"
                      f" → VALID segment")
                print(f"    Segment span={seg_span:.1f}px (GT={gt_span_dir:.1f}px, ratio={span_ratio:.0f}%)")
                print(f"    Endpoints: ({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f})")
                print(f"    Endpoint distance to GT: min(GT ep → seg) = {min(ep_dists):.1f}px, {max(ep_dists):.1f}px")
                print(f"    Support: {_describe_support(seg, nms, angle, distance_thresh=1.5)}")
                
                # Check if segment valid (span >= 80% GT, endpoints within 5px of either GT endpoint)
                span_ok = seg_span >= 0.8 * gt_span_dir
                ep_ok = all(d <= 5.0 for d in ep_dists)
                verdict = "VALID (finder boundary)" if (span_ok and ep_ok) else "LOW QUALITY"
                print(f"    Verdict: {verdict} (span_ok={span_ok}, ep_ok={ep_ok})")
                break  # Found a match — stop widening

            # Also dump the vote fragmentation for the D edge
            print(f"\n  --- Vote fragmentation ---")
            t_idx = _theta_bin_of(gt["normal"], theta_step, n_theta)
            gt_rho_bin = int(round(gt["rho"] / 1.0))
            rho_lo = max(0, gt_rho_bin - 15)
            rho_hi = min(n_rho, gt_rho_bin + 16)
            band = acc[t_idx, rho_lo:rho_hi]
            if band.sum() > 0:
                print(f"    Theta bin {t_idx} (θ≈{np.rad2deg(t_idx * theta_step):.0f}°):")
                print(f"    GT rho bin {gt_rho_bin} (ρ={gt['rho']:.1f}): score={float(acc[t_idx, gt_rho_bin]) if 0 <= gt_rho_bin < n_rho else 0:.0f}")
                top = np.argsort(-band)[:8]
                for k in top:
                    rb = rho_lo + int(k)
                    sc = float(band[k])
                    if sc <= 0:
                        continue
                    print(f"      rho bin {rb} (ρ={rb:.0f}): score={sc:.0f}")
            else:
                print(f"    Theta bin {t_idx}: no votes at all")

    print("\n" + "=" * 70)
    print("I1 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
