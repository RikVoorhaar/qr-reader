"""Diagnostic script — Hough failure modes deep dive.

Uses the same synth-pipeline setup as the Phase II fixture tests, but
dumps per-cluster, per-GT-edge diagnostics to help root-cause failures.

Output (per cluster):
  - GT edges with (normal, rho, span)
  - All Hough peaks with angular/rho distances to each GT edge
  - For GT edges with matching peaks: _describe_support dump
  - Vote fragmentation analysis for failing (D) edges — rho-histogram of
    the accumulator in the closest theta bin, showing vote dilution
  - Peak-suppression simulation — raw peaks above absolute threshold vs
    relative threshold, revealing threshold-filtered true edges
  - Summary of which failure modes manifest

Global output:
  - Cross-cluster failure-mode summary with fragmentation metrics
  - Per-mode root-cause classification
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
# Accumulator reconstruction (mirrors hough_vote_peaks internals)
# ===========================================================================


def _compute_accumulator(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 2.0,
    rho_step: float = 1.0,
) -> tuple[np.ndarray, float, int, int]:
    """Reconstruct the Hough accumulator (mirrors ``hough_vote_peaks``).

    Returns ``(acc, theta_step, n_theta, n_rho)`` where ``acc`` has shape
    ``(n_theta, n_rho)``.  This is a diagnostic copy — the production
    ``hough_vote_peaks`` does not expose it directly.
    """
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
    rho_vals = xs.astype(np.float64) * np.cos(theta_q) + ys.astype(np.float64) * np.sin(
        theta_q
    )
    rho_idx = np.round(rho_vals / rho_step).astype(np.int32)
    valid = (rho_idx >= 0) & (rho_idx < n_rho)

    flat_idx = theta_idx[valid] * n_rho + rho_idx[valid]
    acc_flat = np.bincount(
        flat_idx, weights=strengths[valid], minlength=n_theta * n_rho
    )
    acc = acc_flat.reshape(n_theta, n_rho).astype(np.float64)
    return acc, theta_step, n_theta, n_rho


def _theta_bin_of(normal: np.ndarray, theta_step: float, n_theta: int) -> int:
    """Return the accumulator theta-bin index closest to *normal*'s angle."""
    theta = np.arctan2(normal[1], normal[0]) % np.pi
    return int(np.round(theta / theta_step)) % n_theta


def _analyze_vote_fragmentation(
    gt_edge: dict,
    nms: np.ndarray,
    angle: np.ndarray,
    acc: np.ndarray,
    theta_step: float,
    n_theta: int,
    n_rho: int,
    rho_step: float = 1.0,
    window: int = 15,
) -> list[str]:
    """For a GT edge that has no matching Hough peak (Failure D), show the
    accumulator vote distribution across rho bins in the closest theta bin.

    Reveals whether the true edge's votes are diluted across multiple rho
    bins (fragmentation) and whether a stronger parallel edge dominates.
    """
    lines: list[str] = []
    t_idx = _theta_bin_of(gt_edge["normal"], theta_step, n_theta)
    gt_rho = gt_edge["rho"]
    gt_rho_bin = int(round(gt_rho / rho_step))

    lines.append(
        f"      Vote fragmentation (theta bin {t_idx}, "
        f"θ≈{np.rad2deg(t_idx * theta_step):.0f}°):"
    )

    rho_lo = max(0, gt_rho_bin - window)
    rho_hi = min(n_rho, gt_rho_bin + window + 1)
    band = acc[t_idx, rho_lo:rho_hi]

    total_in_band = float(band.sum())
    if total_in_band <= 0:
        lines.append(f"        no votes in rho [{rho_lo},{rho_hi}) — edge fully absent from NMS")
        return lines

    peak_bin = int(np.argmax(band)) + rho_lo
    peak_score = float(acc[t_idx, peak_bin])
    gt_bin_score = float(acc[t_idx, gt_rho_bin]) if 0 <= gt_rho_bin < n_rho else 0.0

    # Find the strongest bin in the band and its rho
    lines.append(
        f"        band rho=[{rho_lo},{rho_hi}): total votes={total_in_band:.0f}"
    )
    lines.append(
        f"        GT rho bin {gt_rho_bin} (ρ={gt_rho:.1f}): score={gt_bin_score:.0f}"
    )
    lines.append(
        f"        strongest bin {peak_bin} (ρ={peak_bin * rho_step:.0f}): "
        f"score={peak_score:.0f} ({peak_score / max(total_in_band, 1) * 100:.0f}% of band)"
    )

    if peak_bin != gt_rho_bin:
        lines.append(
            f"        → strongest peak is {abs(peak_bin - gt_rho_bin)} rho bins "
            f"({abs(peak_bin - gt_rho_bin) * rho_step:.0f}px) from GT — parallel edge dominates"
        )

    if gt_bin_score > 0:
        # How many bins share the GT edge's votes?
        nearby = acc[t_idx, max(0, gt_rho_bin - 3) : min(n_rho, gt_rho_bin + 4)]
        spread_bins = int(np.count_nonzero(nearby))
        lines.append(
            f"        GT bin ±3 rho: {spread_bins} non-zero bins, "
            f"max={float(nearby.max()):.0f}, sum={float(nearby.sum()):.0f}"
        )
        if spread_bins >= 4:
            lines.append(
                f"        → votes fragmented across {spread_bins} bins "
                f"(dilution) — relative threshold filters out the true edge"
            )
    else:
        lines.append("        → GT rho bin has ZERO votes — edge pixels quantise elsewhere")

    # Show top-5 rho bins in the band
    top = np.argsort(-band)[:5]
    lines.append("        top-5 rho bins in band:")
    for k in top:
        rb = rho_lo + int(k)
        sc = float(band[k])
        if sc <= 0:
            continue
        marker = " ← GT" if rb == gt_rho_bin else ""
        lines.append(
            f"          rho bin {rb} (ρ={rb * rho_step:.0f}): score={sc:.0f}{marker}"
        )

    return lines


def _simulate_peak_suppression(
    nms: np.ndarray,
    angle: np.ndarray,
    acc: np.ndarray,
    theta_step: float,
    n_theta: int,
    n_rho: int,
    threshold_rel: float = 0.25,
    nms_radius_theta: int = 3,
    nms_radius_rho: int = 6,
    max_peaks: int = 20,
) -> list[str]:
    """Show raw peaks above absolute vs relative threshold.

    Lists all accumulator local maxima above a low absolute floor and
    marks which ones survive the production relative-threshold + NMS
    pipeline.  Reveals true-edge peaks that are filtered out.
    """
    lines: list[str] = []
    acc_max = float(acc.max())
    if acc_max <= 0:
        lines.append("      (empty accumulator)")
        return lines

    rel_threshold = threshold_rel * acc_max
    # Low absolute floor: 5% of max — surfaces near-threshold peaks.
    abs_floor = 0.05 * acc_max

    # Find all local maxima above abs_floor (simple non-plateau argmax scan).
    candidates: list[tuple[float, int, int]] = []
    for t in range(n_theta):
        for r in range(n_rho):
            v = float(acc[t, r])
            if v < abs_floor:
                continue
            # Local max check (8-neighbourhood, circular in theta)
            is_max = True
            for dt in (-1, 0, 1):
                tt = (t + dt) % n_theta
                for dr in (-1, 0, 1):
                    if dt == 0 and dr == 0:
                        continue
                    rr = r + dr
                    if 0 <= rr < n_rho and float(acc[tt, rr]) > v:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                candidates.append((v, t, r))

    candidates.sort(key=lambda c: -c[0])
    lines.append(
        f"      Peak suppression simulation (rel_threshold={rel_threshold:.0f}, "
        f"abs_floor={abs_floor:.0f}, acc_max={acc_max:.0f}):"
    )
    lines.append(f"        {len(candidates)} local maxima above abs_floor")

    # Simulate which survive relative threshold + NMS
    work = acc.copy()
    survivors: list[tuple[float, int, int]] = []
    for _ in range(max_peaks):
        idx = int(np.argmax(work.ravel()))
        value = float(work.ravel()[idx])
        if value < rel_threshold:
            break
        t_idx, r_idx = map(int, np.unravel_index(idx, work.shape))
        survivors.append((value, t_idx, r_idx))
        r0 = max(0, r_idx - nms_radius_rho)
        r1 = min(n_rho, r_idx + nms_radius_rho + 1)
        for dt in range(-nms_radius_theta, nms_radius_theta + 1):
            tt = (t_idx + dt) % n_theta
            work[tt, r0:r1] = 0.0

    survivor_set = {(t, r) for _, t, r in survivors}
    filtered_out = [c for c in candidates if (c[1], c[2]) not in survivor_set]

    lines.append(f"        {len(survivors)} survive rel_threshold + NMS")
    lines.append(f"        {len(filtered_out)} filtered out (below rel_threshold or NMS-suppressed)")

    for v, t, r in candidates[:10]:
        status = "survives" if (t, r) in survivor_set else "FILTERED"
        theta_deg = np.rad2deg(t * theta_step)
        lines.append(
            f"          θ={theta_deg:.0f}° ρ={r:.0f} score={v:.0f} — {status}"
        )

    return lines

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
    # Per-cluster detail for the global summary.
    cluster_summaries: list[dict] = []

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

        # --- Accumulator reconstruction (for fragmentation analysis) ---
        acc, theta_step, n_theta, n_rho = _compute_accumulator(nms, angle)
        cluster_acc_max = float(acc.max())

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
                # Vote fragmentation analysis — reveals dilution / parallel dominance
                for line in _analyze_vote_fragmentation(
                    gt, nms, angle, acc, theta_step, n_theta, n_rho
                ):
                    print(line)
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

        # --- Peak-suppression simulation ---
        print("  --- Peak suppression simulation ---")
        for line in _simulate_peak_suppression(
            nms, angle, acc, theta_step, n_theta, n_rho
        ):
            print(line)
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

        # Collect per-cluster summary for the global report.
        n_gt_in_roi = sum(1 for gt in gt_edges if gt["segment"] is not None)
        n_matched = sum(
            1
            for gt in gt_edges
            if gt["segment"] is not None and _match_peak(gt, normals, rhos) >= 0
        )
        cluster_summaries.append(
            {
                "ci": ci,
                "n_peaks": len(normals),
                "n_gt_in_roi": n_gt_in_roi,
                "n_matched": n_matched,
                "acc_max": cluster_acc_max,
                "phantoms": phantom_count,
                "n_edge_pixels": int(np.count_nonzero(nms)),
            }
        )

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

    # --- Root-cause classification ---
    print("=" * 70)
    print("ROOT-CAUSE CLASSIFICATION")
    print("=" * 70)
    causes = [
        (
            "A",
            "gap_tolerance=2.0 can't bridge 3-7 px NMS gaps → longest "
            "contiguous run covers only a fragment of the finder boundary.",
        ),
        (
            "B",
            "Sparse/coincidental pixels concentrate votes in a single "
            "(theta, rho) bin; refine_line produces a non-degenerate segment "
            "from them because there is no minimum-support or contiguity gate.",
        ),
        (
            "C",
            "Weighted-TLS direction drifts ~1° from the Hough peak normal; "
            "with distance_thresh=1.5 px the support set captures a parallel "
            "edge 3-5 px away, extending the span past the GT endpoints.",
        ),
        (
            "D",
            "Fragmented true edge dilutes votes across multiple rho bins; "
            "its peak score falls below threshold_rel*max (dominated by a "
            "stronger parallel internal edge). NOT theta quantization "
            "(mid-bin theta only shifts rho ~2.5 px — see isolation tests).",
        ),
    ]
    for mode, cause in causes:
        print(f"  Failure {mode}: {cause}")
    print()

    # --- Per-cluster summary table ---
    print("=" * 70)
    print("PER-CLUSTER SUMMARY")
    print("=" * 70)
    header = (
        f"  {'C':>2}  {'peaks':>5}  {'GT_in':>6}  {'match':>5}  "
        f"{'phant':>5}  {'acc_max':>7}  {'edge_px':>7}"
    )
    print(header)
    for cs in cluster_summaries:
        print(
            f"  C{cs['ci']:<1}  {cs['n_peaks']:>5}  {cs['n_gt_in_roi']:>6}  "
            f"{cs['n_matched']:>5}  {cs['phantoms']:>5}  "
            f"{cs['acc_max']:>7.0f}  {cs['n_edge_pixels']:>7}"
        )
    print()

    # --- Aggregate metrics ---
    total_gt = sum(cs["n_gt_in_roi"] for cs in cluster_summaries)
    total_matched = sum(cs["n_matched"] for cs in cluster_summaries)
    total_phantoms = sum(cs["phantoms"] for cs in cluster_summaries)
    total_peaks = sum(cs["n_peaks"] for cs in cluster_summaries)
    print(f"  Total GT edges in ROIs: {total_gt}")
    print(f"  Total matched (within 5°+5px): {total_matched}")
    print(f"  Total unmatched (Failure D): {total_gt - total_matched}")
    print(f"  Total phantoms (Failure B): {total_phantoms}")
    print(f"  Total Hough peaks extracted: {total_peaks}")
    if total_gt > 0:
        print(f"  Match rate: {total_matched / total_gt * 100:.0f}%")
    print()
    print("(end)")


if __name__ == "__main__":
    main()
