"""I9 — Endpoint trimming by strength percentile (Phase 13 feasibility check).

For each C-failure GT edge, attempts percentile-based endpoint trimming on the
support set to see whether the segment endpoints can be pulled back within
5 px of GT.  Tests percentile thresholds 5–50 in steps of 5.

Root cause hypothesis (I7): C failures with span>1.0 include support pixels
from the adjacent inner finder-pattern boundary (2–3 px away).  Those extra
pixels are at the extremes of the projection and should have lower NMS
strength (blurred / partial edges).  Trimming weak tails may recover correct
endpoints.
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.hough import LineSegment, hough_vote_peaks, refine_line
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from test_hough_harness import (
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


def _check_c_fail(seg: LineSegment, gt_seg: np.ndarray) -> bool:
    """True if segment endpoints are > 5 px from GT endpoints."""
    if np.all(seg.endpoints == 0):
        return True
    for gt_ep in gt_seg:
        dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
        if dists.min() > 5.0:
            return True
    return False


def _proj_span(pts: np.ndarray, direction: np.ndarray) -> tuple[float, float, float]:
    """Project points onto direction, return (min, max, span)."""
    proj = pts @ direction
    return float(proj.min()), float(proj.max()), float(proj.max() - proj.min())


def _trim_endpoints(
    pts: np.ndarray,
    strengths: np.ndarray,
    direction: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Trim support set by removing weakest-strength pixels at both ends.

    Repeatedly removes the weakest pixel from the left and right extremes
    of the projection until no remaining pixel is below the given strength
    percentile (computed on the *original* support set for stability).
    """
    if len(pts) < 4:
        return pts

    threshold = float(np.percentile(strengths, percentile))

    proj = pts @ direction
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]
    strengths_sorted = strengths[sort_idx]
    pts_sorted = pts[sort_idx]

    left = 0
    right = len(pts_sorted) - 1

    while left < right:
        # Check leftmost remaining
        if strengths_sorted[left] < threshold:
            left += 1
            continue
        # Check rightmost remaining
        if strengths_sorted[right] < threshold:
            right -= 1
            continue
        break

    if left >= right:
        return pts[:0]  # empty
    return pts_sorted[left:right + 1]


def _refine_with_trimming(
    normal: np.ndarray,
    rho: float,
    score: float,
    nms: np.ndarray,
    angle: np.ndarray,
    trim_percentile: float | None = None,
) -> LineSegment:
    """refine_line with optional endpoint trimming by strength."""
    if trim_percentile is None:
        return refine_line(normal, rho, score, nms, angle,
                           gap_tolerance=2.0, distance_thresh=1.5)

    H, W = nms.shape
    ys, xs = np.nonzero(np.asarray(nms))
    strengths = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    dists = np.abs(points @ normal - rho)
    mask = dists < 1.5

    support_pts = points[mask]
    support_strengths = strengths[mask]

    if len(support_pts) < 2:
        return LineSegment(
            normal=normal.copy(), rho=rho,
            endpoints=np.zeros((2, 2), dtype=np.float64),
            vote_score=score,
        )

    # Weighted TLS fit (same as refine_line)
    w = support_strengths / support_strengths.sum()
    c = (support_pts * w[:, None]).sum(axis=0)
    X = support_pts - c
    Xw = X * np.sqrt(w[:, None])
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)

    direction = vt[0]
    refined_normal = vt[1]
    refined_rho = float(refined_normal @ c)
    if refined_rho < 0:
        refined_normal = -refined_normal
        refined_rho = -refined_rho

    # Trim weak tails from support set
    trimmed = _trim_endpoints(support_pts, support_strengths, direction, trim_percentile)

    if len(trimmed) < 2:
        return LineSegment(
            normal=refined_normal, rho=refined_rho,
            endpoints=np.zeros((2, 2), dtype=np.float64),
            vote_score=score,
        )

    # Refit on trimmed set
    w2 = np.ones(len(trimmed), dtype=np.float64) / len(trimmed)
    c2 = (trimmed * w2[:, None]).sum(axis=0)
    X2 = trimmed - c2
    Xw2 = X2 * np.sqrt(w2[:, None])
    _, s2, vt2 = np.linalg.svd(Xw2, full_matrices=False)

    direction2 = vt2[0]
    refined_normal2 = vt2[1]
    refined_rho2 = float(refined_normal2 @ c2)
    if refined_rho2 < 0:
        refined_normal2 = -refined_normal2
        refined_rho2 = -refined_rho2

    # Longest contiguous run on trimmed set
    proj = trimmed @ direction2
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]

    run_a = float(proj_sorted[0])
    run_b = float(proj_sorted[0])
    best_len = 0.0
    best_a = run_a
    best_b = run_b

    for i in range(1, len(proj_sorted)):
        gap = float(proj_sorted[i] - proj_sorted[i - 1])
        if gap <= 2.0:
            run_b = float(proj_sorted[i])
        else:
            run_len = run_b - run_a
            if run_len > best_len:
                best_len = run_len
                best_a, best_b = run_a, run_b
            run_a = float(proj_sorted[i])
            run_b = float(proj_sorted[i])

    run_len = run_b - run_a
    if run_len > best_len:
        best_len = run_len
        best_a, best_b = run_a, run_b

    ep1 = refined_rho2 * refined_normal2 + best_a * direction2
    ep2 = refined_rho2 * refined_normal2 + best_b * direction2

    return LineSegment(
        normal=refined_normal2, rho=refined_rho2,
        endpoints=np.array([ep1, ep2], dtype=np.float64),
        vote_score=score,
    )

CONFIG_V5 = AugmentationConfig(
    version=5,
    content="QR Reader v1",
    error_correction="M",
    ppm_range=(5.0, 12.0),
    target_ppm_range=(4.0, 10.0),
    jitter_fraction=0.15,
    feather_sigma_range=(0.5, 2.0),
    blur_sigma_range=(0.2, 1.0),
    noise_sigma_range=(1.0, 5.0),
    jpeg_quality_range=(65, 95),
    global_seed=123,
)



def analyze_one_config(
    config: AugmentationConfig,
    label: str,
    seed: int = 42,
    debug: bool = False,
) -> dict:
    """Run analysis for one config and return C-fix counts per percentile."""
    bg = _make_background(640, 640)
    rng = np.random.default_rng(seed)
    image, metadata = generate_sample(rng, config, bg)

    roi_results = _run_pipeline_to_rois(image)

    percentiles = list(range(5, 55, 5))
    fixes: dict[float, int] = {p: 0 for p in percentiles}
    total_c = 0
    total_edges = 0

    for roi, nms, angle, bbox, ci in roi_results:
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
            if match_idx < 0:
                continue

            total_edges += 1

            # Baseline refine (no trimming)
            seg = refine_line(
                normals[match_idx], float(rhos[match_idx]), float(scores[match_idx]),
                nms, angle, gap_tolerance=2.0, distance_thresh=1.5,
            )

            gt_seg = gt["segment"]
            is_c = _check_c_fail(seg, gt_seg)
            if is_c:
                if debug:
                    print(f"    C FAIL C{ci} {gt['label']}: "
                          f"endpoints=({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→"
                          f"({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f}) "
                          f"gt=({gt_seg[0][0]:.1f},{gt_seg[0][1]:.1f})→"
                          f"({gt_seg[1][0]:.1f},{gt_seg[1][1]:.1f})")
                total_c += 1

                # Try each percentile
                for p in percentiles:
                    trimmed = _refine_with_trimming(
                        normals[match_idx], float(rhos[match_idx]),
                        float(scores[match_idx]), nms, angle, trim_percentile=p,
                    )
                    if not _check_c_fail(trimmed, gt_seg):
                        fixes[p] += 1

    return {
        "label": label,
        "total_edges": total_edges,
        "total_c": total_c,
        "fixes": fixes,
    }


def main() -> None:
    print("=" * 70)
    print("I9 — Endpoint trimming by strength percentile")
    print("=" * 70)
    print()

    for config, label, seed in [
        (CONFIG, "v12-default", 42),
        (CONFIG_CLEAN, "v12-clean", 42),
        (CONFIG_V5, "v5-default", 123),
    ]:
        result = analyze_one_config(config, label, seed)
        print(f"--- {label} ---")
        print(f"  GT edges with Hough peak: {result['total_edges']}")
        print(f"  C failures (baseline):    {result['total_c']}")
        print()
        print(f"  {'percentile':>12s}  {'C fixes':>8s}  {'residual C':>10s}")
        for p in sorted(result["fixes"].keys()):
            fixed = result["fixes"][p]
            residual = result["total_c"] - fixed
            print(f"  {p:>10.0f}th  {fixed:>8d}  {residual:>10d}")

        # Find best percentile
        best_p = max(result["fixes"], key=lambda p: result["fixes"][p])
        best_fixed = result["fixes"][best_p]
        print(f"\n  Best: {best_p:.0f}th percentile — {best_fixed}/{result['total_c']} C failures fixed")
        print()

    print("=" * 70)
    print("I9 complete.")
    print()


if __name__ == "__main__":
    main()
