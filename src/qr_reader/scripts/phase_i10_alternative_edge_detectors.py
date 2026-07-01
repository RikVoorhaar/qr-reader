"""I10 — Compare Canny + our Hough and OpenCV HoughLinesP against baseline.

Two alternative edge detection / line detection pipelines, scored against the
same ground-truth finder-pattern edges used in the Phase II fixture tests.

Approach A: Canny + our Hough pipeline
  - Replace extract_thin_edges with Canny edges (masked Sobel gradients)
  - Feed into hough_vote_peaks + refine_line (unchanged)
  - Isolates effect of Canny's hysteresis NMS vs our interpolated NMS

Approach B: OpenCV Canny + HoughLinesP
  - cv2.Canny → cv2.HoughLinesP
  - Map to LineSegment-like format
  - Completely different voting/refinement algorithm
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.hough import LineSegment, hough_vote_peaks, refine_line
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
    _normal_angle_deg,
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

# ===========================================================================
# Approach A: Canny-based edge extraction
# ===========================================================================


def extract_canny_edges(
    roi: np.ndarray,
    blur_sigma: float = 1.0,
    canny_low: float = 50.0,
    canny_high: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Canny edge detection → masked Sobel gradients.

    Returns (nms, angle) with the same semantics as extract_thin_edges:
      nms:   gradient magnitude at Canny edge pixels (0 elsewhere)
      angle: gradient-normal angle at Canny edge pixels (0 elsewhere)

    Canny provides binary edges; we overlay Sobel gradient magnitude and
    direction on those pixels so the existing gradient-guided Hough pipeline
    can consume them.
    """
    roi_f = roi.astype(np.float64, copy=False)

    blurred = ndimage.gaussian_filter(roi_f, sigma=blur_sigma, mode="reflect")

    # Sobel gradients (same as extract_thin_edges)
    gx = ndimage.sobel(blurred, axis=1, mode="constant")
    gy = ndimage.sobel(blurred, axis=0, mode="constant")
    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx, out=np.zeros_like(mag), where=mag > 0)

    # Canny edge detection (on the same blurred image)
    roi_uint8 = np.clip(roi_f, 0, 255).astype(np.uint8)
    canny_binary = cv2.Canny(roi_uint8, canny_low, canny_high)

    # Mask magnitude and angle with Canny output
    nms = np.where(canny_binary > 0, mag, 0.0)
    angle = np.where(canny_binary > 0, angle, 0.0)

    return nms, angle


def _find_rois_canny(
    image: np.ndarray,
    *,
    blur_sigma: float = 1.0,
    canny_low: float = 50.0,
    canny_high: float = 150.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], int]]:
    """Like _run_pipeline_to_rois but uses Canny edges."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image)

    img_binary = binarize_image(gray)

    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return []

    clusters = cluster_candidates(rows_valid, cols_valid_all)

    results = []
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)

        if roi.size == 0:
            continue

        nms, angle = extract_canny_edges(roi, blur_sigma=blur_sigma, canny_low=canny_low, canny_high=canny_high)
        results.append((roi, nms, angle, bbox, ci))

    return results


# ===========================================================================
# Approach B: OpenCV HoughLinesP
# ===========================================================================


@dataclass
class CVLine:
    """Line from OpenCV HoughLinesP, converted to our convention.

    Attributes match LineSegment's key geometry:
      normal: unit normal (x,y pixel coords), rho >= 0
      rho: signed distance
      endpoints: (2,2) pixel coords
      vote_score: confidence-like (from line length)
    """
    normal: np.ndarray
    rho: float
    endpoints: np.ndarray
    vote_score: float


def _line_from_houghp(
    x1: int, y1: int, x2: int, y2: int,
    score: float = 1.0,
) -> CVLine:
    """Convert HoughLinesP segment endpoints to our normal+rho convention."""
    a = np.array([float(x1), float(y1)], dtype=np.float64)
    b = np.array([float(x2), float(y2)], dtype=np.float64)
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        return CVLine(
            normal=np.array([1.0, 0.0]),
            rho=0.0,
            endpoints=np.zeros((2, 2), dtype=np.float64),
            vote_score=score,
        )
    direction = d / length
    normal = np.array([direction[1], -direction[0]], dtype=np.float64)
    rho = float(normal @ a)
    if rho < 0:
        normal = -normal
        rho = -rho
    return CVLine(normal=normal, rho=rho, endpoints=np.array([a, b]), vote_score=score)


def detect_lines_houghp(
    roi: np.ndarray,
    canny_low: float = 50.0,
    canny_high: float = 150.0,
    hough_threshold: int = 50,
    min_line_length: float = 20.0,
    max_line_gap: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV Canny → HoughLinesP pipeline.

    Returns (normals, rhos) arrays compatible with _match_peak and
    the other harness assertion helpers.  Also returns scores array
    (line length in pixels).
    """
    gray_uint8 = np.clip(roi, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.deg2rad(1.0),
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,))

    cv_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv_lines.append(_line_from_houghp(x1, y1, x2, y2))

    # Deduplicate: group by normal angle + rho
    normals_list = []
    rhos_list = []
    scores_list = []
    for cvl in cv_lines:
        if np.all(cvl.endpoints == 0):
            continue
        # Check if similar to an existing line
        duplicate = False
        for ni, nj in zip(normals_list, rhos_list):
            ang_dist = _angular_distance_deg(cvl.normal, ni)
            rho_dist = abs(cvl.rho - nj)
            if ang_dist < 5.0 and rho_dist < 5.0:
                duplicate = True
                break
        if not duplicate:
            normals_list.append(cvl.normal)
            rhos_list.append(cvl.rho)
            scores_list.append(cvl.vote_score)

    return (
        np.array(normals_list, dtype=np.float64),
        np.array(rhos_list, dtype=np.float64),
        np.array(scores_list, dtype=np.float64),
    )


def _find_rois_houghp(
    image: np.ndarray,
    canny_low: float = 50.0,
    canny_high: float = 150.0,
    hough_threshold: int = 50,
    min_line_length: float = 20.0,
    max_line_gap: float = 5.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], int]]:
    """Like _run_pipeline_to_rois but detects lines via HoughLinesP.

    Returns (roi_gray, normals, rhos, bbox, cluster_idx) for each cluster.
    The normals/rhos come from HoughLinesP instead of hough_vote_peaks.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image)

    img_binary = binarize_image(gray)

    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return []

    clusters = cluster_candidates(rows_valid, cols_valid_all)

    results = []
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0:
            continue

        normals, rhos, _ = detect_lines_houghp(
            roi,
            canny_low=canny_low,
            canny_high=canny_high,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
        )
        results.append((roi, normals, rhos, bbox, ci))

    return results


# ===========================================================================
# Evaluation helpers
# ===========================================================================


def _count_failures(
    gt_edges: list[dict],
    normals: np.ndarray,
    rhos: np.ndarray,
    nms: np.ndarray | None = None,
    angle: np.ndarray | None = None,
    *,
    label: str = "",
) -> dict:
    """Count A/B/C/D failures for a set of detected lines.

    Uses the same logic as TestFixtureReal but returns counts instead of
    accumulating failure message strings.

    When nms/angle are None (HoughLinesP case), we can only check D
    (missing edge peaks) and maybe A via endpoint proximity.
    """
    d_count = 0
    a_count = 0
    c_count = 0
    b_count = 0

    for gt in gt_edges:
        if gt["segment"] is None:
            continue
        match_idx = _match_peak(gt, normals, rhos)
        if match_idx < 0:
            d_count += 1
            continue

        if nms is not None and angle is not None:
            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                1.0, nms, angle,
                gap_tolerance=2.0,
                distance_thresh=1.5,
            )

            if np.all(seg.endpoints == 0):
                d_count += 1
                continue

            gt_seg = gt["segment"]
            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
            gt_proj = gt_seg @ direction
            gt_span = abs(gt_proj[1] - gt_proj[0])
            ep_proj = seg.endpoints @ direction
            seg_span = abs(ep_proj[1] - ep_proj[0])

            if seg_span < 0.8 * gt_span:
                a_count += 1

            for gt_ep in gt_seg:
                dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
                if dists.min() > 5.0:
                    c_count += 1
                    break
        else:
            # HoughLinesP: no NMS/angle for refine_line.
            # Approximate span check from segment endpoints.
            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)

            # Find the HoughLinesP segment that matched via _match_peak
            # We don't have the segment directly, so estimate from normal/rho.
            seg = _make_segment_from_normal_rho(normals[match_idx], rhos[match_idx])
            if seg is None:
                d_count += 1
                continue

            gt_seg = gt["segment"]
            gt_proj = gt_seg @ direction
            gt_span = abs(gt_proj[1] - gt_proj[0])
            ep_proj = seg @ direction
            seg_span = abs(ep_proj[1] - ep_proj[0])

            if seg_span < 0.8 * gt_span:
                a_count += 1

            for gt_ep in gt_seg:
                dists = np.linalg.norm(seg - gt_ep, axis=1)
                if dists.min() > 5.0:
                    c_count += 1
                    break

    # Phantoms (B) — only check if nms available
    if nms is not None and angle is not None and len(normals) > 0:
        gt_normals = np.array([e["normal"] for e in gt_edges if e["segment"] is not None])
        for i in range(len(normals)):
            matched = any(
                gt["segment"] is not None and _match_peak(gt, normals[[i]], rhos[[i]]) >= 0
                for gt in gt_edges
            )
            if matched:
                continue
            if len(gt_normals) > 0:
                min_ang = min(_angular_distance_deg(normals[i], gn) for gn in gt_normals)
                if min_ang < 12.0:
                    continue

            seg = refine_line(normals[i], float(rhos[i]), 1.0, nms, angle)
            if np.all(seg.endpoints == 0):
                continue

            ys, xs = np.nonzero(np.asarray(nms))
            strengths = nms[ys, xs]
            points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
            dists = np.abs(points @ seg.normal - seg.rho)
            mask = dists < 1.5
            if np.sum(mask) == 0:
                continue
            mean_strength = float(strengths[mask].mean())
            if mean_strength > 400.0:
                b_count += 1

    gt_count = sum(1 for e in gt_edges if e["segment"] is not None)
    return {
        "gt_count": gt_count,
        "D": d_count,
        "A": a_count,
        "C": c_count,
        "B": b_count,
        "total": d_count + a_count + c_count + b_count,
    }


def _make_segment_from_normal_rho(
    normal: np.ndarray, rho: float, extent: float = 100.0
) -> np.ndarray | None:
    """Create approximate segment endpoints from normal/rho (for HoughLinesP)."""
    direction = np.array([-normal[1], normal[0]], dtype=np.float64)
    center = rho * normal
    return np.array([center - extent * direction, center + extent * direction])


def _evaluate_baseline(
    image: np.ndarray, metadata: dict, label: str
) -> dict:
    """Evaluate the current baseline (extract_thin_edges → our Hough)."""
    all_failures = {"D": 0, "A": 0, "C": 0, "B": 0, "gt_count": 0, "total": 0}
    cluster_count = 0

    roi_results = _run_pipeline_to_rois(image)
    if not roi_results:
        return all_failures

    for roi, nms, angle, bbox, ci in roi_results:
        offset = (bbox[0], bbox[2])  # (row0, col0)
        shape = roi.shape[:2]
        gt_edges = _compute_finder_edges(metadata, offset, shape)

        normals, rhos, _ = hough_vote_peaks(nms, angle)
        result = _count_failures(gt_edges, normals, rhos, nms, angle, label=f"{label}_baseline_C{ci}")

        for k in all_failures:
            all_failures[k] += result.get(k, 0)
        cluster_count += 1

    all_failures["cluster_count"] = cluster_count
    return all_failures


def _evaluate_canny(
    image: np.ndarray, metadata: dict, label: str,
    canny_low: float, canny_high: float,
) -> dict:
    """Evaluate Approach A: Canny + our Hough."""
    all_failures = {"D": 0, "A": 0, "C": 0, "B": 0, "gt_count": 0, "total": 0}
    cluster_count = 0

    roi_results = _find_rois_canny(image, canny_low=canny_low, canny_high=canny_high)
    if not roi_results:
        return all_failures

    for roi, nms, angle, bbox, ci in roi_results:
        offset = (bbox[0], bbox[2])  # (row0, col0)
        shape = roi.shape[:2]
        gt_edges = _compute_finder_edges(metadata, offset, shape)

        normals, rhos, _ = hough_vote_peaks(nms, angle)
        result = _count_failures(gt_edges, normals, rhos, nms, angle, label=f"{label}_canny_C{ci}")

        for k in all_failures:
            all_failures[k] += result.get(k, 0)
        cluster_count += 1

    all_failures["cluster_count"] = cluster_count
    return all_failures


def _evaluate_houghp(
    image: np.ndarray, metadata: dict, label: str,
    canny_low: float, canny_high: float,
    hough_threshold: int, min_line_length: float, max_line_gap: float,
) -> dict:
    """Evaluate Approach B: OpenCV Canny → HoughLinesP."""
    all_failures = {"D": 0, "A": 0, "C": 0, "B": 0, "gt_count": 0, "total": 0}
    cluster_count = 0

    roi_results = _find_rois_houghp(
        image,
        canny_low=canny_low, canny_high=canny_high,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length, max_line_gap=max_line_gap,
    )
    if not roi_results:
        return all_failures

    for roi, normals, rhos, bbox, ci in roi_results:
        offset = (bbox[0], bbox[2])  # (row0, col0)
        shape = roi.shape[:2]
        gt_edges = _compute_finder_edges(metadata, offset, shape)

        result = _count_failures(gt_edges, normals, rhos, label=f"{label}_houghp_C{ci}")

        for k in all_failures:
            all_failures[k] += result.get(k, 0)
        cluster_count += 1

    all_failures["cluster_count"] = cluster_count
    return all_failures


# ===========================================================================
# Best-effort Canny threshold sweep
# ===========================================================================

CANNY_SETTINGS = [
    (30, 90),
    (50, 150),
    (70, 210),
    (100, 200),
    (100, 300),
    (150, 300),
    (50, 100),
    (30, 60),
    (20, 60),
]

HOUGHP_SETTINGS = [
    (30, 15.0, 3.0),
    (50, 20.0, 5.0),
    (70, 20.0, 5.0),
    (100, 20.0, 5.0),
    (50, 30.0, 5.0),
    (50, 15.0, 5.0),
    (30, 10.0, 3.0),
]


def _print_result(header: str, result: dict, baseline: dict | None = None) -> None:
    gt = result.get("gt_count", 0)
    d = result.get("D", 0)
    a = result.get("A", 0)
    c = result.get("C", 0)
    b = result.get("B", 0)
    total = result.get("total", 0)
    clusters = result.get("cluster_count", 0)

    rel = ""
    if baseline is not None and baseline.get("total", 0) > 0:
        chg = total - baseline["total"]
        rel = f"  ({chg:+d} vs baseline)"

    print(f"  {header}: D={d} A={a} C={c} B={b}  total={total}{rel}  (clusters={clusters}, gt={gt})")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("I10 — Alternative edge detectors vs baseline")
    print("=" * 70)

    bg = _make_background(640, 640)

    configs = [
        (CONFIG, "v12-default", 42),
        (CONFIG_CLEAN, "v12-clean", 42),
        (CONFIG_V5, "v5-default", 123),
    ]

    for config, label, seed in configs:
        print(f"\n--- {label} ---")
        rng = np.random.default_rng(seed)
        image, metadata = generate_sample(rng, config, bg)

        # --- Baseline ---
        baseline = _evaluate_baseline(image, metadata, label)
        _print_result("Baseline (our NMS + our Hough)", baseline)

        # --- Approach A: Canny threshold sweep ---
        best_a = None
        best_a_total = 999
        for low, high in CANNY_SETTINGS:
            result = _evaluate_canny(image, metadata, label, canny_low=low, canny_high=high)
            _print_result(f"Canny L={low} H={high}", result, baseline)
            if result["total"] < best_a_total:
                best_a_total = result["total"]
                best_a = (low, high, result)

        if best_a:
            print(f"  Best Canny: L={best_a[0]} H={best_a[1]} "
                  f"→ total={best_a_total}")

        # --- Approach B: HoughLinesP sweep ---
        best_b = None
        best_b_total = 999
        if config != CONFIG_CLEAN:
            # Only sweep on non-clean (HoughLinesP will not find clean edges usefully)
            for thresh, min_len, max_gap in HOUGHP_SETTINGS:
                result = _evaluate_houghp(
                    image, metadata, label,
                    canny_low=50, canny_high=150,
                    hough_threshold=thresh,
                    min_line_length=min_len, max_line_gap=max_gap,
                )
                _print_result(
                    f"HoughLinesP thresh={thresh} minLen={min_len} gap={max_gap}",
                    result, baseline,
                )
                if result["total"] < best_b_total:
                    best_b_total = result["total"]
                    best_b = (thresh, min_len, max_gap, result)

            if best_b:
                print(f"  Best HoughLinesP: thresh={best_b[0]} minLen={best_b[1]} "
                      f"gap={best_b[2]} → total={best_b_total}")
        else:
            # Clean: just one run
            result = _evaluate_houghp(
                image, metadata, label,
                canny_low=50, canny_high=150,
                hough_threshold=30, min_line_length=15.0, max_line_gap=3.0,
            )
            _print_result("HoughLinesP", result, baseline)

    print("\n" + "=" * 70)
    print("I10 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
