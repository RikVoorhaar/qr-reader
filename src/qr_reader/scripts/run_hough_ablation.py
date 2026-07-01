"""Hough Pipeline Ablation Harness — Plan 008 Setup Phase.

Deterministic harness that runs the fixture pipeline (v12-default, v12-clean,
v5-default) for many parameter sets, writes structured CSV results, and
generates per-edge diagnostics.

Usage::

    .venv/bin/python -m qr_reader.scripts.run_hough_ablation \\
        --cases v12-default,v12-clean,v5-default \\
        --mode baseline \\
        --seed 42 \\
        --out out/baseline
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.edges import hysteresis_link
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.hough import LineSegment, build_hough_accumulator, hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INSIDE = 0
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8

CSV_HEADER = [
    "case",
    "config_key",
    "D",
    "A",
    "C",
    "B",
    "peak_hit_rate",
    "peak_snr_mean",
    "peak_snr_p05",
    "support_len_ratio_mean",
    "support_len_ratio_p05",
    "corner_reproj_median",
    "corner_reproj_p95",
    "n_zero_gt_roi",
    "runtime_median_ms",
    "runtime_p95_ms",
]

# ---------------------------------------------------------------------------
# Fixture configs
# ---------------------------------------------------------------------------


def _config_v12_default() -> AugmentationConfig:
    return AugmentationConfig(
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


def _config_v12_clean() -> AugmentationConfig:
    return AugmentationConfig(
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


def _config_v5_default() -> AugmentationConfig:
    return AugmentationConfig(
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


FIXTURE_SPECS: dict[str, tuple[AugmentationConfig, int, int, int]] = {
    "v12-default": (_config_v12_default(), 42, 640, 640),
    "v12-clean": (_config_v12_clean(), 42, 640, 640),
    "v5-default": (_config_v5_default(), 123, 640, 640),
}

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def _make_background(H: int = 640, W: int = 640) -> np.ndarray:
    xx = np.linspace(0, 1, W, dtype=np.float32).reshape(1, -1)
    yy = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1)
    bg = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
    return np.stack([bg] * 3, axis=-1)


# ---------------------------------------------------------------------------
# Cohen-Sutherland clipping
# ---------------------------------------------------------------------------


def _compute_outcode(
    x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> int:
    code = _INSIDE
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _TOP
    elif y > ymax:
        code |= _BOTTOM
    return code


def _clip_segment(
    p0: np.ndarray, p1: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> np.ndarray | None:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
    outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)
    while True:
        if (outcode0 | outcode1) == 0:
            return np.array([[x0, y0], [x1, y1]], dtype=np.float64)
        if (outcode0 & outcode1) != 0:
            return None
        oc = outcode0 if outcode0 != 0 else outcode1
        x, y = 0.0, 0.0
        if oc & _TOP:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymin
        elif oc & _BOTTOM:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymax
        elif oc & _RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmax
        elif oc & _LEFT:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmin
        if oc == outcode0:
            x0, y0 = x, y
            outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
        else:
            x1, y1 = x, y
            outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)


# ---------------------------------------------------------------------------
# GT edge geometry
# ---------------------------------------------------------------------------


def _compute_finder_edges(
    metadata: dict,
    roi_offset: tuple[int, int] | None = None,
    roi_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Compute 36 GT finder-pattern edges via module-grid homography.

    12 per finder (TL, TR, BL): 4 sides × 3 module boundaries (k=0,1,2 and k=5,6,7).
    Inner segments clipped: k_vis = min(k, 7-k) — visible feature span only.
    """
    corners = metadata["corners_qr"]
    N = metadata["N"]

    src_xy = np.array(
        [[0.0, 0.0], [float(N), 0.0], [float(N), float(N)], [0.0, float(N)]],
        dtype=np.float64,
    )
    dst_xy = np.array(
        [
            [float(corners["TL"][0]), float(corners["TL"][1])],
            [float(corners["TR"][0]), float(corners["TR"][1])],
            [float(corners["BR"][0]), float(corners["BR"][1])],
            [float(corners["BL"][0]), float(corners["BL"][1])],
        ],
        dtype=np.float64,
    )
    H = estimate_homography_dlt(src_xy, dst_xy)

    def _grid_to_image(row: float, col: float) -> np.ndarray:
        pt = np.array([[col, row]], dtype=np.float64)
        return project_points(H, pt)[0]

    finder_positions: dict[str, tuple[int, int]] = {
        "TL": (0, 0),
        "TR": (0, N - 7),
        "BL": (N - 7, 0),
    }

    TOP = [0, 1, 2]
    BOTTOM = [5, 6, 7]
    LEFT = [0, 1, 2]
    RIGHT = [5, 6, 7]

    results: list[dict] = []

    for finder_name, (r0, c0) in finder_positions.items():

        for side, offsets in [("top", TOP), ("bot", BOTTOM)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k), float(c0 + k_vis))
                b = _grid_to_image(float(r0 + k), float(c0 + 7 - k_vis))
                _add_edge(results, finder_name, side, k, a, b, roi_offset, roi_shape)

        for side, offsets in [("left", LEFT), ("right", RIGHT)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k_vis), float(c0 + k))
                b = _grid_to_image(float(r0 + 7 - k_vis), float(c0 + k))
                _add_edge(results, finder_name, side, k, a, b, roi_offset, roi_shape)

    return results


def _add_edge(
    results: list[dict],
    finder_name: str,
    side: str,
    k: int,
    a: np.ndarray,
    b: np.ndarray,
    roi_offset: tuple[int, int] | None,
    roi_shape: tuple[int, int] | None,
) -> None:
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        normal = np.array([1.0, 0.0], dtype=np.float64)
        rho = 0.0
    else:
        direction = d / length
        normal = np.array([direction[1], -direction[0]], dtype=np.float64)
        rho = float(normal @ a)
        if rho < 0:
            normal = -normal
            rho = -rho

    label = f"{finder_name}_{side}{k}"

    if roi_offset is not None and roi_shape is not None:
        row0, col0 = int(roi_offset[0]), int(roi_offset[1])
        H_img, W_img = int(roi_shape[0]), int(roi_shape[1])
        offset_xy = np.array([col0, row0], dtype=np.float64)
        a_local = a - offset_xy
        b_local = b - offset_xy
        rho_local = float(rho - normal @ offset_xy)
        if rho_local < 0:
            rho_local = -rho_local
            normal_local = -normal
        else:
            normal_local = normal.copy()
        clipped = _clip_segment(
            a_local, b_local, 0.0, float(W_img - 1), 0.0, float(H_img - 1)
        )
        segment = None if clipped is None else clipped
    else:
        normal_local = normal.copy()
        rho_local = rho
        segment = np.array([a.copy(), b.copy()], dtype=np.float64)

    results.append(
        {
            "label": label,
            "normal": normal_local,
            "rho": rho_local,
            "segment": segment,
        }
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _run_pipeline_to_rois(
    image: np.ndarray, *, blur_sigma: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], int]]:
    if image.ndim == 3:
        import cv2

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
        nms, angle = extract_thin_edges(roi, blur_sigma=blur_sigma)
        results.append((roi, nms, angle, bbox, ci))
    return results


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _normal_angle_deg(normal: np.ndarray) -> float:
    rad = np.arctan2(normal[1], normal[0])
    if rad < 0:
        rad += np.pi
    return np.rad2deg(rad)


def _angular_distance_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    dot = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


def _match_peak(
    gt_edge: dict,
    normals: np.ndarray,
    rhos: np.ndarray,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
) -> int:
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


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    cluster_idx: int
    bbox: tuple[int, int, int, int]
    roi_shape: tuple[int, int]
    normals: np.ndarray
    rhos: np.ndarray
    scores: np.ndarray
    segments: list[LineSegment]
    gt_edges: list[dict]
    failures_D: list[str]
    failures_A: list[str]
    failures_C: list[str]
    failures_B: list[str]
    peak_hit: list[bool]
    matched_peak_idx: list[int]
    support_len_ratios: list[float]
    corner_reproj_errors: list[float]
    runtime_vote_ms: float
    runtime_refine_ms: float
    nms: np.ndarray | None = None
    angle: np.ndarray | None = None
    acc_data: dict | None = None
    vote_cloud_classifications: list[str] = field(default_factory=list)
    vote_cloud_notes: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case_name: str
    config_key: str
    clusters: list[ClusterResult]
    runtime_total_ms: float


# ---------------------------------------------------------------------------
# Failure classification (extracted from test_hough_harness.py)
# ---------------------------------------------------------------------------


def _classify_failures(
    gt_edges: list[dict],
    normals: np.ndarray,
    rhos: np.ndarray,
    scores: np.ndarray,
    nms: np.ndarray,
    angle: np.ndarray,
    cluster_idx: int,
    *,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
    endpoint_tol: float = 5.0,
    strength_threshold: float = 400.0,
    angular_match_deg: float = 12.0,
    gap_tolerance: float = 2.0,
    distance_thresh: float = 1.5,
    support_mask: np.ndarray | None = None,
    support_dilate: int = 0,
) -> tuple[list[str], list[str], list[str], list[str], list[bool], list[int],
           list[LineSegment], list[float], list[float]]:
    D_failures: list[str] = []
    A_failures: list[str] = []
    C_failures: list[str] = []
    B_failures: list[str] = []
    peak_hit: list[bool] = []
    matched_idx: list[int] = []
    segments: list[LineSegment] = []
    len_ratios: list[float] = []
    reproj_errors: list[float] = []

    for gt in gt_edges:
        if gt["segment"] is None:
            peak_hit.append(False)
            matched_idx.append(-1)
            segments.append(_degenerate_segment(gt))
            len_ratios.append(0.0)
            reproj_errors.append(float("inf"))
            continue

        match_i = _match_peak(gt, normals, rhos, angle_tol_deg, rho_tol)
        if match_i < 0:
            D_failures.append(
                f"[C{cluster_idx}] {gt['label']}: no Hough peak within "
                f"5deg and 5px of gt"
            )
            peak_hit.append(False)
            matched_idx.append(-1)
            segments.append(_degenerate_segment(gt))
            len_ratios.append(0.0)
            reproj_errors.append(float("inf"))
            continue

        peak_hit.append(True)
        matched_idx.append(match_i)

        seg = refine_line(
            normals[match_i],
            float(rhos[match_i]),
            float(scores[match_i]),
            nms,
            angle,
            gap_tolerance=gap_tolerance,
            distance_thresh=distance_thresh,
            support_mask=support_mask,
            support_dilate=support_dilate,
        )
        segments.append(seg)

        gt_seg = gt["segment"]
        direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
        gt_proj = gt_seg @ direction
        gt_span = abs(gt_proj[1] - gt_proj[0])

        if np.all(seg.endpoints == 0):
            A_failures.append(
                f"[C{cluster_idx}] {gt['label']}: refined segment is degenerate"
            )
            len_ratios.append(0.0)
            reproj_errors.append(float("inf"))
            continue

        ep_proj = seg.endpoints @ direction
        seg_span = abs(ep_proj[1] - ep_proj[0])

        ratio = min(seg_span, gt_span) / max(seg_span, gt_span) if max(seg_span, gt_span) > 0 else 0.0
        len_ratios.append(ratio)

        if seg_span < 0.8 * gt_span:
            A_failures.append(
                f"[C{cluster_idx}] {gt['label']}: span={seg_span:.1f}px "
                f"< 80% of gt_span={gt_span:.1f}px"
            )

        # Corner reprojection error: find the max distance from each segment
        # endpoint to its closest GT endpoint.
        dists = np.linalg.norm(seg.endpoints[:, None] - gt_seg[None, :], axis=2)
        # min distance from each segment endpoint to any GT endpoint
        min_dists = dists.min(axis=1)
        er = float(min_dists.max())
        reproj_errors.append(er)

        # C failure: any GT endpoint > endpoint_tol from ALL segment endpoints
        c_fail = False
        for gt_ep in gt_seg:
            ep_dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
            if ep_dists.min() > endpoint_tol:
                C_failures.append(
                    f"[C{cluster_idx}] {gt['label']}: refined endpoints "
                    f"too far from gt"
                )
                break

    # Failure B: phantoms
    gt_normals = np.array([e["normal"] for e in gt_edges if e["segment"] is not None])
    for i in range(len(normals)):
        matched = any(
            gt["segment"] is not None
            and _match_peak(gt, normals[[i]], rhos[[i]], angle_tol_deg, rho_tol) >= 0
            for gt in gt_edges
        )
        if matched:
            continue
        if len(gt_normals) > 0:
            min_ang = min(_angular_distance_deg(normals[i], gn) for gn in gt_normals)
            if min_ang < angular_match_deg:
                continue
        seg = refine_line(
            normals[i],
            float(rhos[i]),
            float(scores[i]),
            nms,
            angle,
            gap_tolerance=gap_tolerance,
            distance_thresh=distance_thresh,
            support_mask=support_mask,
            support_dilate=support_dilate,
        )
        if np.all(seg.endpoints == 0):
            continue
        ys, xs = np.nonzero(np.asarray(nms))
        strengths = nms[ys, xs]
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        dists = np.abs(points @ seg.normal - seg.rho)
        mask = dists < distance_thresh
        if np.sum(mask) == 0:
            continue
        mean_strength = float(strengths[mask].mean())
        if mean_strength > strength_threshold:
            B_failures.append(
                f"[C{cluster_idx}] phantom peak {i}: "
                f"mean NMS strength={mean_strength:.1f}"
            )

    return D_failures, A_failures, C_failures, B_failures, peak_hit, matched_idx, segments, len_ratios, reproj_errors


def _degenerate_segment(gt_edge: dict) -> LineSegment:
    return LineSegment(
        normal=gt_edge["normal"].copy(),
        rho=gt_edge["rho"],
        endpoints=np.zeros((2, 2), dtype=np.float64),
        vote_score=0.0,
    )


# ---------------------------------------------------------------------------
# Vote-cloud audit helpers
# ---------------------------------------------------------------------------


def _normal_to_theta(normal: np.ndarray) -> float:
    rad = float(np.arctan2(normal[1], normal[0]))
    if rad < 0:
        rad += np.pi
    return rad


def _theta_to_idx(theta: float, theta_step_rad: float, n_theta: int) -> int:
    return int(np.round(theta / theta_step_rad)) % n_theta


def _classify_vote_cloud(
    gt_edge: dict,
    acc: np.ndarray,
    acc_data: dict,
) -> tuple[str, str]:
    """Classify why a GT edge has weak/no peak in the Hough accumulator.

    Returns ``(classification, diagnostic_note)``.  Classification is one of
    ``empty``, ``vote_dilution``, ``theta_spread``, ``rho_spread``,
    ``origin_shift``.

    Parameters
    ----------
    gt_edge : dict
        GT edge with ``normal``, ``rho``, ``segment``.
    acc : ndarray, shape (n_theta, n_rho)
        Hough accumulator.
    acc_data : dict
        From ``build_hough_accumulator`` (contains n_theta, n_rho, theta_step_rad).
    """
    if gt_edge["segment"] is None:
        return "no_roi_overlap", "GT edge does not intersect ROI"

    gt_theta = _normal_to_theta(gt_edge["normal"])
    gt_rho = gt_edge["rho"]
    n_theta = acc_data["n_theta"]
    n_rho = acc_data["n_rho"]
    theta_step_deg = np.rad2deg(acc_data["theta_step_rad"])
    rho_step = 1.0  # baseline

    gt_theta_idx = _theta_to_idx(gt_theta, acc_data["theta_step_rad"], n_theta)
    gt_rho_idx = int(np.round(gt_rho / rho_step))
    gt_rho_idx = max(0, min(n_rho - 1, gt_rho_idx))

    # search window: ±5° and ±5 px around GT
    theta_tol_deg = 5.0
    rho_tol_px = 5.0
    dtheta_bins = max(1, int(np.ceil(theta_tol_deg / theta_step_deg)))
    drho_bins = max(1, int(np.ceil(rho_tol_px / rho_step)))

    # 1. Empty check: sum of votes within window
    window_sum = 0.0
    for dt in range(-dtheta_bins, dtheta_bins + 1):
        tt = (gt_theta_idx + dt) % n_theta
        r0 = max(0, gt_rho_idx - drho_bins)
        r1 = min(n_rho, gt_rho_idx + drho_bins + 1)
        window_sum += float(acc[tt, r0:r1].sum())

    if window_sum < 10.0:
        return (
            "empty",
            f"vote sum in ±{theta_tol_deg:.0f}° × ±{rho_tol_px:.0f} px = {window_sum:.1f}",
        )

    acc_max = float(acc.max())
    gt_bin_value = float(acc[gt_theta_idx, gt_rho_idx])

    # 2. Vote dilution: GT bin exists but is weak
    if gt_bin_value > 0 and gt_bin_value < 0.25 * acc_max:
        return (
            "vote_dilution",
            f"GT bin={gt_bin_value:.1f}, max={acc_max:.1f}, ratio={gt_bin_value / acc_max:.3f}",
        )

    # 3. Theta spread: votes in GT rho-bin exist at non-GT theta bins
    gt_rho_band = acc[:, gt_rho_idx]
    non_gt_theta_sum = float(gt_rho_band.sum()) - gt_bin_value
    if non_gt_theta_sum > 0 and gt_bin_value < 0.5 * float(gt_rho_band.sum()):
        theta_peak_idx = int(np.argmax(gt_rho_band))
        if theta_peak_idx != gt_theta_idx:
            peak_deg = theta_peak_idx * theta_step_deg
            gt_deg = gt_theta_idx * theta_step_deg
            return (
                "theta_spread",
                f"peak at theta={peak_deg:.1f}° vs GT={gt_deg:.1f}°",
            )

    # 4. Rho spread: votes at GT theta exist in non-GT rho bins
    gt_theta_band = acc[gt_theta_idx, :]
    non_gt_rho_sum = float(gt_theta_band.sum()) - gt_bin_value
    if non_gt_rho_sum > 0 and gt_bin_value < 0.5 * float(gt_theta_band.sum()):
        rho_peak_idx = int(np.argmax(gt_theta_band))
        if rho_peak_idx != gt_rho_idx:
            return (
                "rho_spread",
                f"peak at rho={rho_peak_idx * rho_step:.1f} px vs GT={gt_rho:.1f} px",
            )

    # 5. Origin shift: in the theta band, the peak is offset from GT rho
    r0 = max(0, gt_rho_idx - drho_bins * 2)
    r1 = min(n_rho, gt_rho_idx + drho_bins * 2 + 1)
    theta_band_rho = np.zeros(n_rho, dtype=np.float64)
    for dt in range(-dtheta_bins, dtheta_bins + 1):
        tt = (gt_theta_idx + dt) % n_theta
        theta_band_rho[r0:r1] += acc[tt, r0:r1]
    peak_rho_idx = int(np.argmax(theta_band_rho[r0:r1])) + r0
    rho_offset = abs(peak_rho_idx - gt_rho_idx) * rho_step
    if rho_offset >= 2.0:
        return (
            "origin_shift",
            f"peak rho offset from GT by {rho_offset:.1f} px",
        )

    # Fallback: should not reach here, but mark as weak
    return (
        "vote_dilution",
        f"ambiguous — GT bin={gt_bin_value:.1f}, window_sum={window_sum:.1f}",
    )


# ---------------------------------------------------------------------------
# Vote-audit run mode
# ---------------------------------------------------------------------------


def run_vote_audit_case(
    case_name: str,
    seed: int,
    H: int,
    W: int,
    config: AugmentationConfig,
    background: np.ndarray,
    hough_kwargs: dict | None = None,
) -> CaseResult:
    hough_kwargs = hough_kwargs or {}
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    image, metadata = generate_sample(rng, config, background)

    roi_results = _run_pipeline_to_rois(image)

    cluster_results: list[ClusterResult] = []
    for roi, nms, angle, bbox, ci in roi_results:
        t_vote = time.perf_counter()
        result = hough_vote_peaks(nms, angle, return_acc=True, **hough_kwargs)
        if len(result) == 4:
            normals, rhos, scores, acc_data = result
        else:
            normals, rhos, scores = result
            acc_data = {}
        t_vote_elapsed = (time.perf_counter() - t_vote) * 1000.0

        gt_edges = _compute_finder_edges(
            metadata, roi_offset=(bbox[0], bbox[2]), roi_shape=roi.shape
        )

        t_refine = time.perf_counter()
        Df, Af, Cf, Bf, hits, m_idxs, segs, len_rs, rep_es = _classify_failures(
            gt_edges, normals, rhos, scores, nms, angle, cluster_idx=ci,
        )
        t_refine_elapsed = (time.perf_counter() - t_refine) * 1000.0

        vc_classifications: list[str] = []
        vc_notes: list[str] = []

        if acc_data:
            for gt in gt_edges:
                cls, note = _classify_vote_cloud(gt, acc_data["acc"], acc_data)
                vc_classifications.append(cls)
                vc_notes.append(note)
        else:
            vc_classifications = [""] * len(gt_edges)
            vc_notes = [""] * len(gt_edges)

        cluster_results.append(
            ClusterResult(
                cluster_idx=ci,
                bbox=bbox,
                roi_shape=roi.shape,
                normals=normals,
                rhos=rhos,
                scores=scores,
                segments=segs,
                gt_edges=gt_edges,
                failures_D=Df,
                failures_A=Af,
                failures_C=Cf,
                failures_B=Bf,
                peak_hit=hits,
                matched_peak_idx=m_idxs,
                support_len_ratios=len_rs,
                corner_reproj_errors=rep_es,
                runtime_vote_ms=t_vote_elapsed,
                runtime_refine_ms=t_refine_elapsed,
                nms=nms,
                angle=angle,
                acc_data=acc_data,
                vote_cloud_classifications=vc_classifications,
                vote_cloud_notes=vc_notes,
            )
        )

    t_total = (time.perf_counter() - t0) * 1000.0
    return CaseResult(
        case_name=case_name,
        config_key="vote_audit",
        clusters=cluster_results,
        runtime_total_ms=t_total,
    )


# ---------------------------------------------------------------------------
# Harness core
# ---------------------------------------------------------------------------


def run_case(
    case_name: str,
    config_key: str,
    seed: int,
    H: int,
    W: int,
    config: AugmentationConfig,
    background: np.ndarray,
    hough_kwargs: dict | None = None,
    refine_kwargs: dict | None = None,
) -> CaseResult:
    hough_kwargs = hough_kwargs or {}
    refine_kwargs = refine_kwargs or {}
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    image, metadata = generate_sample(rng, config, background)

    roi_results = _run_pipeline_to_rois(image)

    cluster_results: list[ClusterResult] = []
    for roi, nms, angle, bbox, ci in roi_results:
        t_vote = time.perf_counter()

        # Extract internal params (copy to avoid mutating for later clusters)
        cur_kwargs = dict(hough_kwargs)
        _hysteresis = cur_kwargs.pop("_hysteresis", None)
        _hysteresis_high_pct = cur_kwargs.pop("_hysteresis_high_pct", 90.0)
        _hysteresis_low_pct = cur_kwargs.pop("_hysteresis_low_pct", 70.0)

        result = hough_vote_peaks(nms, angle, **cur_kwargs)
        if len(result) == 4:
            normals, rhos, scores, acc_data_out = result
        else:
            normals, rhos, scores = result
            acc_data_out = None
        t_vote_elapsed = (time.perf_counter() - t_vote) * 1000.0

        gt_edges = _compute_finder_edges(
            metadata, roi_offset=(bbox[0], bbox[2]), roi_shape=roi.shape
        )

        # Compute hysteresis linked mask if requested
        hysteresis_mask = None
        if _hysteresis == "lite":
            hysteresis_mask = hysteresis_link(
                nms, angle,
                high_pct=_hysteresis_high_pct,
                low_pct=_hysteresis_low_pct,
            )

        t_refine = time.perf_counter()
        Df, Af, Cf, Bf, hits, m_idxs, segs, len_rs, rep_es = _classify_failures(
            gt_edges, normals, rhos, scores, nms, angle, cluster_idx=ci,
            support_mask=hysteresis_mask,
            **refine_kwargs,
        )
        t_refine_elapsed = (time.perf_counter() - t_refine) * 1000.0

        cluster_results.append(
            ClusterResult(
                cluster_idx=ci,
                bbox=bbox,
                roi_shape=roi.shape,
                normals=normals,
                rhos=rhos,
                scores=scores,
                segments=segs,
                gt_edges=gt_edges,
                failures_D=Df,
                failures_A=Af,
                failures_C=Cf,
                failures_B=Bf,
                peak_hit=hits,
                matched_peak_idx=m_idxs,
                support_len_ratios=len_rs,
                corner_reproj_errors=rep_es,
                runtime_vote_ms=t_vote_elapsed,
                runtime_refine_ms=t_refine_elapsed,
                acc_data=acc_data_out,
            )
        )

    t_total = (time.perf_counter() - t0) * 1000.0
    return CaseResult(
        case_name=case_name,
        config_key=config_key,
        clusters=cluster_results,
        runtime_total_ms=t_total,
    )


def aggregate_result(result: CaseResult) -> dict[str, Any]:
    D = sum(len(c.failures_D) for c in result.clusters)
    A = sum(len(c.failures_A) for c in result.clusters)
    C = sum(len(c.failures_C) for c in result.clusters)
    B = sum(len(c.failures_B) for c in result.clusters)

    all_hits = []
    all_hit_ratios = []
    all_len_ratios = []
    all_reproj = []
    n_zero_gt = 0
    for c in result.clusters:
        gt_with_seg = [e for e in c.gt_edges if e["segment"] is not None]
        if len(gt_with_seg) == 0:
            n_zero_gt += 1
            continue
        n_hit = sum(1 for h in c.peak_hit if h)
        all_hit_ratios.append(n_hit / len(gt_with_seg))
        for hit, pe in zip(c.peak_hit, c.peak_hit):
            all_hits.append(hit)

        for lr in c.support_len_ratios:
            if lr > 0:
                all_len_ratios.append(lr)
        for re in c.corner_reproj_errors:
            if re < float("inf"):
                all_reproj.append(re)

    hit_rate = float(np.mean(all_hit_ratios)) if all_hit_ratios else 0.0

    # Peak SNR is computed per GT edge: ratio of GT-bin score to mean non-GT
    # score in the same theta band.  If acc_data is available, compute real SNR.
    snr_vals: list[float] = []
    for c in result.clusters:
        if c.acc_data is not None:
            acc = c.acc_data["acc"]
            n_theta = c.acc_data["n_theta"]
            n_rho = c.acc_data["n_rho"]
            theta_step_deg = np.rad2deg(c.acc_data["theta_step_rad"])
            for gt in c.gt_edges:
                if gt["segment"] is None:
                    continue
                gt_theta = _normal_to_theta(gt["normal"])
                gt_rho = gt["rho"]
                gt_ti = _theta_to_idx(gt_theta, c.acc_data["theta_step_rad"], n_theta)
                gt_ri = int(np.round(gt_rho))
                gt_ri = max(0, min(n_rho - 1, gt_ri))
                gt_val = float(acc[gt_ti, gt_ri])
                non_gt_mean = 0.0
                count = 0
                dtheta_bins = max(1, int(np.ceil(5.0 / theta_step_deg)))
                for dt in range(-dtheta_bins, dtheta_bins + 1):
                    tt = (gt_ti + dt) % n_theta
                    for ri in range(n_rho):
                        if tt == gt_ti and ri == gt_ri:
                            continue
                        non_gt_mean += float(acc[tt, ri])
                        count += 1
                if count > 0 and non_gt_mean > 0:
                    snr = gt_val / (non_gt_mean / count)
                    snr_vals.append(snr)

    snr_mean = float(np.mean(snr_vals)) if snr_vals else float("nan")
    snr_p05 = float(np.percentile(snr_vals, 5)) if snr_vals else float("nan")

    # Support length ratio statistics
    lr_mean = float(np.mean(all_len_ratios)) if all_len_ratios else float("nan")
    lr_p05 = float(np.percentile(all_len_ratios, 5)) if all_len_ratios else float("nan")

    # Corner reprojection
    pr_median = float(np.median(all_reproj)) if all_reproj else float("nan")
    pr_p95 = float(np.percentile(all_reproj, 95)) if all_reproj else float("nan")

    # Runtime
    runtimes = [c.runtime_vote_ms + c.runtime_refine_ms for c in result.clusters]
    rt_median = float(np.median(runtimes)) if runtimes else float("nan")
    rt_p95 = float(np.percentile(runtimes, 95)) if runtimes else float("nan")

    return {
        "case": result.case_name,
        "config_key": result.config_key,
        "D": D,
        "A": A,
        "C": C,
        "B": B,
        "peak_hit_rate": round(hit_rate, 4),
        "peak_snr_mean": snr_mean,
        "peak_snr_p05": snr_p05,
        "support_len_ratio_mean": round(lr_mean, 4) if not np.isnan(lr_mean) else "",
        "support_len_ratio_p05": round(lr_p05, 4) if not np.isnan(lr_p05) else "",
        "corner_reproj_median": round(pr_median, 2) if not np.isnan(pr_median) else "",
        "corner_reproj_p95": round(pr_p95, 2) if not np.isnan(pr_p95) else "",
        "n_zero_gt_roi": n_zero_gt,
        "runtime_median_ms": round(rt_median, 2) if not np.isnan(rt_median) else "",
        "runtime_p95_ms": round(rt_p95, 2) if not np.isnan(rt_p95) else "",
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _write_diagnostics(result: CaseResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for cl in result.clusters:
        cluster_dir = out_dir / f"cluster_{cl.cluster_idx}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        _plot_roi_overlay(cl, cluster_dir)
        _plot_edge_angle_histogram(cl, cluster_dir)
        _plot_accumulator_heatmaps(cl, cluster_dir)
        _plot_support_maps(cl, cluster_dir)
        _plot_support_density(cl, cluster_dir)
        _plot_rho_vs_theta(cl, cluster_dir)
        _write_cluster_summary(cl, cluster_dir)

    plt.close("all")


def _plot_roi_overlay(cl: ClusterResult, out_dir: Path) -> None:
    r0, r1, c0, c1 = cl.bbox
    H, W = cl.roi_shape
    gt_with_seg = [e for e in cl.gt_edges if e["segment"] is not None]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(f"ROI Overlay — Cluster {cl.cluster_idx}")

    # GT edges
    colors = plt.cm.tab10.colors
    for i, gt in enumerate(gt_with_seg):
        seg = gt["segment"]
        ax.plot(
            [seg[0][0], seg[1][0]],
            [seg[0][1], seg[1][1]],
            color=colors[i % len(colors)],
            linewidth=2,
            label=gt["label"],
        )

    # Hough peaks (clipped to ROI)
    for i in range(len(cl.normals)):
        n = cl.normals[i]
        rho = cl.rhos[i]
        d = np.array([-n[1], n[0]])
        center = rho * n
        pts_along = np.array([center - 500 * d, center + 500 * d])
        clipped = _clip_segment(pts_along[0], pts_along[1],
                                0.0, float(W - 1), 0.0, float(H - 1))
        if clipped is not None:
            ax.plot(
                [clipped[0][0], clipped[1][0]],
                [clipped[0][1], clipped[1][1]],
                "k--",
                alpha=0.3,
                linewidth=0.5,
            )

    # Cluster centre
    center_row = (r0 + r1) / 2
    center_col = (c0 + c1) / 2
    ax.plot(center_col - c0, center_row - r0, "r+", markersize=10, label="ROI centre")

    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_overlay.png", dpi=150)
    plt.close(fig)


def _plot_edge_angle_histogram(cl: ClusterResult, out_dir: Path) -> None:
    if cl.nms is None or cl.angle is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Edge-Angle Histogram — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "edge-angle histogram\n(needs NMS/angle from pipeline)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "edge_angle_histogram.png", dpi=150)
        plt.close(fig)
        return

    nms = np.asarray(cl.nms)
    angle_arr = np.asarray(cl.angle)
    ys, xs = np.nonzero(nms)
    if len(ys) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Edge-Angle Histogram — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "no edge pixels", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "edge_angle_histogram.png", dpi=150)
        plt.close(fig)
        return

    thetas = np.fmod(angle_arr[ys, xs], np.pi)
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)
    strengths = nms[ys, xs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Edge-Angle Histogram — Cluster {cl.cluster_idx}")
    ax.hist(np.rad2deg(thetas), bins=90, range=(0, 180), weights=strengths,
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Weighted count (NMS magnitude)")

    # GT edge normals as vertical lines
    colors = plt.cm.tab10.colors
    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None:
            continue
        gt_theta = _normal_to_theta(gt["normal"])
        ax.axvline(np.rad2deg(gt_theta), color=colors[i % len(colors)],
                   linestyle="--", linewidth=1, label=f"GT {gt['label']}")

    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "edge_angle_histogram.png", dpi=150)
    plt.close(fig)


def _plot_accumulator_heatmaps(cl: ClusterResult, out_dir: Path) -> None:
    if cl.acc_data is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Accumulator — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "accumulator heatmaps\n(no acc_data available)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "accumulator_heatmaps.png", dpi=150)
        plt.close(fig)
        return

    acc = cl.acc_data["acc"]
    n_theta = cl.acc_data["n_theta"]
    n_rho = cl.acc_data["n_rho"]
    theta_step_deg = np.rad2deg(cl.acc_data["theta_step_rad"])

    theta_extent = [0, min(n_theta * theta_step_deg, 180)]
    rho_extent = [0, n_rho]

    # Full accumulator heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Accumulator — Cluster {cl.cluster_idx}")
    im = ax.imshow(acc.T, origin="lower", aspect="auto",
                   extent=[theta_extent[0], theta_extent[1], rho_extent[0], rho_extent[1]],
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="Vote weight")

    # GT edge markers
    colors = plt.cm.tab10.colors
    gt_labels = []
    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None:
            continue
        gt_theta = np.rad2deg(_normal_to_theta(gt["normal"]))
        gt_rho = gt["rho"]
        ax.plot(gt_theta, gt_rho, "o", color=colors[i % len(colors)],
                markersize=8, markeredgewidth=1.5, markeredgecolor="white")
        gt_labels.append(gt["label"])

    # Detected peaks
    for i in range(len(cl.normals)):
        peak_theta = np.rad2deg(_normal_to_theta(cl.normals[i]))
        ax.plot(peak_theta, cl.rhos[i], "x", color="cyan", markersize=6, markeredgewidth=1)

    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("ρ (px)")
    fig.tight_layout()
    fig.savefig(out_dir / "accumulator_heatmaps.png", dpi=150)
    plt.close(fig)

    # Per-GT-edge zoomed heatmaps
    per_edge_dir = out_dir / "accumulator_per_edge"
    per_edge_dir.mkdir(parents=True, exist_ok=True)
    zoom_deg = 15.0
    zoom_rho = 15
    dtheta_bins = max(1, int(np.ceil(zoom_deg / theta_step_deg)))

    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None:
            continue
        gt_theta = _normal_to_theta(gt["normal"])
        gt_rho = gt["rho"]
        gt_ti = _theta_to_idx(gt_theta, cl.acc_data["theta_step_rad"], n_theta)
        gt_ri = int(np.round(gt_rho))
        gt_ri = max(0, min(n_rho - 1, gt_ri))

        t0 = max(0, gt_ti - dtheta_bins)
        t1 = min(n_theta, gt_ti + dtheta_bins + 1)
        r0 = max(0, gt_ri - zoom_rho)
        r1 = min(n_rho, gt_ri + zoom_rho + 1)

        zoom_acc = acc[t0:t1, r0:r1]
        if zoom_acc.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        label = gt["label"]

        # Classification info
        cls = ""
        if i < len(cl.vote_cloud_classifications):
            cls = cl.vote_cloud_classifications[i]
        note = ""
        if i < len(cl.vote_cloud_notes):
            note = cl.vote_cloud_notes[i]

        hit_str = "HIT" if i < len(cl.peak_hit) and cl.peak_hit[i] else "MISS"
        ax.set_title(f"{label}  [{hit_str}]  {cls}\n{note}")

        im = ax.imshow(zoom_acc.T, origin="lower", aspect="auto",
                       extent=[t0 * theta_step_deg, t1 * theta_step_deg, r0, r1],
                       cmap="inferno")
        plt.colorbar(im, ax=ax, label="Vote weight")

        # GT bin marker
        ax.plot(gt_ti * theta_step_deg, gt_rho, "o", color="lime",
                markersize=10, markeredgewidth=1.5, markeredgecolor="white",
                label="GT bin")

        # Nearest peak marker
        if i < len(cl.matched_peak_idx) and cl.matched_peak_idx[i] >= 0:
            mi = cl.matched_peak_idx[i]
            p_theta = np.rad2deg(_normal_to_theta(cl.normals[mi]))
            ax.plot(p_theta, cl.rhos[mi], "x", color="cyan",
                    markersize=10, markeredgewidth=2, label="Peak")

        ax.set_xlabel("θ (deg)")
        ax.set_ylabel("ρ (px)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        fig.savefig(per_edge_dir / f"acc_{safe_label}.png", dpi=150)
        plt.close(fig)


def _plot_support_maps(cl: ClusterResult, out_dir: Path) -> None:
    if cl.nms is None or cl.angle is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Support Maps — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "per-peak support maps\n(needs NMS/angle data)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "support_maps.png", dpi=150)
        plt.close(fig)
        return

    nms = np.asarray(cl.nms)
    H, W = nms.shape

    per_edge_dir = out_dir / "support_per_edge"
    per_edge_dir.mkdir(parents=True, exist_ok=True)
    distance_thresh = 1.5

    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None or i >= len(cl.segments):
            continue
        seg = cl.segments[i]
        if np.all(seg.endpoints == 0):
            continue

        ys_s, xs_s = np.nonzero(nms)
        pts = np.column_stack([xs_s.astype(np.float64), ys_s.astype(np.float64)])
        dists = np.abs(pts @ seg.normal - seg.rho)
        inlier = dists < distance_thresh

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        label = gt["label"]
        ax.set_title(f"Support Map — Cluster {cl.cluster_idx} — {label}")

        ax.scatter(xs_s[inlier], ys_s[inlier], c="lime", s=2, alpha=0.6, label="inlier")
        ax.scatter(xs_s[~inlier], ys_s[~inlier], c="gray", s=1, alpha=0.2, label="outlier")

        if not np.all(seg.endpoints == 0):
            ax.plot([seg.endpoints[0][0], seg.endpoints[1][0]],
                    [seg.endpoints[0][1], seg.endpoints[1][1]],
                    "r-", linewidth=2, label="refined")

        if gt["segment"] is not None:
            gs = gt["segment"]
            ax.plot([gs[0][0], gs[1][0]], [gs[0][1], gs[1][1]],
                    "b--", linewidth=2, label="GT")

        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        fig.savefig(per_edge_dir / f"support_{safe_label}.png", dpi=150)
        plt.close(fig)


def _plot_support_density(cl: ClusterResult, out_dir: Path) -> None:
    if cl.nms is None or cl.angle is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Support Density — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "support-density plots\n(needs NMS/angle data)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "support_density.png", dpi=150)
        plt.close(fig)
        return

    nms = np.asarray(cl.nms)
    distance_thresh = 1.5
    per_edge_dir = out_dir / "density_per_edge"
    per_edge_dir.mkdir(parents=True, exist_ok=True)

    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None or i >= len(cl.segments):
            continue
        seg = cl.segments[i]
        if np.all(seg.endpoints == 0):
            continue

        ys_s, xs_s = np.nonzero(nms)
        pts = np.column_stack([xs_s.astype(np.float64), ys_s.astype(np.float64)])
        dists = np.abs(pts @ seg.normal - seg.rho)
        inlier = dists < distance_thresh
        inlier_pts = pts[inlier]

        if len(inlier_pts) == 0:
            continue

        direction = np.array([-seg.normal[1], seg.normal[0]])
        proj = inlier_pts @ direction

        fig, ax = plt.subplots(figsize=(8, 4))
        label = gt["label"]
        ax.set_title(f"Support Density — Cluster {cl.cluster_idx} — {label}")
        ax.hist(proj, bins=50, color="steelblue", edgecolor="white")
        ax.set_xlabel("t (projection along line, px)")
        ax.set_ylabel("Count")

        if gt["segment"] is not None:
            gs = gt["segment"]
            gt_proj = gs @ direction
            for gp in gt_proj:
                ax.axvline(gp, color="red", linestyle="--", linewidth=1)

        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        fig.savefig(per_edge_dir / f"density_{safe_label}.png", dpi=150)
        plt.close(fig)


def _plot_rho_vs_theta(cl: ClusterResult, out_dir: Path) -> None:
    if cl.acc_data is None or cl.nms is None or cl.angle is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Rho-vs-Theta — Cluster {cl.cluster_idx}")
        ax.text(0.5, 0.5, "rho-vs-theta scatter\n(needs acc_data + NMS/angle)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "rho_vs_theta.png", dpi=150)
        plt.close(fig)
        return

    acc_data = cl.acc_data
    n_theta = acc_data["n_theta"]
    n_rho = acc_data["n_rho"]
    theta_step_deg = np.rad2deg(acc_data["theta_step_rad"])
    theta_idx = acc_data["theta_idx"]
    rho_idx = acc_data["rho_idx"]
    strengths = acc_data["strengths"]
    valid = theta_idx >= 0

    per_edge_dir = out_dir / "rho_vs_theta_per_edge"
    per_edge_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    zoom_deg_rvt = 10.0
    zoom_rho_rvt = 10
    dtheta_bins_rvt = max(1, int(np.ceil(zoom_deg_rvt / theta_step_deg)))

    for i, gt in enumerate(cl.gt_edges):
        if gt["segment"] is None:
            continue
        gt_theta = _normal_to_theta(gt["normal"])
        gt_rho = gt["rho"]
        gt_ti = _theta_to_idx(gt_theta, acc_data["theta_step_rad"], n_theta)
        gt_ri = int(np.round(gt_rho))
        gt_ri = max(0, min(n_rho - 1, gt_ri))

        t0 = max(0, gt_ti - dtheta_bins_rvt)
        t1 = min(n_theta, gt_ti + dtheta_bins_rvt + 1)
        r0 = max(0, gt_ri - zoom_rho_rvt)
        r1 = min(n_rho, gt_ri + zoom_rho_rvt + 1)

        in_window = valid & (theta_idx >= t0) & (theta_idx < t1) & (rho_idx >= r0) & (rho_idx < r1)

        fig, ax = plt.subplots(figsize=(8, 6))
        label = gt["label"]
        ax.set_title(f"ρ-vs-θ Votes — Cluster {cl.cluster_idx} — {label}")

        if in_window.any():
            tw = theta_idx[in_window].astype(np.float64) * theta_step_deg
            rw = rho_idx[in_window].astype(np.float64)
            sw = strengths[in_window]
            sc = ax.scatter(tw, rw, c=sw, s=5 * np.sqrt(sw / sw.max() + 0.1),
                           cmap="inferno", alpha=0.6)
            plt.colorbar(sc, ax=ax, label="Strength")

        ax.plot(gt_ti * theta_step_deg, gt_rho, "o", color="lime",
                markersize=12, markeredgewidth=1.5, markeredgecolor="white",
                label="GT bin")

        if i < len(cl.matched_peak_idx) and cl.matched_peak_idx[i] >= 0:
            mi = cl.matched_peak_idx[i]
            p_theta = np.rad2deg(_normal_to_theta(cl.normals[mi]))
            ax.plot(p_theta, cl.rhos[mi], "x", color="cyan",
                    markersize=12, markeredgewidth=2, label="Peak")

        ax.set_xlabel("θ (deg)")
        ax.set_ylabel("ρ (px)")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        safe_label = label.replace("/", "_").replace(" ", "_")
        fig.savefig(per_edge_dir / f"rvt_{safe_label}.png", dpi=150)
        plt.close(fig)


def _write_cluster_summary(cl: ClusterResult, out_dir: Path) -> None:
    path = out_dir / "summary.txt"
    lines = [
        f"Cluster {cl.cluster_idx}",
        f"  BBOX: {cl.bbox}",
        f"  ROI shape: {cl.roi_shape}",
        f"  Hough peaks: {len(cl.normals)}",
        f"  GT edges with segment: {sum(1 for e in cl.gt_edges if e['segment'] is not None)}",
        f"  Failures: D={len(cl.failures_D)} A={len(cl.failures_A)} C={len(cl.failures_C)} B={len(cl.failures_B)}",
        f"  Peak hit rate: {sum(cl.peak_hit)}/{len(cl.peak_hit)}",
    ]
    if cl.failures_D:
        lines.append(f"  D failures ({len(cl.failures_D)}):")
        for f in cl.failures_D:
            lines.append(f"    {f}")
    if cl.failures_A:
        lines.append(f"  A failures ({len(cl.failures_A)}):")
        for f in cl.failures_A:
            lines.append(f"    {f}")
    if cl.failures_C:
        lines.append(f"  C failures ({len(cl.failures_C)}):")
        for f in cl.failures_C:
            lines.append(f"    {f}")
    if cl.failures_B:
        lines.append(f"  B failures ({len(cl.failures_B)}):")
        for f in cl.failures_B:
            lines.append(f"    {f}")

    if cl.vote_cloud_classifications:
        lines.append("")
        lines.append("  Vote-cloud classifications (per GT edge):")
        for i, gt in enumerate(cl.gt_edges):
            label = gt["label"]
            cls = cl.vote_cloud_classifications[i] if i < len(cl.vote_cloud_classifications) else "?"
            note = cl.vote_cloud_notes[i] if i < len(cl.vote_cloud_notes) else ""
            d_flag = " [D]" if any(label in f for f in cl.failures_D) else ""
            lines.append(f"    {label}{d_flag}: {cls}  — {note}")
    lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hough Pipeline Ablation Harness — Plan 008 Setup"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="v12-default,v12-clean,v5-default",
        help="Comma-separated case names (default: v12-default,v12-clean,v5-default)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "roi_audit", "vote_audit", "sweep_E3", "sweep_E4", "sweep_E5", "sweep_E6"],
        help="Experiment mode (default: baseline)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global seed (default: 42)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/baseline",
        help="Output directory (default: out/baseline)",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip diagnostic plot generation",
    )
    return parser


# ---------------------------------------------------------------------------
# E1 — ROI Audit helpers
# ---------------------------------------------------------------------------


def _compute_finder_centres(metadata: dict) -> dict[str, np.ndarray]:
    """Compute the centre of each finder pattern (module 3.5, 3.5) in image coords."""
    corners = metadata["corners_qr"]
    N = float(metadata["N"])

    TL = np.array(corners["TL"], dtype=np.float64)
    TR = np.array(corners["TR"], dtype=np.float64)
    BR = np.array(corners["BR"], dtype=np.float64)
    BL = np.array(corners["BL"], dtype=np.float64)

    offset = 3.5 / N

    return {
        "TL": TL + offset * (TR - TL) + offset * (BL - TL),
        "TR": TR + offset * (TL - TR) + offset * (BR - TR),
        "BL": BL + offset * (TL - BL) + offset * (BR - BL),
    }


def _assign_clusters_to_finders(
    cluster_centres_xy: list[np.ndarray],
    finder_centres: dict[str, np.ndarray],
) -> dict[int, str | None]:
    """Greedy bipartite assignment of clusters to nearest unassigned finder.

    Returns ``{cluster_idx: finder_name | None}``.
    """
    finder_names = sorted(finder_centres.keys())
    finder_positions = np.array([finder_centres[n] for n in finder_names])

    remaining_finders = set(range(len(finder_names)))
    assignments: dict[int, str | None] = {}

    # Sort clusters by distance to nearest finder (best-fit first)
    scored: list[tuple[int, int, float]] = []
    for ci, cc in enumerate(cluster_centres_xy):
        for fi in range(len(finder_names)):
            dist = float(np.linalg.norm(cc - finder_positions[fi]))
            scored.append((ci, fi, dist))
    scored.sort(key=lambda x: x[2])

    for ci, fi, _dist in scored:
        if ci in assignments:
            continue
        if fi in remaining_finders:
            assignments[ci] = finder_names[fi]
            remaining_finders.discard(fi)

    for ci in range(len(cluster_centres_xy)):
        if ci not in assignments:
            assignments[ci] = None

    return assignments


def _edge_intersects_roi(
    seg: np.ndarray | None, bbox: tuple[int, int, int, int]
) -> bool:
    if seg is None:
        return False
    r0, r1, c0, c1 = bbox
    W = c1 - c0
    H = r1 - r0
    clipped = _clip_segment(seg[0], seg[1], 0.0, float(W - 1), 0.0, float(H - 1))
    return clipped is not None


def _edge_coverage_fraction(
    seg: np.ndarray | None, bbox: tuple[int, int, int, int]
) -> float:
    if seg is None:
        return 0.0
    r0, r1, c0, c1 = bbox
    W = c1 - c0
    H = r1 - r0
    clipped = _clip_segment(seg[0], seg[1], 0.0, float(W - 1), 0.0, float(H - 1))
    if clipped is None:
        return 0.0
    full_len = float(np.linalg.norm(seg[1] - seg[0]))
    if full_len < 1e-12:
        return 0.0
    clip_len = float(np.linalg.norm(clipped[1] - clipped[0]))
    return min(clip_len / full_len, 1.0)


def _run_roi_audit_case(
    case_name: str,
    seed: int,
    H: int,
    W: int,
    config: AugmentationConfig,
    background: np.ndarray,
) -> dict[str, Any]:
    """Run the pipeline and collect E1 ROI audit metrics."""
    import cv2

    rng = np.random.default_rng(seed)
    image, metadata = generate_sample(rng, config, background)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image)

    img_binary = binarize_image(gray)
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return {"case": case_name, "clusters": [], "n_clusters": 0, "finder_assignment": {}}

    clusters = cluster_candidates(rows_valid, cols_valid_all)
    finder_centres = _compute_finder_centres(metadata)

    cluster_centres_xy: list[np.ndarray] = []
    for cl in clusters:
        bbox = cluster_to_bbox(cl, scale=1.5)
        cx = (bbox[2] + bbox[3]) / 2.0
        cy = (bbox[0] + bbox[1]) / 2.0
        cluster_centres_xy.append(np.array([cx, cy], dtype=np.float64))

    assignments = _assign_clusters_to_finders(cluster_centres_xy, finder_centres)

    cluster_metrics: list[dict] = []
    for ci, cl in enumerate(clusters):
        bbox = cluster_to_bbox(cl, scale=1.5)
        r0, r1, c0, c1 = bbox
        roi_centre = np.array([(c0 + c1) / 2.0, (r0 + r1) / 2.0], dtype=np.float64)

        finder_name = assignments.get(ci)
        if finder_name is None:
            centre_error = float("nan")
            n_overlap = 0
            coverage_frac = 0.0
            zero_gt = True
            gt_centre = np.array([float("nan"), float("nan")])
        else:
            gt_centre = finder_centres[finder_name]
            centre_error = float(np.linalg.norm(roi_centre - gt_centre))

            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),
                roi_shape=(r1 - r0, c1 - c0),
            )
            n_overlap = sum(
                1 for e in gt_edges if e["segment"] is not None
            )
            zero_gt = n_overlap == 0

            # Edge coverage fraction: average of per-edge coverage fractions
            # for edges that intersect the ROI
            coverages = [
                _edge_coverage_fraction(e["segment"], bbox)
                for e in gt_edges
                if e["segment"] is not None
            ]
            coverage_frac = float(np.mean(coverages)) if coverages else 0.0

        cluster_metrics.append({
            "case": case_name,
            "cluster_idx": ci,
            "finder": finder_name if finder_name else "unassigned",
            "centre_error_px": centre_error,
            "n_gt_edges_overlap": n_overlap,
            "zero_gt_edges": zero_gt,
            "edge_coverage_frac": coverage_frac,
            "roi_centre_x": float(roi_centre[0]),
            "roi_centre_y": float(roi_centre[1]),
            "gt_centre_x": float(gt_centre[0]),
            "gt_centre_y": float(gt_centre[1]),
        })

    return {
        "case": case_name,
        "clusters": cluster_metrics,
        "n_clusters": len(clusters),
        "finder_assignment": {ci: assignments.get(ci) for ci in range(len(clusters))},
    }


def _write_roi_audit_diagnostics(
    audit: dict, background: np.ndarray, case_metrics: list[dict], out_dir: Path
) -> None:
    case_name = audit["case"]
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # Centre-error scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    errors = [m["centre_error_px"] for m in case_metrics
              if not np.isnan(m["centre_error_px"])]
    labels = [f"C{m['cluster_idx']} ({m['finder']})" for m in case_metrics
              if not np.isnan(m["centre_error_px"])]
    colors = ["green" if m["zero_gt_edges"] else "blue" for m in case_metrics
              if not np.isnan(m["centre_error_px"])]
    if errors:
        ax.bar(range(len(errors)), errors, color=colors, tick_label=labels)
        ax.axhline(y=2.0, color="orange", linestyle="--", label="2 px threshold")
        ax.axhline(y=4.0, color="red", linestyle="--", label="4 px threshold")
        ax.set_ylabel("Centre error (px)")
        ax.set_title(f"ROI Centre Error — {case_name}")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No assigned finders", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(case_dir / "centre_error.png", dpi=150)
    plt.close(fig)

    # Coverage scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    cov = [m["edge_coverage_frac"] for m in case_metrics]
    labels = [f"C{m['cluster_idx']} ({m['finder']})" for m in case_metrics]
    if cov:
        ax.bar(range(len(cov)), cov, tick_label=labels)
        ax.set_ylabel("Avg edge coverage fraction")
        ax.set_title(f"ROI Edge Coverage — {case_name}")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(case_dir / "edge_coverage.png", dpi=150)
    plt.close(fig)

    # Per-cluster ROI overlays (show ROI vs GT finder centre)
    for cm in case_metrics:
        cl_dir = case_dir / f"cluster_{cm['cluster_idx']}"
        cl_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"ROI Overlay — C{cm['cluster_idx']} ({cm['finder']})")
        ax.set_xlim(0, 640)
        ax.set_ylim(640, 0)
        ax.set_aspect("equal")
        ax.text(0.5, 0.95,
                f"centre_error={cm['centre_error_px']:.2f}px  "
                f"coverage={cm['edge_coverage_frac']:.2%}  "
                f"zero_gt={cm['zero_gt_edges']}",
                ha="center", va="top", transform=ax.transAxes, fontsize=9)
        fig.savefig(cl_dir / "roi_context.png", dpi=150)
        plt.close(fig)

    plt.close("all")

    # Per-cluster summary text
    txt_path = case_dir / "audit_summary.txt"
    lines = [f"ROI Audit — {case_name}", f"  Clusters: {audit['n_clusters']}", ""]
    for cm in case_metrics:
        lines.append(
            f"  C{cm['cluster_idx']} → {cm['finder']}: "
            f"centre_err={cm['centre_error_px']:.2f}px  "
            f"n_gt_edges={cm['n_gt_edges_overlap']}  "
            f"coverage={cm['edge_coverage_frac']:.3f}  "
            f"zero_gt={cm['zero_gt_edges']}"
        )
    lines.append("")
    txt_path.write_text("\n".join(lines))


E1_CSV_HEADER = [
    "case",
    "cluster_idx",
    "finder",
    "centre_error_px",
    "n_gt_edges_overlap",
    "zero_gt_edges",
    "edge_coverage_frac",
    "roi_centre_x",
    "roi_centre_y",
    "gt_centre_x",
    "gt_centre_y",
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    case_names = [c.strip() for c in args.cases.split(",")]
    unknown = set(case_names) - set(FIXTURE_SPECS)
    if unknown:
        print(f"Unknown case(s): {unknown}. Known: {sorted(FIXTURE_SPECS)}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = _make_background()

    # ------------------------------------------------------------
    # ROI Audit mode
    # ------------------------------------------------------------
    if args.mode == "roi_audit":
        all_rows: list[dict] = []

        for case_name in case_names:
            config, seed, H, W = FIXTURE_SPECS[case_name]
            print(f"  E1 ROI audit: {case_name} ... ", end="", flush=True)

            audit = _run_roi_audit_case(
                case_name, seed, H, W, config, background
            )
            all_rows.extend(audit["clusters"])

            n_cl = audit["n_clusters"]
            zero_gt = sum(1 for m in audit["clusters"] if m["zero_gt_edges"])
            errors = [m["centre_error_px"] for m in audit["clusters"]
                      if not np.isnan(m["centre_error_px"])]
            p95_err = np.percentile(errors, 95) if errors else float("nan")
            print(
                f"done. clusters={n_cl}, zero_gt={zero_gt}, "
                f"centre_err: p95={p95_err:.2f}px, "
                f"mean={np.mean(errors):.2f}px"
            )

            if not args.no_diagnostics:
                _write_roi_audit_diagnostics(audit, background, audit["clusters"], out_dir)

        # Write CSV
        csv_path = out_dir / "roi_audit.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=E1_CSV_HEADER)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

        print(f"\n  ROI audit CSV written to {csv_path}")

        # Decision rule summary
        print("\n  Decision rule assessment:")
        all_errors = [r["centre_error_px"] for r in all_rows
                      if not np.isnan(r["centre_error_px"])]
        p95 = float(np.percentile(all_errors, 95)) if all_errors else float("nan")
        print(f"    Centre error p95 = {p95:.2f} px")
        if p95 > 4:
            print("    → p95 > 4 px: ROI origin correction warranted before E4/E8")
        elif p95 > 2:
            print("    → 2 < p95 ≤ 4 px: rho-gate experiments in E8 may need wider ranges")
        else:
            print("    → p95 ≤ 2 px: rho-gate experiments in E8 don't need origin correction")

        zero_gt_clusters = [r for r in all_rows if r["zero_gt_edges"]]
        if zero_gt_clusters:
            print(f"    Zero-GT-edge clusters: {len(zero_gt_clusters)}")
            for z in zero_gt_clusters:
                print(f"      {z['case']} C{z['cluster_idx']} ({z['finder']})")
            print("    → These produce B failures as test artefacts; exclude from future tallies")
        else:
            print("    No zero-GT-edge clusters found.")

        return

    # ------------------------------------------------------------
    # Vote-Audit mode (E2)
    # ------------------------------------------------------------
    if args.mode == "vote_audit":
        hough_kwargs: dict = {"theta_step_deg": 2.0, "rho_step": 1.0}

        all_aggregates: list[dict] = []
        detail_rows: list[dict] = []

        for case_name in case_names:
            config, seed, H, W = FIXTURE_SPECS[case_name]
            print(f"  E2 vote audit: {case_name} ... ", end="", flush=True)

            result = run_vote_audit_case(
                case_name, seed, H, W, config, background,
                hough_kwargs=hough_kwargs,
            )

            agg = aggregate_result(result)
            all_aggregates.append(agg)

            D = agg["D"]
            A = agg["A"]
            C = agg["C"]
            B = agg["B"]
            total = D + A + C + B

            # Count classifications per D edge
            d_classifications: list[str] = []
            for cl in result.clusters:
                for i, gt in enumerate(cl.gt_edges):
                    if gt["segment"] is None:
                        continue
                    if i < len(cl.peak_hit) and cl.peak_hit[i]:
                        continue  # HIT — not D
                    dcs = cl.vote_cloud_classifications[i] if i < len(cl.vote_cloud_classifications) else ""
                    dns = cl.vote_cloud_notes[i] if i < len(cl.vote_cloud_notes) else ""
                    d_classifications.append(dcs)
                    detail_rows.append({
                        "case": case_name,
                        "cluster": cl.cluster_idx,
                        "gt_edge": gt["label"],
                        "status": "D",
                        "classification": dcs,
                        "note": dns,
                    })

            # Control edges (non-D): also record
            for cl in result.clusters:
                for i, gt in enumerate(cl.gt_edges):
                    if gt["segment"] is None:
                        continue
                    if i < len(cl.peak_hit) and not cl.peak_hit[i]:
                        continue  # D edge, already recorded
                    dcs = cl.vote_cloud_classifications[i] if i < len(cl.vote_cloud_classifications) else ""
                    dns = cl.vote_cloud_notes[i] if i < len(cl.vote_cloud_notes) else ""
                    detail_rows.append({
                        "case": case_name,
                        "cluster": cl.cluster_idx,
                        "gt_edge": gt["label"],
                        "status": "OK",
                        "classification": dcs,
                        "note": dns,
                    })

            # Classification tally
            from collections import Counter
            d_tally = Counter(d_classifications)
            tally_str = ", ".join(f"{k}={v}" for k, v in d_tally.items())
            print(f"done. D={D} A={A} C={C} B={B} total={total}  D-classes: [{tally_str}]")

            if not args.no_diagnostics:
                case_dir = out_dir / case_name
                _write_diagnostics(result, case_dir)

        # Write summary CSV
        csv_path = out_dir / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in all_aggregates:
                writer.writerow(row)

        print(f"\n  Summary CSV written to {csv_path}")

        # Write per-edge detail CSV
        detail_path = out_dir / "vote_cloud_detail.csv"
        detail_header = ["case", "cluster", "gt_edge", "status", "classification", "note"]
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_header)
            writer.writeheader()
            for row in detail_rows:
                writer.writerow(row)

        print(f"  Per-edge detail CSV written to {detail_path}")

        # Print classification table
        print("\n  Vote-cloud classification summary:")
        d_rows = [r for r in detail_rows if r["status"] == "D"]
        if d_rows:
            print("    D-failure edges:")
            for r in d_rows:
                print(f"      [{r['case']}] C{r['cluster']} {r['gt_edge']}: {r['classification']} — {r['note']}")
        else:
            print("    No D-failure edges found.")

        ok_rows = [r for r in detail_rows if r["status"] == "OK"]
        n_ok = len(ok_rows)
        if n_ok > 0:
            from collections import Counter
            ok_tally = Counter(r["classification"] for r in ok_rows)
            print(f"\n    Control edges (non-D, n={n_ok}):")
            for cls in sorted(ok_tally):
                print(f"      {cls}: {ok_tally[cls]}")

        # Decision rule
        print("\n  Decision rule assessment:")
        all_d_classes = [r["classification"] for r in d_rows]
        if all_d_classes:
            from collections import Counter
            ccount = Counter(all_d_classes)
            print(f"    D classification counts: {dict(ccount)}")
            if any(c in ("theta_spread", "vote_dilution") for c in all_d_classes):
                print("    → theta_spread or vote_dilution detected: soft angular voting (E3) is well-motivated")
            if all(c == "empty" for c in all_d_classes):
                print("    → All D edges are 'empty': problem is upstream of Hough (edge extraction or ROI centering)")
                print("      E3 soft voting won't help D.")
        else:
            print("    No D edges to classify (all peaks hit).")

        return

    # ------------------------------------------------------------
    # E3 — Angular Sweep
    # ------------------------------------------------------------
    if args.mode == "sweep_E3":
        import itertools
        import json as json_mod

        theta_steps = [0.5, 1.0, 2.0, 5.0]
        theta_windows = [0.0, 1.0, 3.0, 6.0]
        vote_schemes = ["onebin", "gaussian", "dot"]

        sweep_cases = ["v12-default", "v12-clean"]
        all_aggregates: list[dict] = []
        all_configs: list[dict] = []

        target_cases = {c for c in case_names if c in sweep_cases}
        n_combos = len(theta_steps) * len(theta_windows) * len(vote_schemes)
        n_total = n_combos * len(target_cases)
        n_done = 0

        for ts in theta_steps:
            for tw in theta_windows:
                for vs in vote_schemes:
                    config_key = f"ts={ts:.1f}_tw={tw:.1f}_{vs}"
                    hough_kw = {
                        "theta_step_deg": ts,
                        "theta_window_deg": tw,
                        "vote_scheme": vs,
                        "rho_step": 1.0,
                        "return_acc": True,
                    }

                    for case_name in target_cases:
                        config, seed, H, W = FIXTURE_SPECS[case_name]
                        n_done += 1
                        print(
                            f"  [{n_done}/{n_total}] E3: {case_name} {config_key} ... ",
                            end="", flush=True,
                        )

                        result = run_case(
                            case_name, config_key, seed, H, W, config,
                            background, hough_kwargs=hough_kw,
                        )

                        # Tally with C3 exclusion for v12-default
                        D, A, C, B = 0, 0, 0, 0
                        peak_hit_ratios: list[float] = []
                        for cl in result.clusters:
                            gt_with_seg = [e for e in cl.gt_edges if e["segment"] is not None]
                            if len(gt_with_seg) == 0:
                                continue  # skip non-finder clusters
                            D += len(cl.failures_D)
                            A += len(cl.failures_A)
                            C += len(cl.failures_C)
                            B += len(cl.failures_B)
                            n_hit = sum(1 for h in cl.peak_hit if h)
                            if len(gt_with_seg) > 0:
                                peak_hit_ratios.append(n_hit / len(gt_with_seg))

                        hit_rate = float(np.mean(peak_hit_ratios)) if peak_hit_ratios else 0.0
                        total = D + A + C + B
                        print(f"done. D={D} A={A} C={C} B={B} total={total} hit_rate={hit_rate:.3f}")

                        agg = {
                            "case": case_name,
                            "config_key": config_key,
                            "D": D,
                            "A": A,
                            "C": C,
                            "B": B,
                            "peak_hit_rate": round(hit_rate, 4),
                            "theta_step_deg": ts,
                            "theta_window_deg": tw,
                            "vote_scheme": vs,
                        }
                        all_aggregates.append(agg)
                        all_configs.append({
                            "case": case_name,
                            "config_key": config_key,
                            "theta_step_deg": ts,
                            "theta_window_deg": tw,
                            "vote_scheme": vs,
                            "D": D, "A": A, "C": C, "B": B,
                            "peak_hit_rate": round(hit_rate, 4),
                        })

        # Write summary CSV
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep_E3.csv"
        sweep_header = [
            "case", "config_key", "theta_step_deg", "theta_window_deg",
            "vote_scheme", "D", "A", "C", "B", "peak_hit_rate",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_header)
            writer.writeheader()
            for row in all_aggregates:
                writer.writerow(row)

        print(f"\n  Sweep CSV written to {csv_path}")

        # Find best config: max peak_hit_rate on v12-default, zero new B on v12-clean
        v12d_rows = [r for r in all_configs if r["case"] == "v12-default"]
        v12c_rows = {r["config_key"]: r for r in all_configs if r["case"] == "v12-clean"}

        # baseline: ts=2.0 tw=0.0 onebin
        baseline_key = "ts=2.0_tw=0.0_onebin"
        baseline_v12d = next((r for r in v12d_rows if r["config_key"] == baseline_key), None)
        baseline_hit = baseline_v12d["peak_hit_rate"] if baseline_v12d else 0.0
        baseline_B = 0  # baseline v12-clean has B=0

        best_row = None
        best_hit = -1.0
        for row in v12d_rows:
            ck = row["config_key"]
            if row["peak_hit_rate"] <= best_hit:
                continue
            if ck in v12c_rows:
                v12c = v12c_rows[ck]
                b_regression = v12c["B"] - baseline_B
                if b_regression > 1:
                    continue  # B regression > +1
            best_hit = row["peak_hit_rate"]
            best_row = row

        print(f"\n  Baseline (ts=2.0 tw=0 onebin) v12-default hit_rate = {baseline_hit:.4f}")
        if best_row:
            print(f"  Best config:  {best_row['config_key']}")
            print(f"    v12-default:  D={best_row['D']} A={best_row['A']} C={best_row['C']} B={best_row['B']} hit_rate={best_row['peak_hit_rate']:.4f}")
            if best_row["config_key"] in v12c_rows:
                bc = v12c_rows[best_row["config_key"]]
                print(f"    v12-clean:    D={bc['D']} A={bc['A']} C={bc['C']} B={bc['B']}")
            if best_hit > baseline_hit:
                print(f"    Improvement: +{best_hit - baseline_hit:.4f} in peak hit rate")
            else:
                print("    No improvement over baseline.")
            best_path = out_dir / "best_config_E3.json"
            best_path.write_text(json_mod.dumps(best_row, indent=2))
            print(f"  Best config saved to {best_path}")
        else:
            print("  No valid config found (all regress B > +1 on v12-clean).")

        return

    # ------------------------------------------------------------
    # E4 — Radial Sweep
    # ------------------------------------------------------------
    if args.mode == "sweep_E4":
        import itertools
        import json as json_mod

        # Best config from E3
        e3_baseline = {"theta_step_deg": 0.5, "theta_window_deg": 0.0, "vote_scheme": "onebin"}

        rho_steps = [0.5, 1.0, 2.0]
        nms_rho_radii = [2, 3, 4, 6]
        nms_theta_radii = [1, 2, 3]
        acc_smooths = ["none", "1x3_triangular", "1x5_triangular"]

        sweep_cases = ["v12-default", "v12-clean"]
        all_aggregates: list[dict] = []

        target_cases = {c for c in case_names if c in sweep_cases}
        n_combos = len(rho_steps) * len(nms_rho_radii) * len(nms_theta_radii) * len(acc_smooths)
        n_total = n_combos * len(target_cases)
        n_done = 0

        for rs in rho_steps:
            for nr in nms_rho_radii:
                for nt in nms_theta_radii:
                    for sm in acc_smooths:
                        config_key = f"rs={rs:.1f}_nr={nr}_nt={nt}_sm={sm}"
                        hough_kw = {
                            **e3_baseline,
                            "rho_step": rs,
                            "nms_radius_rho": nr,
                            "nms_radius_theta": nt,
                            "acc_smooth": sm if sm != "none" else None,
                            "return_acc": True,
                        }

                        for case_name in target_cases:
                            config, seed, H, W = FIXTURE_SPECS[case_name]
                            n_done += 1
                            print(
                                f"  [{n_done}/{n_total}] E4: {case_name} {config_key} ... ",
                                end="", flush=True,
                            )

                            result = run_case(
                                case_name, config_key, seed, H, W, config,
                                background, hough_kwargs=hough_kw,
                            )

                            D, A, C, B = 0, 0, 0, 0
                            peak_hit_ratios: list[float] = []
                            for cl in result.clusters:
                                gt_with_seg = [e for e in cl.gt_edges if e["segment"] is not None]
                                if len(gt_with_seg) == 0:
                                    continue
                                D += len(cl.failures_D)
                                A += len(cl.failures_A)
                                C += len(cl.failures_C)
                                B += len(cl.failures_B)
                                n_hit = sum(1 for h in cl.peak_hit if h)
                                if len(gt_with_seg) > 0:
                                    peak_hit_ratios.append(n_hit / len(gt_with_seg))

                            hit_rate = float(np.mean(peak_hit_ratios)) if peak_hit_ratios else 0.0
                            total = D + A + C + B
                            print(f"done. D={D} A={A} C={C} B={B} total={total} hit_rate={hit_rate:.3f}")

                            all_aggregates.append({
                                "case": case_name,
                                "config_key": config_key,
                                "D": D, "A": A, "C": C, "B": B,
                                "peak_hit_rate": round(hit_rate, 4),
                                "rho_step": rs,
                                "nms_radius_rho": nr,
                                "nms_radius_theta": nt,
                                "acc_smooth": sm,
                            })

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep_E4.csv"
        sweep_header = [
            "case", "config_key", "rho_step", "nms_radius_rho", "nms_radius_theta",
            "acc_smooth", "D", "A", "C", "B", "peak_hit_rate",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_header)
            writer.writeheader()
            for row in all_aggregates:
                writer.writerow(row)

        print(f"\n  Sweep CSV written to {csv_path}")

        # Find best config: lowest D on v12-default, C on v12-clean ≤ baseline+1
        v12d_rows = [r for r in all_aggregates if r["case"] == "v12-default"]
        v12c_rows = {r["config_key"]: r for r in all_aggregates if r["case"] == "v12-clean"}

        # Baseline for v12-clean C count
        baseline_v12c_C = 0

        best_row = None
        best_score = (999, 999, 999, 999)  # (D, A, C, B) lexicographic

        for row in v12d_rows:
            ck = row["config_key"]
            d, a, c, b = row["D"], row["A"], row["C"], row["B"]
            score = (d, a, c, b)

            if ck in v12c_rows:
                v12c = v12c_rows[ck]
                if v12c["C"] > baseline_v12c_C + 1:
                    continue
                if v12c["B"] > 1:
                    continue

            if score < best_score:
                best_score = score
                best_row = row

        if best_row:
            print(f"  Best config:  {best_row['config_key']}")
            print(f"    v12-default:  D={best_row['D']} A={best_row['A']} C={best_row['C']} B={best_row['B']} hit_rate={best_row['peak_hit_rate']:.4f}")
            if best_row["config_key"] in v12c_rows:
                bc = v12c_rows[best_row["config_key"]]
                print(f"    v12-clean:    D={bc['D']} A={bc['A']} C={bc['C']} B={bc['B']}")
            best_path = out_dir / "best_config_E4.json"
            best_path.write_text(json_mod.dumps(best_row, indent=2))
            print(f"  Best config saved to {best_path}")
        else:
            print("  No valid config found.")

        return

    # ------------------------------------------------------------
    # E5 — Edge Continuity (Hysteresis) Sweep
    # ------------------------------------------------------------
    if args.mode == "sweep_E5":
        import itertools
        import json as json_mod

        # Best config from E4
        e4_baseline = {
            "theta_step_deg": 0.5, "theta_window_deg": 0.0, "vote_scheme": "onebin",
            "rho_step": 2.0, "nms_radius_rho": 2, "nms_radius_theta": 2,
            "acc_smooth": None,
        }

        hysteresis_opts = ["none", "lite"]
        high_pcts = [80, 85, 90, 95]
        low_pcts = [60, 65, 70, 75, 80]

        sweep_cases = ["v12-default", "v12-clean"]
        all_aggregates: list[dict] = []

        target_cases = {c for c in case_names if c in sweep_cases}
        n_combos = len(hysteresis_opts) * len(high_pcts) * len(low_pcts)
        n_total = n_combos * len(target_cases)
        n_done = 0

        for hys in hysteresis_opts:
            for hp in high_pcts:
                for lp in low_pcts:
                    if hys == "none":
                        config_key = "hys=none"
                        hough_kw = {
                            "theta_step_deg": 0.5, "theta_window_deg": 0.0,
                            "vote_scheme": "onebin", "rho_step": 2.0,
                            "nms_radius_rho": 2, "nms_radius_theta": 2,
                            "acc_smooth": None, "return_acc": True,
                        }
                    else:
                        if lp >= hp:
                            continue  # skip invalid combos
                        config_key = f"hys=lite_hp={hp}_lp={lp}"
                        hough_kw = {
                            **e4_baseline,
                            "return_acc": True,
                            "_hysteresis": "lite",
                            "_hysteresis_high_pct": float(hp),
                            "_hysteresis_low_pct": float(lp),
                        }

                    for case_name in target_cases:
                        config, seed, H, W = FIXTURE_SPECS[case_name]
                        n_done += 1
                        print(
                            f"  [{n_done}/{n_total}] E5: {case_name} {config_key} ... ",
                            end="", flush=True,
                        )

                        result = run_case(
                            case_name, config_key, seed, H, W, config,
                            background, hough_kwargs=hough_kw,
                        )

                        D, A, C, B = 0, 0, 0, 0
                        peak_hit_ratios: list[float] = []
                        for cl in result.clusters:
                            gt_with_seg = [e for e in cl.gt_edges if e["segment"] is not None]
                            if len(gt_with_seg) == 0:
                                continue
                            D += len(cl.failures_D)
                            A += len(cl.failures_A)
                            C += len(cl.failures_C)
                            B += len(cl.failures_B)
                            n_hit = sum(1 for h in cl.peak_hit if h)
                            if len(gt_with_seg) > 0:
                                peak_hit_ratios.append(n_hit / len(gt_with_seg))

                        hit_rate = float(np.mean(peak_hit_ratios)) if peak_hit_ratios else 0.0
                        total = D + A + C + B
                        print(f"done. D={D} A={A} C={C} B={B} total={total} hit_rate={hit_rate:.3f}")

                        all_aggregates.append({
                            "case": case_name,
                            "config_key": config_key,
                            "D": D, "A": A, "C": C, "B": B,
                            "peak_hit_rate": round(hit_rate, 4),
                            "hysteresis": hys,
                            "high_pct": str(hp) if hys != "none" else "",
                            "low_pct": str(lp) if hys != "none" else "",
                        })

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep_E5.csv"
        sweep_header = [
            "case", "config_key", "hysteresis", "high_pct", "low_pct",
            "D", "A", "C", "B", "peak_hit_rate",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_header)
            writer.writeheader()
            for row in all_aggregates:
                writer.writerow(row)

        print(f"\n  Sweep CSV written to {csv_path}")

        v12d_rows = [r for r in all_aggregates if r["case"] == "v12-default"]
        v12c_rows = {r["config_key"]: r for r in all_aggregates if r["case"] == "v12-clean"}

        best_row = None
        best_score = (999, 999, 999, 999)

        for row in v12d_rows:
            ck = row["config_key"]
            d, a, c, b = row["D"], row["A"], row["C"], row["B"]
            score = (d, a, c, b)

            if ck in v12c_rows:
                v12c = v12c_rows[ck]
                if v12c["B"] > 1:
                    continue

            if score < best_score:
                best_score = score
                best_row = row

        if best_row:
            print(f"  Best config:  {best_row['config_key']}")
            print(f"    v12-default:  D={best_row['D']} A={best_row['A']} C={best_row['C']} B={best_row['B']} hit_rate={best_row['peak_hit_rate']:.4f}")
            if best_row["config_key"] in v12c_rows:
                bc = v12c_rows[best_row["config_key"]]
                print(f"    v12-clean:    D={bc['D']} A={bc['A']} C={bc['C']} B={bc['B']}")
            best_path = out_dir / "best_config_E5.json"
            best_path.write_text(json_mod.dumps(best_row, indent=2))
            print(f"  Best config saved to {best_path}")
        else:
            print("  No valid config found.")

        return

    # ------------------------------------------------------------
    # E6 — Support Sweep (distance_thresh, gap_tolerance, support_dilate)
    # ------------------------------------------------------------
    if args.mode == "sweep_E6":
        import json as json_mod

        # Fixed hough kwargs from E4/E5 best (no hysteresis, best radial config)
        hough_kw: dict = {
            "theta_step_deg": 0.5, "theta_window_deg": 0.0, "vote_scheme": "onebin",
            "rho_step": 2.0, "nms_radius_rho": 2, "nms_radius_theta": 2,
            "acc_smooth": None, "return_acc": True,
        }

        distance_threshs = [1.0, 1.5, 2.0, 3.0]
        gap_tolerances = [1.0, 2.0, 3.0, 4.0]
        support_dilates = [0, 1, 2]

        sweep_cases = ["v12-default", "v12-clean"]
        all_aggregates: list[dict] = []

        target_cases = {c for c in case_names if c in sweep_cases}
        n_combos = len(distance_threshs) * len(gap_tolerances) * len(support_dilates)
        n_total = n_combos * len(target_cases)
        n_done = 0

        for dt in distance_threshs:
            for gt in gap_tolerances:
                for sd in support_dilates:
                    config_key = f"dt={dt:.1f}_gt={gt:.0f}_sd={sd}"
                    refine_kw = {
                        "distance_thresh": dt,
                        "gap_tolerance": gt,
                        "support_dilate": sd,
                    }

                    for case_name in target_cases:
                        config, seed, H, W = FIXTURE_SPECS[case_name]
                        n_done += 1
                        print(
                            f"  [{n_done}/{n_total}] E6: {case_name} {config_key} ... ",
                            end="", flush=True,
                        )

                        result = run_case(
                            case_name, config_key, seed, H, W, config,
                            background, hough_kwargs=hough_kw, refine_kwargs=refine_kw,
                        )

                        D, A, C, B = 0, 0, 0, 0
                        peak_hit_ratios: list[float] = []
                        for cl in result.clusters:
                            gt_with_seg = [e for e in cl.gt_edges if e["segment"] is not None]
                            if len(gt_with_seg) == 0:
                                continue
                            D += len(cl.failures_D)
                            A += len(cl.failures_A)
                            C += len(cl.failures_C)
                            B += len(cl.failures_B)
                            n_hit = sum(1 for h in cl.peak_hit if h)
                            if len(gt_with_seg) > 0:
                                peak_hit_ratios.append(n_hit / len(gt_with_seg))

                        hit_rate = float(np.mean(peak_hit_ratios)) if peak_hit_ratios else 0.0
                        total = D + A + C + B
                        print(f"done. D={D} A={A} C={C} B={B} total={total} hit_rate={hit_rate:.3f}")

                        all_aggregates.append({
                            "case": case_name,
                            "config_key": config_key,
                            "D": D, "A": A, "C": C, "B": B,
                            "peak_hit_rate": round(hit_rate, 4),
                            "distance_thresh": dt,
                            "gap_tolerance": gt,
                            "support_dilate": sd,
                        })

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "sweep_E6.csv"
        sweep_header = [
            "case", "config_key", "distance_thresh", "gap_tolerance", "support_dilate",
            "D", "A", "C", "B", "peak_hit_rate",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_header)
            writer.writeheader()
            for row in all_aggregates:
                writer.writerow(row)

        print(f"\n  Sweep CSV written to {csv_path}")

        v12d_rows = [r for r in all_aggregates if r["case"] == "v12-default"]
        v12c_rows = {r["config_key"]: r for r in all_aggregates if r["case"] == "v12-clean"}

        best_row = None
        best_score = (999, 999, 999, 999)

        for row in v12d_rows:
            ck = row["config_key"]
            d, a, c, b = row["D"], row["A"], row["C"], row["B"]
            score = (d, a, c, b)

            if ck in v12c_rows:
                v12c = v12c_rows[ck]
                if v12c["B"] > 1 or v12c["C"] > 1 or v12c["A"] > 0 or v12c["D"] > 0:
                    continue

            if score < best_score:
                best_score = score
                best_row = row

        if best_row:
            print(f"  Best config:  {best_row['config_key']}")
            print(f"    v12-default:  D={best_row['D']} A={best_row['A']} C={best_row['C']} B={best_row['B']} hit_rate={best_row['peak_hit_rate']:.4f}")
            if best_row["config_key"] in v12c_rows:
                bc = v12c_rows[best_row["config_key"]]
                print(f"    v12-clean:    D={bc['D']} A={bc['A']} C={bc['C']} B={bc['B']}")
            best_path = out_dir / "best_config_E6.json"
            best_path.write_text(json_mod.dumps(best_row, indent=2))
            print(f"  Best config saved to {best_path}")
        else:
            print("  No valid config found (all regress on v12-clean?).")

        return

    # ------------------------------------------------------------
    # Baseline mode (default)
    # ------------------------------------------------------------
    hough_kwargs: dict = {}
    refine_kwargs: dict = {}

    all_aggregates: list[dict] = []

    for case_name in case_names:
        config, seed, H, W = FIXTURE_SPECS[case_name]
        config_key = "baseline"
        print(f"  Running {case_name} ({config_key}) ... ", end="", flush=True)

        result = run_case(
            case_name,
            config_key,
            seed,
            H,
            W,
            config,
            background,
            hough_kwargs=hough_kwargs,
            refine_kwargs=refine_kwargs,
        )

        agg = aggregate_result(result)
        all_aggregates.append(agg)

        D, A, C, B = agg["D"], agg["A"], agg["C"], agg["B"]
        total = D + A + C + B
        print(f"done. D={D} A={A} C={C} B={B} total={total}")

        if not args.no_diagnostics:
            case_dir = out_dir / case_name
            _write_diagnostics(result, case_dir)

    # Write CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in all_aggregates:
            writer.writerow(row)

    print(f"\n  Summary CSV written to {csv_path}")

    # Print tallies
    print("\n  Failure tallies:")
    for row in all_aggregates:
        print(
            f"    {row['case']:>15s}: D={row['D']}, A={row['A']}, C={row['C']}, "
            f"B={row['B']}, total={row['D'] + row['A'] + row['C'] + row['B']}"
        )

    # Gate check
    expected = {
        "v12-default": (2, 2, 4, 5),
        "v12-clean": (0, 0, 0, 0),
        "v5-default": (2, 1, 3, 0),
    }
    all_ok = True
    for row in all_aggregates:
        case = row["case"]
        if case in expected:
            ed, ea, ec, eb = expected[case]
            got = (row["D"], row["A"], row["C"], row["B"])
            if got != (ed, ea, ec, eb):
                print(f"  WARNING: {case} expected D={ed} A={ea} C={ec} B={eb}, got D={got[0]} A={got[1]} C={got[2]} B={got[3]}")
                all_ok = False

    if all_ok:
        print("  Baseline tallies match expected values.")
    else:
        print("  Baseline tallies DO NOT match — check above warnings.")


if __name__ == "__main__":
    main()
