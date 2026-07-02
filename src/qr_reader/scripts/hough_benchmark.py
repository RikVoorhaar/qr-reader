"""Hough Benchmark — standardised per-edge TP/FN/FP with overlap quality.

Generates QR images, runs the Hough pipeline on GT-derived ROI cutouts,
reports granular line- and segment-level metrics, and saves everything
to a structured output directory for cross-experiment comparison.

Usage:
    .venv/bin/python -m qr_reader.scripts.hough_benchmark \\
        --cases v12-default,v12-clean \\
        --n-images 5 \\
        --hough-config e6best \\
        --tag my-experiment \\
        --out out/bench

Output directory structure
    (default: out/bench_{tag}_{timestamp}/):
    ├── config.json             # CLI args + Hough preset
    ├── per_edge.csv            # One row per GT edge (all images)
    ├── per_cluster.csv         # One row per cluster (same as console)
    ├── summary.json            # Aggregate metrics per case
    └── plots/                  # Only with --save-plots
        ├── C{ci}_roi_overlay.png
        ├── C{ci}_hough_accumulator.png
        └── C{ci}_support_strips.png
"""

# %% setup
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.hough import LineSegment, hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

HOUGH_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "e6best": {
        "theta_step_deg": 0.5,
        "rho_step": 1.0,
        "nms_radius_rho": 2,
        "nms_radius_theta": 2,
        "gap_tolerance": 3,
        "distance_thresh": 1.5,
        "support_dilate": 0,
        "vote_scheme": "onebin",
        "theta_window_deg": 0,
        "acc_smooth": None,
    },
    "e6tuned": {
        "theta_step_deg": 0.5,
        "rho_step": 0.5,
        "nms_radius_rho": 5,
        "nms_radius_theta": 5,
        "threshold_rel": 0.15,
        "gap_tolerance": 3,
        "distance_thresh": 1.5,
        "support_dilate": 0,
        "vote_scheme": "onebin",
        "theta_window_deg": 0,
        "acc_smooth": None,
        "max_peaks": 20,
    },
}

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

_INSIDE = 0
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8


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
    oc0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
    oc1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)
    while True:
        if (oc0 | oc1) == 0:
            return np.array([[x0, y0], [x1, y1]], dtype=np.float64)
        if (oc0 & oc1) != 0:
            return None
        oc = oc0 if oc0 != 0 else oc1
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
        if oc == oc0:
            x0, y0 = x, y
            oc0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
        else:
            x1, y1 = x, y
            oc1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)


# ---------------------------------------------------------------------------
# GT edge geometry
# ---------------------------------------------------------------------------


def _normal_angle_deg(normal: np.ndarray) -> float:
    rad = np.arctan2(normal[1], normal[0])
    if rad < 0:
        rad += np.pi
    return float(np.rad2deg(rad))


def _angular_distance_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    dot = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


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


def _compute_finder_edges(
    metadata: dict,
    roi_offset: tuple[int, int] | None = None,
    roi_shape: tuple[int, int] | None = None,
) -> list[dict]:
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
        return project_points(H, np.array([[col, row]], dtype=np.float64))[0]

    finder_positions: dict[str, tuple[int, int]] = {
        "TL": (0, 0),
        "TR": (0, N - 7),
        "BL": (N - 7, 0),
    }
    TOP, BOTTOM = [0, 1, 2], [5, 6, 7]
    LEFT, RIGHT = [0, 1, 2], [5, 6, 7]
    results: list[dict] = []
    for fname, (r0, c0) in finder_positions.items():
        for side, offsets in [("top", TOP), ("bot", BOTTOM)]:
            for k in offsets:
                kv = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k), float(c0 + kv))
                b = _grid_to_image(float(r0 + k), float(c0 + 7 - kv))
                _add_edge(results, fname, side, k, a, b, roi_offset, roi_shape)
        for side, offsets in [("left", LEFT), ("right", RIGHT)]:
            for k in offsets:
                kv = min(k, 7 - k)
                a = _grid_to_image(float(r0 + kv), float(c0 + k))
                b = _grid_to_image(float(r0 + 7 - kv), float(c0 + k))
                _add_edge(results, fname, side, k, a, b, roi_offset, roi_shape)
    return results


# ---------------------------------------------------------------------------
# Cluster-to-finder assignment
# ---------------------------------------------------------------------------


def _compute_gt_finder_centres(metadata: dict) -> dict[str, np.ndarray]:
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
    grid_pts = {
        "TL": (3.5, 3.5),
        "TR": (3.5, N - 3.5),
        "BL": (N - 3.5, 3.5),
    }
    centres = {}
    for name, (r, c) in grid_pts.items():
        centres[name] = project_points(H, np.array([[c, r]], dtype=np.float64))[0]
    return centres


def _assign_cluster(
    cluster: CandidateCluster,
    gt_centres: dict[str, np.ndarray],
    max_dist: float = 50.0,
) -> str | None:
    cx = (cluster.cols[2] + cluster.cols[3]) / 2.0
    cy = cluster.row
    cluster_xy = np.array([cx, cy])
    best_name = None
    best_dist = max_dist
    for name, centre_xy in gt_centres.items():
        d = float(np.linalg.norm(cluster_xy - centre_xy))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _match_peak(
    gt_edge: dict,
    normals: np.ndarray,
    rhos: np.ndarray,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
) -> tuple[int, float, float]:
    best_i = -1
    best_dist = float("inf")
    best_ang = float("inf")
    best_rho = float("inf")
    for i in range(len(normals)):
        ang_dist = _angular_distance_deg(gt_edge["normal"], normals[i])
        rho_dist = abs(gt_edge["rho"] - rhos[i])
        if ang_dist <= angle_tol_deg and rho_dist <= rho_tol:
            score = ang_dist + rho_dist
            if score < best_dist:
                best_dist = score
                best_i = i
                best_ang = ang_dist
                best_rho = rho_dist
    return best_i, best_ang, best_rho


# ---------------------------------------------------------------------------
# 1-D segment IoU
# ---------------------------------------------------------------------------


def _segment_iou_1d(gt_seg: np.ndarray, det_seg: np.ndarray) -> dict[str, float]:
    direction = gt_seg[1] - gt_seg[0]
    span_gt = float(np.linalg.norm(direction))
    if span_gt < 1e-12:
        return {
            "iou": 0.0,
            "coverage_gt": 0.0,
            "coverage_seg": 0.0,
            "endpoint_err": float("inf"),
            "lateral_err": float("inf"),
        }
    direction /= span_gt
    t_gt = np.sort(gt_seg @ direction)
    t_det = np.sort(det_seg @ direction)
    lo = max(t_gt[0], t_det[0])
    hi = min(t_gt[1], t_det[1])
    inter = max(0.0, hi - lo)
    union = max(t_gt[1], t_det[1]) - min(t_gt[0], t_det[0])
    iou = inter / union if union > 0 else 0.0
    cov_gt = inter / span_gt if span_gt > 0 else 0.0
    seg_span = t_det[1] - t_det[0]
    cov_seg = inter / seg_span if seg_span > 0 else 0.0
    n_gt = np.array([-direction[1], direction[0]])
    lateral_err = float(np.mean(np.abs(det_seg @ n_gt - gt_seg[0] @ n_gt)))
    dists = np.linalg.norm(det_seg[:, None] - gt_seg[None, :], axis=-1)
    endpoint_err = float(np.max(np.min(dists, axis=1)))
    return {
        "iou": iou,
        "coverage_gt": cov_gt,
        "coverage_seg": cov_seg,
        "endpoint_err": endpoint_err,
        "lateral_err": lateral_err,
    }


# ---------------------------------------------------------------------------
# Peak- and segment-level classification
# ---------------------------------------------------------------------------


@dataclass
class PeakMetrics:
    n_tp: int = 0
    n_fn: int = 0
    n_fp: int = 0
    n_fp_near: int = 0
    ang_errs: list[float] = field(default_factory=list)
    rho_errs: list[float] = field(default_factory=list)
    gt_bin_snrrs: list[float] = field(default_factory=list)
    per_gt_info: list[dict] = field(default_factory=list)


def _classify_peaks(
    gt_edges: list[dict],
    normals: np.ndarray,
    rhos: np.ndarray,
    scores: np.ndarray,
    acc: np.ndarray | None,
    theta_step_rad: float,
    n_theta: int,
    n_rho: int,
    rho_step: float,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
    angular_match_deg: float = 12.0,
) -> PeakMetrics:
    pm = PeakMetrics()
    used_peaks = set()
    gt_vis = [ge for ge in gt_edges if ge["segment"] is not None]
    for ge in gt_vis:
        mi, ang_err, rho_err = _match_peak(ge, normals, rhos, angle_tol_deg, rho_tol)
        pm.per_gt_info.append(
            {
                "label": ge["label"],
                "matched": mi >= 0,
                "peak_idx": mi,
                "ang_err_deg": ang_err,
                "rho_err_px": rho_err,
            }
        )
        if mi >= 0:
            pm.n_tp += 1
            pm.ang_errs.append(ang_err)
            pm.rho_errs.append(rho_err)
            used_peaks.add(mi)
        else:
            pm.n_fn += 1
    n_gt_normals = len(gt_vis)
    if n_gt_normals > 0:
        all_gt_normals = np.array([ge["normal"] for ge in gt_vis])
        for i in range(len(normals)):
            if i in used_peaks:
                continue
            min_ang = min(
                _angular_distance_deg(normals[i], gn) for gn in all_gt_normals
            )
            if min_ang < angular_match_deg:
                pm.n_fp_near += 1
            else:
                pm.n_fp += 1
    if acc is not None:
        for ge_idx, ge in enumerate(gt_vis):
            if len(pm.gt_bin_snrrs) < n_gt_normals:
                theta_rad = np.arctan2(ge["normal"][1], ge["normal"][0])
                if theta_rad < 0:
                    theta_rad += np.pi
                ti = int(round(theta_rad / theta_step_rad)) % n_theta
                ri = int(round(ge["rho"] / rho_step))
                ri = np.clip(ri, 0, n_rho - 1)
                gt_val = float(acc[ti, ri])
                dtheta = max(1, int(round(np.deg2rad(5.0) / theta_step_rad)))
                window_vals = []
                for dt in range(-dtheta, dtheta + 1):
                    tt = (ti + dt) % n_theta
                    window_vals.extend(acc[tt, :].tolist())
                bg_mean = (sum(window_vals) - gt_val) / max(
                    1, len(window_vals) - 1
                )
                snrr = gt_val / bg_mean if bg_mean > 0 else 0.0
                pm.gt_bin_snrrs.append(snrr)
    return pm


@dataclass
class SegMetrics:
    n_tp: int = 0
    n_fn: int = 0
    n_fp: int = 0
    n_fp_near: int = 0
    ious: list[float] = field(default_factory=list)
    coverage_gts: list[float] = field(default_factory=list)
    coverage_segs: list[float] = field(default_factory=list)
    endpoint_errs: list[float] = field(default_factory=list)
    lateral_errs: list[float] = field(default_factory=list)
    per_gt_info: list[dict] = field(default_factory=list)
    fp_chars: list[dict] = field(default_factory=list)


def _classify_segments(
    gt_edges: list[dict],
    normals: np.ndarray,
    rhos: np.ndarray,
    segments: list[LineSegment],
    nms: np.ndarray,
    angle: np.ndarray,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
    iou_thresh: float = 0.3,
    angular_match_deg: float = 12.0,
    gap_tolerance: float = 2.0,
    distance_thresh: float = 1.5,
    support_dilate: int = 0,
) -> SegMetrics:
    sm = SegMetrics()
    gt_vis = [(i, ge) for i, ge in enumerate(gt_edges) if ge["segment"] is not None]
    n_gt_vis = len(gt_vis)
    if n_gt_vis == 0:
        return sm
    used_peaks = set()
    refined = {}
    for i in range(len(normals)):
        seg = refine_line(
            normals[i],
            rhos[i],
            0.0,
            nms,
            angle,
            gap_tolerance=gap_tolerance,
            distance_thresh=distance_thresh,
            support_dilate=support_dilate,
        )
        refined[i] = seg
    for gt_idx, ge in gt_vis:
        mi, _, _ = _match_peak(ge, normals, rhos, angle_tol_deg, rho_tol)
        if mi < 0:
            sm.n_fn += 1
            sm.per_gt_info.append(
                {"label": ge["label"], "tp": False, "reason": "no_peak", "iou": 0.0}
            )
            continue
        seg = refined[mi]
        if np.allclose(seg.endpoints, 0):
            sm.n_fn += 1
            sm.per_gt_info.append(
                {
                    "label": ge["label"],
                    "tp": False,
                    "reason": "degenerate",
                    "iou": 0.0,
                }
            )
            continue
        iou_info = _segment_iou_1d(ge["segment"], seg.endpoints)
        if iou_info["iou"] >= iou_thresh:
            sm.n_tp += 1
            used_peaks.add(mi)
            sm.ious.append(iou_info["iou"])
            sm.coverage_gts.append(iou_info["coverage_gt"])
            sm.coverage_segs.append(iou_info["coverage_seg"])
            sm.endpoint_errs.append(iou_info["endpoint_err"])
            sm.lateral_errs.append(iou_info["lateral_err"])
            sm.per_gt_info.append(
                {
                    "label": ge["label"],
                    "tp": True,
                    "reason": "ok",
                    "iou": iou_info["iou"],
                }
            )
        else:
            sm.n_fn += 1
            sm.per_gt_info.append(
                {
                    "label": ge["label"],
                    "tp": False,
                    "reason": f"low_iou_{iou_info['iou']:.2f}",
                    "iou": iou_info["iou"],
                }
            )
    all_gt_normals = np.array([ge["normal"] for _, ge in gt_vis])
    for i, seg in refined.items():
        if i in used_peaks:
            continue
        if np.allclose(seg.endpoints, 0):
            continue
        min_ang = min(
            _angular_distance_deg(normals[i], gn) for gn in all_gt_normals
        )
        is_near = min_ang < angular_match_deg
        if is_near:
            sm.n_fp_near += 1
        else:
            sm.n_fp += 1
        support_pts = np.argwhere(
            (nms > 0)
            & (
                np.abs(
                    nms
                    * np.sin(
                        angle
                        - np.arctan2(normals[i][1], normals[i][0])
                    )
                )
                < distance_thresh
            )
        )
        mean_str = (
            float(np.mean(nms[support_pts[:, 0], support_pts[:, 1]]))
            if len(support_pts) > 0
            else 0.0
        )
        sm.fp_chars.append(
            {
                "peak_idx": i,
                "is_near": is_near,
                "support_len": float(
                    np.linalg.norm(seg.endpoints[1] - seg.endpoints[0])
                ),
                "mean_nms_strength": mean_str,
            }
        )
    return sm


# ---------------------------------------------------------------------------
# Per-cluster runner
# ---------------------------------------------------------------------------


@dataclass
class ClusterBenchResult:
    cluster_idx: int
    finder_name: str | None
    n_gt_edges: int
    gt_edges: list[dict]
    normals: np.ndarray
    rhos: np.ndarray
    scores: np.ndarray
    segments: list[LineSegment]
    peak_metrics: PeakMetrics
    seg_metrics: SegMetrics
    roi_shape: tuple[int, int]
    runtime_ms: float
    theta_step_rad: float = 0.0
    rho_step: float = 1.0


def _run_cluster(
    roi: np.ndarray,
    nms: np.ndarray,
    angle: np.ndarray,
    bbox: tuple[int, int, int, int],
    ci: int,
    metadata: dict,
    gt_centres: dict[str, np.ndarray],
    hough_kwargs: dict[str, Any],
    refine_kwargs: dict[str, Any],
    cluster: CandidateCluster,
) -> ClusterBenchResult:
    t0 = time.perf_counter()
    finder = _assign_cluster(cluster, gt_centres)
    gt_edges = _compute_finder_edges(
        metadata, roi_offset=(bbox[0], bbox[2]), roi_shape=roi.shape
    )
    gt_vis = [ge for ge in gt_edges if ge["segment"] is not None]
    result = hough_vote_peaks(nms, angle, return_acc=True, **hough_kwargs)
    if len(result) == 4:
        normals, rhos, scores, acc_data = result
    else:
        normals, rhos, scores = result
        acc_data = None
    acc = acc_data["acc"] if acc_data is not None else np.zeros((1, 1))
    theta_step_rad = acc_data["theta_step_rad"] if acc_data else np.deg2rad(2.0)
    n_theta = acc_data["n_theta"] if acc_data else 1
    n_rho = acc_data["n_rho"] if acc_data else 1
    rho_step = hough_kwargs.get("rho_step", 1.0)
    pm = _classify_peaks(
        gt_edges, normals, rhos, scores, acc, theta_step_rad, n_theta, n_rho, rho_step
    )
    segments = [
        refine_line(normals[i], rhos[i], scores[i], nms, angle, **refine_kwargs)
        for i in range(len(normals))
    ]
    sm = _classify_segments(gt_edges, normals, rhos, segments, nms, angle, **refine_kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return ClusterBenchResult(
        cluster_idx=ci,
        finder_name=finder,
        n_gt_edges=len(gt_vis),
        gt_edges=gt_edges,
        normals=normals,
        rhos=rhos,
        scores=scores,
        segments=segments,
        peak_metrics=pm,
        seg_metrics=sm,
        roi_shape=roi.shape,
        runtime_ms=elapsed,
        theta_step_rad=theta_step_rad,
        rho_step=rho_step,
    )


# ---------------------------------------------------------------------------
# Per-image runner
# ---------------------------------------------------------------------------


@dataclass
class ImageBenchResult:
    case_name: str
    seed: int
    clusters: list[ClusterBenchResult]
    runtime_ms: float


def _run_image(
    case_name: str,
    seed: int,
    config: AugmentationConfig,
    H: int,
    W: int,
    bg: np.ndarray,
    hough_kwargs: dict[str, Any],
    refine_kwargs: dict[str, Any],
) -> ImageBenchResult:
    rng = np.random.default_rng(seed)
    image, metadata = generate_sample(rng, config, bg)
    t0 = time.perf_counter()
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else np.asarray(image)
    img_binary = binarize_image(gray)
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return ImageBenchResult(case_name=case_name, seed=seed, clusters=[], runtime_ms=0.0)
    clusters = cluster_candidates(rows_valid, cols_valid_all)
    roi_results = []
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0:
            continue
        nms, angle_arr = extract_thin_edges(roi, blur_sigma=1.0)
        roi_results.append((roi, nms, angle_arr, bbox, ci))
    gt_centres = _compute_gt_finder_centres(metadata)
    cluster_results = []
    for (roi, nms_arr, angle_arr, bbox, ci), cluster in zip(roi_results, clusters):
        cr = _run_cluster(
            roi, nms_arr, angle_arr, bbox, ci, metadata,
            gt_centres, hough_kwargs, refine_kwargs, cluster,
        )
        cluster_results.append(cr)
    elapsed = (time.perf_counter() - t0) * 1000
    return ImageBenchResult(
        case_name=case_name, seed=seed, clusters=cluster_results, runtime_ms=elapsed
    )


# ---------------------------------------------------------------------------
# Breakdown helpers
# ---------------------------------------------------------------------------

_SIDES = {"top": "top", "bot": "bot", "left": "left", "right": "right"}
_FINDERS = {"TL": "TL", "TR": "TR", "BL": "BL"}

# k-group labels: 0&7 = outer boundary, 1&6 = ring transition, 2&5 = centre square
_K_GROUPS = {0: "outer", 1: "inner-ring", 2: "inner-centre",
             5: "inner-centre", 6: "inner-ring", 7: "outer"}


def _parse_gt_label(label: str) -> dict:
    """Parse 'TL_top0' -> {finder, side, k, k_group}."""
    finder = label[:2]
    rest = label[3:]  # e.g. "top0"
    side = rest.rstrip("01234567")
    k = int(rest[len(side):])
    return {"finder": finder, "side": side, "k": k,
            "k_group": _K_GROUPS.get(k, "other")}


def _build_breakdown(
    all_results: list[ImageBenchResult],
) -> dict[str, Any]:
    """Aggregate metrics per finder / side / k-group."""
    keys = ["finder", "side", "k_group"]
    breakdown: dict[str, dict] = {}
    for key in keys:
        groups: dict[str, dict] = {}
        for ir in all_results:
            for cr in ir.clusters:
                pm = cr.peak_metrics
                sm = cr.seg_metrics
                for pi in pm.per_gt_info:
                    label = pi["label"]
                    parsed = _parse_gt_label(label)
                    g = parsed[key]
                    if g not in groups:
                        groups[g] = {"tp": 0, "fn": 0, "total": 0,
                                      "ious": [], "endpoint_errs": []}
                    groups[g]["total"] += 1
                    if pi["matched"]:
                        groups[g]["tp"] += 1
                    else:
                        groups[g]["fn"] += 1
                for si in sm.per_gt_info:
                    label = si["label"]
                    parsed = _parse_gt_label(label)
                    g = parsed[key]
                    if si["tp"]:
                        groups[g]["ious"].append(si["iou"])
        for g, data in groups.items():
            n = data["total"]
            tp = data["tp"]
            fn = data["fn"]
            recall = tp / n if n > 0 else 0.0
            mean_iou = float(np.mean(data["ious"])) if data["ious"] else 0.0
            p05_iou = (
                float(np.percentile(data["ious"], 5))
                if len(data["ious"]) >= 2
                else (data["ious"][0] if data["ious"] else 0.0)
            )
            groups[g] = {
                "total": n, "tp": tp, "fn": fn, "recall": round(recall, 4),
                "mean_iou": round(mean_iou, 4), "p05_iou": round(p05_iou, 4),
            }
        breakdown[key] = groups
    return breakdown


def _format_breakdown(breakdown: dict[str, dict], label: str) -> str:
    lines = [f"  Breakdown by {label}:"]
    groups = breakdown.get(label, {})
    if not groups:
        lines.append("    (no data)")
        return "\n".join(lines)
    for g in sorted(groups):
        d = groups[g]
        lines.append(
            f"    {g:>12}: recall={d['recall']:.3f}  "
            f"TP={d['tp']}/{d['total']}  "
            f"mean_iou={d['mean_iou']:.3f}  p05_iou={d['p05_iou']:.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_roi_overlay(cluster: ClusterBenchResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    # Reconstruct roi from our data... we don't store it anymore.
    # For now, just show a text summary in the plot.
    pm, sm = cluster.peak_metrics, cluster.seg_metrics
    n_gt = cluster.n_gt_edges
    peak_fp_total = pm.n_tp + pm.n_fp + pm.n_fp_near
    seg_fp_total = sm.n_tp + sm.n_fp + sm.n_fp_near
    text = (
        f"C{cluster.cluster_idx} ({cluster.finder_name or '?'}): {n_gt} GT edges\n"
        f"Peaks:   TP={pm.n_tp} FN={pm.n_fn} FP={pm.n_fp + pm.n_fp_near} "
        f"(near={pm.n_fp_near})\n"
        f"Segments: TP={sm.n_tp} FN={sm.n_fn} FP={sm.n_fp + sm.n_fp_near} "
        f"(near={sm.n_fp_near})\n"
        f"IoU: mean={np.mean(sm.ious):.3f}" if sm.ious else ""
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=12, family="monospace")
    ax.set_title(f"C{cluster.cluster_idx} roi={cluster.roi_shape}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_hough_accumulator(cluster: ClusterBenchResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    pm = cluster.peak_metrics
    text = (
        f"Peaks: TP={pm.n_tp} FN={pm.n_fn} FP={pm.n_fp + pm.n_fp_near} "
        f"(near={pm.n_fp_near})\n"
        f"theta_step={np.rad2deg(cluster.theta_step_rad):.2f}°  "
        f"rho_step={cluster.rho_step:.1f}px"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=12, family="monospace")
    ax.set_title(f"Hough accumulator — C{cluster.cluster_idx}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _plot_support_strips(cluster: ClusterBenchResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Support strips (not available in batch mode)",
            transform=ax.transAxes, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _save_plots(out_dir: str, all_results: list[ImageBenchResult]) -> None:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for ir in all_results:
        for cr in ir.clusters:
            prefix = f"C{cr.cluster_idx}"
            fig = _plot_roi_overlay(cr)
            fig.savefig(os.path.join(plots_dir, f"{prefix}_roi_overlay.png"), dpi=100)
            plt.close(fig)
            fig = _plot_hough_accumulator(cr)
            fig.savefig(os.path.join(plots_dir, f"{prefix}_hough_accumulator.png"), dpi=100)
            plt.close(fig)
    print(f"  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------


def _print_image_report(result: ImageBenchResult) -> None:
    print(f"\n=== {result.case_name} seed={result.seed} ===")
    for cr in result.clusters:
        pm, sm = cr.peak_metrics, cr.seg_metrics
        n_gt = cr.n_gt_edges
        peak_fp_total = pm.n_fp + pm.n_fp_near
        seg_fp_total = sm.n_fp + sm.n_fp_near
        prec_p = (
            pm.n_tp / (pm.n_tp + peak_fp_total) if (pm.n_tp + peak_fp_total) > 0 else 0
        )
        rec_p = pm.n_tp / (pm.n_tp + pm.n_fn) if (pm.n_tp + pm.n_fn) > 0 else 0
        prec_s = (
            sm.n_tp / (sm.n_tp + seg_fp_total) if (sm.n_tp + seg_fp_total) > 0 else 0
        )
        rec_s = sm.n_tp / (sm.n_tp + sm.n_fn) if (sm.n_tp + sm.n_fn) > 0 else 0
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
        f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
        mean_iou = np.mean(sm.ious) if sm.ious else 0
        p05_iou = (
            np.percentile(sm.ious, 5)
            if len(sm.ious) >= 2
            else (sm.ious[0] if sm.ious else 0)
        )
        mean_ep = np.mean(sm.endpoint_errs) if sm.endpoint_errs else 0
        mean_ae = np.mean(pm.ang_errs) if pm.ang_errs else 0
        mean_re = np.mean(pm.rho_errs) if pm.rho_errs else 0
        print(f"  C{cr.cluster_idx} ({cr.finder_name or '?'}): {n_gt} GT edges")
        print(
            f"    Peaks:   TP={pm.n_tp} FN={pm.n_fn} FP={peak_fp_total} "
            f"(near={pm.n_fp_near}) prec={prec_p:.2f} rec={rec_p:.2f} F1={f1_p:.2f})"
        )
        print(
            f"    Segments: TP={sm.n_tp} FN={sm.n_fn} FP={seg_fp_total} "
            f"(near={sm.n_fp_near}) prec={prec_s:.2f} rec={rec_s:.2f} F1={f1_s:.2f})"
        )
        print(
            f"    IoU: mean={mean_iou:.3f} p05={p05_iou:.3f}  "
            f"endpoint_err={mean_ep:.2f}px  "
            f"angle_err={mean_ae:.2f}°  rho_err={mean_re:.2f}px"
        )


def _print_summary(
    all_results: list[ImageBenchResult], breakdown: dict[str, dict]
) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    cases = set(r.case_name for r in all_results)
    for case in sorted(cases):
        case_results = [r for r in all_results if r.case_name == case]
        n_seeds = len(case_results)
        all_pm = [pm for r in case_results for cr in r.clusters for pm in [cr.peak_metrics]]
        all_sm = [sm for r in case_results for cr in r.clusters for sm in [cr.seg_metrics]]
        total_peak_tp = sum(pm.n_tp for pm in all_pm)
        total_peak_fn = sum(pm.n_fn for pm in all_pm)
        total_peak_fp = sum(pm.n_fp + pm.n_fp_near for pm in all_pm)
        total_peak_near = sum(pm.n_fp_near for pm in all_pm)
        total_seg_tp = sum(sm.n_tp for sm in all_sm)
        total_seg_fn = sum(sm.n_fn for sm in all_sm)
        total_seg_fp = sum(sm.n_fp + sm.n_fp_near for sm in all_sm)
        total_seg_near = sum(sm.n_fp_near for sm in all_sm)
        prec_p = (
            total_peak_tp / (total_peak_tp + total_peak_fp)
            if (total_peak_tp + total_peak_fp) > 0
            else 0
        )
        rec_p = (
            total_peak_tp / (total_peak_tp + total_peak_fn)
            if (total_peak_tp + total_peak_fn) > 0
            else 0
        )
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
        prec_s = (
            total_seg_tp / (total_seg_tp + total_seg_fp)
            if (total_seg_tp + total_seg_fp) > 0
            else 0
        )
        rec_s = (
            total_seg_tp / (total_seg_tp + total_seg_fn)
            if (total_seg_tp + total_seg_fn) > 0
            else 0
        )
        f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
        all_ious = [iou for sm in all_sm for iou in sm.ious]
        mean_iou = np.mean(all_ious) if all_ious else 0
        p05_iou = (
            np.percentile(all_ious, 5)
            if len(all_ious) >= 2
            else (all_ious[0] if all_ious else 0)
        )
        all_ae = [ae for pm in all_pm for ae in pm.ang_errs]
        all_re = [re for pm in all_pm for re in pm.rho_errs]
        mean_ae = np.mean(all_ae) if all_ae else 0
        mean_re = np.mean(all_re) if all_re else 0
        print(f"\n{case} ({n_seeds} seeds):")
        print(
            f"  Peaks:     TP={total_peak_tp} FN={total_peak_fn} FP={total_peak_fp} "
            f"(near={total_peak_near})  prec={prec_p:.3f} rec={rec_p:.3f} F1={f1_p:.3f}"
        )
        print(
            f"  Segments:  TP={total_seg_tp} FN={total_seg_fn} FP={total_seg_fp} "
            f"(near={total_seg_near})  prec={prec_s:.3f} rec={rec_s:.3f} F1={f1_s:.3f}"
        )
        print(f"  IoU:       mean={mean_iou:.3f} p05={p05_iou:.3f}")
        print(f"  Errors:    angle={mean_ae:.3f}°  rho={mean_re:.3f}px")
    print("\n" + _format_breakdown(breakdown, "finder"))
    print("")
    print(_format_breakdown(breakdown, "side"))
    print("")
    print(_format_breakdown(breakdown, "k_group"))


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

CLUSTER_CSV_HEADER = [
    "case", "seed", "cluster_idx", "finder",
    "n_gt_edges",
    "peak_TP", "peak_FN", "peak_FP", "peak_FP_near",
    "peak_precision", "peak_recall", "peak_F1",
    "seg_TP", "seg_FN", "seg_FP", "seg_FP_near",
    "seg_precision", "seg_recall", "seg_F1",
    "mean_iou_1d", "p05_iou_1d",
    "mean_coverage_gt", "mean_endpoint_err", "mean_lateral_err",
    "mean_ang_err", "mean_rho_err", "n_fp_near_peak",
    "n_fp_non_finder", "runtime_ms",
]

EDGE_CSV_HEADER = [
    "case", "seed", "cluster_idx", "finder",
    "label", "finder_name", "side", "k", "k_group",
    "peak_hit", "ang_err_deg", "rho_err_px",
    "seg_tp", "seg_iou_1d", "seg_reason",
]


def _write_cluster_csv(path: str, all_results: list[ImageBenchResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CLUSTER_CSV_HEADER)
        for ir in all_results:
            for cr in ir.clusters:
                pm, sm = cr.peak_metrics, cr.seg_metrics
                pf = pm.n_fp + pm.n_fp_near
                sf = sm.n_fp + sm.n_fp_near
                prec_p = pm.n_tp / (pm.n_tp + pf) if (pm.n_tp + pf) > 0 else 0
                rec_p = pm.n_tp / (pm.n_tp + pm.n_fn) if (pm.n_tp + pm.n_fn) > 0 else 0
                f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
                prec_s = sm.n_tp / (sm.n_tp + sf) if (sm.n_tp + sf) > 0 else 0
                rec_s = sm.n_tp / (sm.n_tp + sm.n_fn) if (sm.n_tp + sm.n_fn) > 0 else 0
                f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0
                mean_iou = float(np.mean(sm.ious)) if sm.ious else 0.0
                p05_iou = (
                    float(np.percentile(sm.ious, 5))
                    if len(sm.ious) >= 2
                    else (sm.ious[0] if sm.ious else 0.0)
                )
                mean_cov = float(np.mean(sm.coverage_gts)) if sm.coverage_gts else 0.0
                mean_ep = float(np.mean(sm.endpoint_errs)) if sm.endpoint_errs else 0.0
                mean_lat = float(np.mean(sm.lateral_errs)) if sm.lateral_errs else 0.0
                mean_ae = float(np.mean(pm.ang_errs)) if pm.ang_errs else 0.0
                mean_re = float(np.mean(pm.rho_errs)) if pm.rho_errs else 0.0
                nfn = sm.n_fp + sm.n_fp_near if cr.finder_name is None else 0
                w.writerow([
                    ir.case_name, ir.seed, cr.cluster_idx, cr.finder_name or "NON_FINDER",
                    cr.n_gt_edges,
                    pm.n_tp, pm.n_fn, pm.n_fp, pm.n_fp_near,
                    f"{prec_p:.4f}", f"{rec_p:.4f}", f"{f1_p:.4f}",
                    sm.n_tp, sm.n_fn, sm.n_fp, sm.n_fp_near,
                    f"{prec_s:.4f}", f"{rec_s:.4f}", f"{f1_s:.4f}",
                    f"{mean_iou:.4f}", f"{p05_iou:.4f}",
                    f"{mean_cov:.4f}", f"{mean_ep:.4f}", f"{mean_lat:.4f}",
                    f"{mean_ae:.4f}", f"{mean_re:.4f}",
                    pm.n_fp_near, nfn,
                    f"{cr.runtime_ms:.2f}",
                ])


def _write_edge_csv(path: str, all_results: list[ImageBenchResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(EDGE_CSV_HEADER)
        for ir in all_results:
            for cr in ir.clusters:
                finder = cr.finder_name or "NON_FINDER"
                pm = cr.peak_metrics
                sm = cr.seg_metrics
                for gi, pi in enumerate(pm.per_gt_info):
                    label = pi["label"]
                    parsed = _parse_gt_label(label)
                    pe = pi.get("ang_err_deg", float("nan"))
                    pr = pi.get("rho_err_px", float("nan"))
                    si = sm.per_gt_info[gi] if gi < len(sm.per_gt_info) else {}
                    seg_tp = 1 if si.get("tp") else 0
                    seg_iou = si.get("iou", 0.0)
                    seg_reason = si.get("reason", "")
                    w.writerow([
                        ir.case_name, ir.seed, cr.cluster_idx, finder,
                        label,
                        parsed["finder"], parsed["side"], parsed["k"],
                        parsed["k_group"],
                        1 if pi["matched"] else 0,
                        f"{pe:.4f}" if isinstance(pe, float) and not np.isnan(pe) else "",
                        f"{pr:.4f}" if isinstance(pr, float) and not np.isnan(pr) else "",
                        seg_tp,
                        f"{seg_iou:.4f}",
                        seg_reason,
                    ])


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def _write_json_summary(
    path: str,
    all_results: list[ImageBenchResult],
    breakdown: dict,
    args: argparse.Namespace,
) -> None:
    cases = set(r.case_name for r in all_results)
    summary: dict[str, Any] = {"config": vars(args), "cases": {}}
    for case in sorted(cases):
        case_results = [r for r in all_results if r.case_name == case]
        all_pm = [pm for r in case_results for cr in r.clusters for pm in [cr.peak_metrics]]
        all_sm = [sm for r in case_results for cr in r.clusters for sm in [cr.seg_metrics]]
        total_fn = sum(pm.n_fn for pm in all_pm)
        seg_fn = sum(sm.n_fn for sm in all_sm)
        peak_fp = sum(pm.n_fp + pm.n_fp_near for pm in all_pm)
        seg_fp = sum(sm.n_fp + sm.n_fp_near for sm in all_sm)
        peak_tp = sum(pm.n_tp for pm in all_pm)
        seg_tp = sum(sm.n_tp for sm in all_sm)
        all_ious = [iou for sm in all_sm for iou in sm.ious]
        all_ae = [ae for pm in all_pm for ae in pm.ang_errs]
        all_re = [re for pm in all_pm for re in pm.rho_errs]
        summary["cases"][case] = {
            "n_seeds": len(case_results),
            "peaks": {
                "tp": int(peak_tp), "fn": int(total_fn), "fp": int(peak_fp),
                "fp_near": int(sum(pm.n_fp_near for pm in all_pm)),
                "recall": round((peak_tp / (peak_tp + total_fn)) if (peak_tp + total_fn) > 0 else 0, 4),
                "precision": round((peak_tp / (peak_tp + peak_fp)) if (peak_tp + peak_fp) > 0 else 0, 4),
                "mean_ang_err": round(float(np.mean(all_ae)) if all_ae else 0, 4),
                "mean_rho_err": round(float(np.mean(all_re)) if all_re else 0, 4),
            },
            "segments": {
                "tp": int(seg_tp), "fn": int(seg_fn), "fp": int(seg_fp),
                "fp_near": int(sum(sm.n_fp_near for sm in all_sm)),
                "recall": round((seg_tp / (seg_tp + seg_fn)) if (seg_tp + seg_fn) > 0 else 0, 4),
                "precision": round((seg_tp / (seg_tp + seg_fp)) if (seg_tp + seg_fp) > 0 else 0, 4),
                "mean_iou": round(float(np.mean(all_ious)) if all_ious else 0, 4),
                "p05_iou": round(float(np.percentile(all_ious, 5)) if len(all_ious) >= 2 else (all_ious[0] if all_ious else 0), 4),
            },
        }
        bd = breakdown.get("finder", {})
        for f in ["TL", "TR", "BL"]:
            if f in bd:
                summary["cases"][case][f"recall_{f}"] = bd[f]["recall"]
        summary["cases"][case]["breakdown"] = breakdown
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hough Benchmark — standardised per-edge TP/FN/FP evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cases", type=str,
        default="v12-clean,v12-default,v5-default",
        help="Comma-separated case names (default: v12-clean,v12-default,v5-default)",
    )
    p.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed values (overrides per-case defaults)",
    )
    p.add_argument(
        "--n-images", type=int, default=5,
        help="Number of images per case (default: 5, used when --seeds not given)",
    )
    p.add_argument(
        "--hough-config", type=str, default="e6best",
        choices=list(HOUGH_PRESETS),
        help="Hough parameter preset (default: e6best)",
    )
    p.add_argument(
        "--tag", type=str, default="",
        help="Optional experiment tag for the output directory name",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output directory (default: out/bench_{tag}_{timestamp})",
    )
    p.add_argument(
        "--save-plots", action="store_true",
        help="Save diagnostic plots to output directory",
    )
    p.add_argument(
        "--no-print", action="store_true",
        help="Skip console output (quiet mode)",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # ---- resolve output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.tag}" if args.tag else ""
    if args.out:
        out_dir = args.out
    else:
        out_dir = f"out/bench{tag_part}_{timestamp}"
    # If --out was given but includes no timestamp, append one
    if args.out and tag_part:
        out_dir = args.out.replace("{timestamp}", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # ---- resolve seeds ----
    case_names = [c.strip() for c in args.cases.split(",")]

    # ---- split hough / refine kwargs ----
    presets = HOUGH_PRESETS[args.hough_config]
    hough_kwargs: dict[str, Any] = {}
    refine_kwargs: dict[str, Any] = {}
    for k, v in presets.items():
        if k in (
            "gap_tolerance",
            "distance_thresh",
            "support_dilate",
            "angle_gate_deg",
            "gap_angle_gate_deg",
        ):
            refine_kwargs[k] = v
        else:
            hough_kwargs[k] = v

    # ---- save config ----
    config_info = {
        "cli_args": vars(args),
        "hough_preset": args.hough_config,
        "hough_kwargs": {k: v for k, v in hough_kwargs.items()},
        "refine_kwargs": {k: v for k, v in refine_kwargs.items()},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_info, f, indent=2, default=str)

    # ---- main loop ----
    all_results: list[ImageBenchResult] = []
    for case_name in case_names:
        if case_name not in FIXTURE_SPECS:
            print(f"Unknown case: {case_name}", file=sys.stderr)
            return 1
        config, default_seed, H, W = FIXTURE_SPECS[case_name]
        if args.seeds:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        else:
            seeds = list(range(default_seed, default_seed + args.n_images))
        bg = _make_background(H, W)
        for seed in seeds:
            if not args.no_print:
                print(f"  [{case_name}] seed={seed}...", end=" ", flush=True)
            result = _run_image(case_name, seed, config, H, W, bg, hough_kwargs, refine_kwargs)
            all_results.append(result)
            if not args.no_print:
                n_clusters = len(result.clusters)
                n_gt = sum(cr.n_gt_edges for cr in result.clusters)
                tp = sum(cr.seg_metrics.n_tp for cr in result.clusters)
                fn = sum(cr.seg_metrics.n_fn for cr in result.clusters)
                fp = sum(cr.seg_metrics.n_fp + cr.seg_metrics.n_fp_near for cr in result.clusters)
                print(f"{n_clusters} clusters, {n_gt} GT edges, seg TP={tp} FN={fn} FP={fp}")

    # ---- breakdown ----
    breakdown = _build_breakdown(all_results)

    # ---- print summary always (per-image detail only if not --no-print) ----
    for result in all_results:
        _print_image_report(result)
    _print_summary(all_results, breakdown)

    # ---- CSV outputs ----
    cluster_csv = os.path.join(out_dir, "per_cluster.csv")
    _write_cluster_csv(cluster_csv, all_results)
    print(f"\n  Cluster CSV: {cluster_csv}")

    edge_csv = os.path.join(out_dir, "per_edge.csv")
    _write_edge_csv(edge_csv, all_results)
    print(f"  Per-edge CSV: {edge_csv}")

    json_path = os.path.join(out_dir, "summary.json")
    _write_json_summary(json_path, all_results, breakdown, args)
    print(f"  JSON summary: {json_path}")

    if args.save_plots:
        _save_plots(out_dir, all_results)

    print(f"\n  Output directory: {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
