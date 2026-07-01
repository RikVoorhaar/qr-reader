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
from qr_reader.detector.hough import LineSegment, hough_vote_peaks, refine_line
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


def _edge_normal_from_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0
    direction = d / length
    normal = np.array([direction[1], -direction[0]], dtype=np.float64)
    rho = float(normal @ a)
    if rho < 0:
        normal = -normal
        rho = -rho
    return normal, rho


def _compute_finder_edges(
    metadata: dict,
    roi_offset: tuple[int, int] | None = None,
    roi_shape: tuple[int, int] | None = None,
) -> list[dict]:
    corners = metadata["corners_qr"]
    N = metadata["N"]
    frac = 7.0 / N
    TL = np.array(corners["TL"], dtype=np.float64)
    TR = np.array(corners["TR"], dtype=np.float64)
    BR = np.array(corners["BR"], dtype=np.float64)
    BL = np.array(corners["BL"], dtype=np.float64)
    edge_specs = [
        (TL, TR, "TL_top"),
        (TL, BL, "TL_left"),
        (TR, TL, "TR_top"),
        (TR, BR, "TR_right"),
        (BL, TL, "BL_left"),
        (BL, BR, "BL_bottom"),
        (BR, TR, "BR_right"),
        (BR, BL, "BR_bottom"),
    ]
    results: list[dict] = []
    for start, toward, label in edge_specs:
        a = start
        b = start + frac * (toward - start)
        normal, rho = _edge_normal_from_points(a, b)
        if roi_offset is not None and roi_shape is not None:
            row0, col0 = int(roi_offset[0]), int(roi_offset[1])
            H, W = int(roi_shape[0]), int(roi_shape[1])
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
                a_local, b_local, 0.0, float(W - 1), 0.0, float(H - 1)
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
    return results


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
        normals, rhos, scores = hough_vote_peaks(nms, angle, **hough_kwargs)
        t_vote_elapsed = (time.perf_counter() - t_vote) * 1000.0

        gt_edges = _compute_finder_edges(
            metadata, roi_offset=(bbox[0], bbox[2]), roi_shape=roi.shape
        )

        t_refine = time.perf_counter()
        Df, Af, Cf, Bf, hits, m_idxs, segs, len_rs, rep_es = _classify_failures(
            gt_edges, normals, rhos, scores, nms, angle, cluster_idx=ci,
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
    # score in the same theta band.  We don't compute full accumulator here —
    # record as NaN for baseline runs.
    snr_mean = float("nan")
    snr_p05 = float("nan")

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
    from qr_reader.detector.edges import extract_thin_edges

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Edge-Angle Histogram — Cluster {cl.cluster_idx}")

    # We need the raw NMS + angle from the ROI — stored in the pipeline step
    # For now, mark this as placeholder.
    ax.text(0.5, 0.5, "edge-angle histogram\n(needs NMS/angle from pipeline)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "edge_angle_histogram.png", dpi=150)
    plt.close(fig)


def _plot_accumulator_heatmaps(cl: ClusterResult, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Accumulator Schematic — Cluster {cl.cluster_idx}")
    ax.text(0.5, 0.5, "accumulator heatmaps\n(full E2 mode will generate per-GT-edge heatmaps)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "accumulator_heatmaps.png", dpi=150)
    plt.close(fig)


def _plot_support_maps(cl: ClusterResult, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Support Maps — Cluster {cl.cluster_idx}")
    ax.text(0.5, 0.5, "per-peak support maps\n(needs NMS/angle data)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "support_maps.png", dpi=150)
    plt.close(fig)


def _plot_support_density(cl: ClusterResult, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Support Density — Cluster {cl.cluster_idx}")
    ax.text(0.5, 0.5, "support-density plots\n(per-GT-edge)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "support_density.png", dpi=150)
    plt.close(fig)


def _plot_rho_vs_theta(cl: ClusterResult, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Rho-vs-Theta — Cluster {cl.cluster_idx}")
    ax.text(0.5, 0.5, "rho-vs-theta scatter\n(per-GT-edge)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_theta.png", dpi=150)
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
        choices=["baseline", "roi_audit", "vote_audit"],
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
