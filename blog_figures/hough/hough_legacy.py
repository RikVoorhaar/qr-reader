"""Historical Sobel-NMS + gradient-guided Hough detector, frozen for figures.

This is a verbatim-enough copy of the experimental detector that lived in
``src/qr_reader/detector/edges.py`` and ``src/qr_reader/detector/hough.py`` at
commit ``f5451e8^``.  The production package no longer contains it; keeping a
local copy means the blog figures keep rendering the same algorithm even if
the working tree moves on.

Only the parts needed for the figures are kept:

* ``extract_thin_edges`` -- Gaussian blur -> Sobel -> L2 magnitude ->
  interpolated non-maximum suppression.
* ``build_hough_accumulator`` -- one-theta-per-pixel voting weighted by edge
  strength.
* ``hough_vote_peaks`` -- iterative global-max peak extraction with local
  non-maximum suppression in (theta, rho).
* ``refine_line`` -- weighted TLS fit to the support set and longest
  contiguous run for the segment endpoints.

Coordinate convention is pixel coordinates (x = column, y = row), with the
line ``normal . p = rho`` and ``rho >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class LineSegment:
    normal: np.ndarray
    rho: float
    endpoints: np.ndarray
    vote_score: float


# ---------------------------------------------------------------------------
# Edge extraction (edges.py)
# ---------------------------------------------------------------------------


def extract_thin_edges(
    roi: np.ndarray, blur_sigma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blur -> Sobel -> L2 magnitude -> interpolated NMS."""
    roi_f = roi.astype(np.float64, copy=False)

    blurred = ndimage.gaussian_filter(roi_f, sigma=blur_sigma, mode="reflect")

    gx = ndimage.sobel(blurred, axis=1, mode="constant")
    gy = ndimage.sobel(blurred, axis=0, mode="constant")

    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx, out=np.zeros_like(mag), where=mag > 0)
    nms = _non_maximum_suppression_interpolated(mag, gx, gy)
    angle = np.where(nms > 0, angle, 0.0)

    return nms, angle


def _bilinear_sample(img: np.ndarray, y: float, x: float) -> float:
    h, w = img.shape
    y0 = int(np.clip(np.floor(y), 0, h - 1))
    x0 = int(np.clip(np.floor(x), 0, w - 1))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)
    fy = np.clip(y - y0, 0.0, 1.0)
    fx = np.clip(x - x0, 0.0, 1.0)
    return float(
        (1.0 - fy) * ((1.0 - fx) * img[y0, x0] + fx * img[y0, x1])
        + fy * ((1.0 - fx) * img[y1, x0] + fx * img[y1, x1])
    )


def _non_maximum_suppression_interpolated(
    mag: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    h, w = mag.shape
    nms = np.zeros_like(mag)
    hdom = np.abs(gx) >= np.abs(gy)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = mag[y, x]
            if m == 0:
                continue
            gx_val = gx[y, x]
            gy_val = gy[y, x]
            if hdom[y, x]:
                dy = gy_val / gx_val
                sx = np.sign(gx_val)
                n1 = _bilinear_sample(mag, y - dy, x - sx)
                n2 = _bilinear_sample(mag, y + dy, x + sx)
            else:
                dx = gx_val / gy_val
                sy = np.sign(gy_val)
                n1 = _bilinear_sample(mag, y - sy, x - dx)
                n2 = _bilinear_sample(mag, y + sy, x + dx)
            if m >= n1 and m >= n2:
                nms[y, x] = m
    return nms


# ---------------------------------------------------------------------------
# Hough voting + peak extraction (hough.py)
# ---------------------------------------------------------------------------


def build_hough_accumulator(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 0.5,
    rho_step: float = 0.5,
) -> dict:
    """One-theta-per-pixel Hough voting, weighted by NMS edge strength."""
    H, W = nms.shape

    ys, xs = np.nonzero(nms)
    strengths = nms[ys, xs].astype(np.float64)

    thetas = np.fmod(angle[ys, xs], np.pi)
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)

    theta_step_rad = np.deg2rad(theta_step_deg)
    n_theta = int(np.ceil(np.pi / theta_step_rad))

    rho_max = np.hypot(W, H)
    n_rho = int(np.ceil(rho_max / rho_step)) + 1

    theta_idx = np.round(thetas / theta_step_rad).astype(np.int32) % n_theta
    theta_q = theta_idx.astype(np.float64) * theta_step_rad

    rho_vals = (
        xs.astype(np.float64) * np.cos(theta_q)
        + ys.astype(np.float64) * np.sin(theta_q)
    )
    rho_idx = np.round(rho_vals / rho_step).astype(np.int32)

    valid = (rho_idx >= 0) & (rho_idx < n_rho)
    flat_idx = theta_idx[valid] * n_rho + rho_idx[valid]
    acc_flat = np.bincount(
        flat_idx, weights=strengths[valid], minlength=n_theta * n_rho
    )
    acc = acc_flat.reshape(n_theta, n_rho).astype(np.float64)

    return {
        "acc": acc,
        "n_theta": n_theta,
        "n_rho": n_rho,
        "theta_step_rad": theta_step_rad,
        "rho_max": rho_max,
    }


def hough_vote_peaks(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 0.5,
    rho_step: float = 0.5,
    threshold_rel: float = 0.25,
    max_peaks: int = 20,
    nms_radius_theta: int = 2,
    nms_radius_rho: int = 4,
    return_acc: bool = False,
):
    """Gradient-guided Hough voting followed by iterative peak NMS."""
    acc_data = build_hough_accumulator(
        nms, angle, theta_step_deg, rho_step
    )
    acc = acc_data["acc"]
    n_theta = acc_data["n_theta"]
    n_rho = acc_data["n_rho"]
    theta_step = acc_data["theta_step_rad"]

    work = acc.copy()
    acc_max = work.max()
    if acc_max <= 0:
        empty = (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
        return (*empty, acc_data) if return_acc else empty

    threshold = threshold_rel * acc_max

    peaks_theta: list[int] = []
    peaks_rho: list[int] = []
    peaks_score: list[float] = []

    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        value = float(work.ravel()[idx])
        if value < threshold:
            break
        t_idx, r_idx = map(int, np.unravel_index(idx, work.shape))
        peaks_theta.append(t_idx)
        peaks_rho.append(r_idx)
        peaks_score.append(value)

        r0 = max(0, r_idx - nms_radius_rho)
        r1 = min(n_rho, r_idx + nms_radius_rho + 1)
        for dt in range(-nms_radius_theta, nms_radius_theta + 1):
            tt = (t_idx + dt) % n_theta
            work[tt, r0:r1] = 0.0

    if not peaks_theta:
        empty = (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
        return (*empty, acc_data) if return_acc else empty

    t_arr = np.array(peaks_theta, dtype=np.float64) * theta_step
    r_arr = np.array(peaks_rho, dtype=np.float64) * rho_step

    neg = r_arr < 0
    if np.any(neg):
        r_arr = np.where(neg, -r_arr, r_arr)
        t_arr = np.where(neg, (t_arr + np.pi) % np.pi, t_arr)

    normals = np.column_stack([np.cos(t_arr), np.sin(t_arr)])
    scores = np.array(peaks_score, dtype=np.float64)

    order = np.argsort(-scores)
    if return_acc:
        return normals[order], r_arr[order], scores[order], acc_data
    return normals[order], r_arr[order], scores[order]


# ---------------------------------------------------------------------------
# Line refinement
# ---------------------------------------------------------------------------


def refine_line(
    normal: np.ndarray,
    rho: float,
    vote_score: float,
    nms: np.ndarray,
    angle: np.ndarray,
    gap_tolerance: float = 2.0,
    distance_thresh: float = 1.5,
) -> LineSegment:
    """Weighted TLS fit to the support set + longest contiguous run."""
    H, W = nms.shape

    ys, xs = np.nonzero(np.asarray(nms))
    strengths = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    dists = np.abs(points @ normal - rho)
    mask = dists < distance_thresh
    support_pts = points[mask]
    support_strengths = strengths[mask]

    if len(support_pts) < 2:
        return LineSegment(
            normal=normal.copy(),
            rho=rho,
            endpoints=np.zeros((2, 2), dtype=np.float64),
            vote_score=vote_score,
        )

    w = support_strengths / support_strengths.sum()
    c = (support_pts * w[:, None]).sum(axis=0)
    X = support_pts - c
    Xw = X * np.sqrt(w[:, None])
    _, _, vt = np.linalg.svd(Xw, full_matrices=False)

    direction = vt[0]
    refined_normal = vt[1]

    refined_rho = float(refined_normal @ c)
    if refined_rho < 0:
        refined_normal = -refined_normal
        refined_rho = -refined_rho

    proj = support_pts @ direction
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]

    best_len = 0.0
    best_a = 0.0
    best_b = 0.0
    run_a = float(proj_sorted[0])
    run_b = float(proj_sorted[0])

    for i in range(1, len(proj_sorted)):
        gap = float(proj_sorted[i] - proj_sorted[i - 1])
        if gap <= gap_tolerance:
            run_b = float(proj_sorted[i])
        else:
            run_len = run_b - run_a
            if run_len > best_len:
                best_len = run_len
                best_a = run_a
                best_b = run_b
            run_a = float(proj_sorted[i])
            run_b = float(proj_sorted[i])

    run_len = run_b - run_a
    if run_len > best_len:
        best_len = run_len
        best_a = run_a
        best_b = run_b

    ep1 = refined_rho * refined_normal + best_a * direction
    ep2 = refined_rho * refined_normal + best_b * direction

    return LineSegment(
        normal=refined_normal,
        rho=refined_rho,
        endpoints=np.array([ep1, ep2], dtype=np.float64),
        vote_score=vote_score,
    )
