"""Gradient-guided Hough line detection on NMS-thinned edges.

Input is the (nms, angle) pair from ``extract_thin_edges`` in ``edges.py``.
Output is a list of ``LineSegment`` instances with refined lines and
segment endpoints suitable for both visualization and downstream corner extraction.

Coordinate convention
---------------------
All geometric primitives use **pixel coordinates**::

    x = column index  (0 … W-1)
    y = row index     (0 … H-1)

Line equation:  ``normal · p = rho``  where ``p = [x, y]ᵀ``.

``rho`` is canonicalised to **≥ 0** (the sign of ``rho`` is not used
for orientation — the line is undirected).  The ``endpoints`` on a
``LineSegment`` are the projected extent of the longest contiguous
support run in pixel coordinates, ready to plot directly onto the ROI.

Algorithm sketch
----------------
1. **Gradient-guided Hough voting** — one theta bin per edge pixel
   (the gradient-normal angle modulo π).  Votes are weighted by edge
   strength.  Accumulator built with ``numpy.bincount``.

2. **Peak NMS** — iterative argmax with local suppression in the
   accumulator.  ``nms_radius_theta`` / ``nms_radius_rho`` control how
   many accumulator bins are zeroed around each peak.  These are
   **key tuning knobs**: larger radii reduce duplicate line registrations
   but may suppress genuinely distinct nearby lines; smaller radii risk
   registering the same physical line multiple times.

3. **Line refinement** — for each candidate (normal, rho) the support
   edge pixels are collected (within ``distance_thresh``), a weighted
   total-least-squares (TLS) fit refines the line, and the longest
   contiguous support run (with ``gap_tolerance`` bridging) defines the
   segment endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LineSegment:
    """A refined line segment detected by gradient-guided Hough voting.

    Attributes
    ----------
    normal : ndarray, shape (2,)
        Unit normal vector of the line (``normal · p = rho``).
        In pixel coordinates (x=col, y=row). Canonicalised so ``rho >= 0``.
    rho : float
        Signed distance of the line from the image origin (top-left corner).
        Canonicalised ≥ 0.
    endpoints : ndarray, shape (2, 2)
        Two (x, y) pixel-coordinate points marking the projected extent of
        the longest contiguous support run.  Ready for direct plotting.
    vote_score : float
        Accumulator peak score from the Hough voting stage (weighted sum of
        edge strengths that voted for this line's bin).
    """

    normal: np.ndarray
    rho: float
    endpoints: np.ndarray
    vote_score: float


# ---------------------------------------------------------------------------
# Hough voting + peak extraction
# ---------------------------------------------------------------------------


def hough_vote_peaks(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 2.0,
    rho_step: float = 1.0,
    threshold_rel: float = 0.25,
    max_peaks: int = 20,
    nms_radius_theta: int = 3,
    nms_radius_rho: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient-guided one-theta Hough voting followed by peak extraction.

    Each edge pixel votes into exactly one theta bin (its gradient-normal
    angle modulo π).  The vote weight is the NMS edge magnitude.

    Parameters
    ----------
    nms : ndarray, shape (H, W)
        NMS edge magnitudes from ``extract_thin_edges``.
    angle : ndarray, shape (H, W)
        Edge-normal angles ``atan2(gy, gx)`` in [-π, π], also from
        ``extract_thin_edges``.  Zero where ``nms == 0``.
    theta_step_deg : float
        Angular bin size in degrees.  Default 2.0°.
    rho_step : float
        Rho bin size in pixels.  Default 1.0 px.
    threshold_rel : float
        Peaks below ``threshold_rel * acc.max()`` are discarded.  Default 0.25.
    max_peaks : int
        Maximum number of peaks to return.  Default 20.
    nms_radius_theta : int
        Number of theta *bins* suppressed around each detected peak.
        Larger values reduce duplicate line registrations at the cost of
        possibly missing distinct nearby lines.  Default 3 (≈ ±6°).
    nms_radius_rho : int
        Number of rho *bins* suppressed around each detected peak.
        Larger values reduce duplicate registrations; too large may merge
        distinct parallel lines.  Default 6 (≈ ±6 px).

    Returns
    -------
    normals : ndarray, shape (K, 2)
        Unit normal vectors for each detected line.
    rhos : ndarray, shape (K,)
        Signed distances (≥ 0) from the top-left origin in pixels.
    scores : ndarray, shape (K,)
        Accumulator peak scores (higher = stronger evidence).
    """
    H, W = nms.shape

    # ---- edge pixels -------------------------------------------------------
    ys, xs = np.nonzero(nms)
    strengths = nms[ys, xs].astype(np.float64)

    # Gradient-normal angle modulo π (the same line is described by
    # normals θ and θ+π, so we collapse them).
    thetas = np.fmod(angle[ys, xs], np.pi)
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)

    # ---- binning -----------------------------------------------------------
    theta_step = np.deg2rad(theta_step_deg)
    n_theta = int(np.ceil(np.pi / theta_step))

    rho_max = np.hypot(W, H)
    n_rho = int(np.ceil(rho_max / rho_step)) + 1

    # Quantise theta (one bin per edge pixel).
    theta_idx = np.round(thetas / theta_step).astype(np.int32) % n_theta
    theta_q = theta_idx.astype(np.float64) * theta_step

    # Rho in pixel coordinates: rho = x cos θ + y sin θ.
    rho_vals = xs.astype(np.float64) * np.cos(theta_q) + ys.astype(np.float64) * np.sin(
        theta_q
    )
    rho_idx = np.round(rho_vals / rho_step).astype(np.int32)

    valid = (rho_idx >= 0) & (rho_idx < n_rho)

    # ---- accumulator via bincount ------------------------------------------
    flat_idx = theta_idx[valid] * n_rho + rho_idx[valid]
    acc_flat = np.bincount(
        flat_idx, weights=strengths[valid], minlength=n_theta * n_rho
    )
    acc = acc_flat.reshape(n_theta, n_rho).astype(np.float64)

    # ---- iterative peak NMS ------------------------------------------------
    work = acc.copy()
    acc_max = work.max()
    if acc_max <= 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    threshold = threshold_rel * acc_max

    peaks_theta: list[int] = []
    peaks_rho: list[int] = []
    peaks_score: list[float] = []

    for _ in range(max_peaks):
        idx = np.argmax(work.ravel())
        value = float(work.ravel()[idx])

        if value < threshold:
            break

        t_idx, r_idx = map(int, np.unravel_index(idx, work.shape))
        peaks_theta.append(t_idx)
        peaks_rho.append(r_idx)
        peaks_score.append(value)

        # Circular suppression in theta, bounded in rho.
        r0 = max(0, r_idx - nms_radius_rho)
        r1 = min(n_rho, r_idx + nms_radius_rho + 1)
        for dt in range(-nms_radius_theta, nms_radius_theta + 1):
            tt = (t_idx + dt) % n_theta
            work[tt, r0:r1] = 0.0

    k = len(peaks_theta)
    if k == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    # ---- convert to geometric form -----------------------------------------
    t_arr = np.array(peaks_theta, dtype=np.float64) * theta_step
    r_arr = np.array(peaks_rho, dtype=np.float64) * rho_step

    # Keep rho ≥ 0 by flipping the normal when rho < 0.
    neg = r_arr < 0
    if np.any(neg):
        r_arr = np.where(neg, -r_arr, r_arr)
        t_arr = np.where(neg, (t_arr + np.pi) % np.pi, t_arr)

    cos_t = np.cos(t_arr)
    sin_t = np.sin(t_arr)
    normals = np.column_stack([cos_t, sin_t])
    rhos = r_arr
    scores = np.array(peaks_score, dtype=np.float64)

    # Re-sort by score descending (peak NMS may not preserve order).
    order = np.argsort(-scores)
    return normals[order], rhos[order], scores[order]


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
    """Refine a Hough candidate line by weighted TLS fit to nearby edge pixels.

    Collects edge pixels within ``distance_thresh`` of the approximate line,
    fits a weighted total-least-squares line, then finds the longest
    contiguous support run with gap bridging to determine segment endpoints.

    Parameters
    ----------
    normal : ndarray, shape (2,)
        Approximate unit normal from Hough voting.
    rho : float
        Approximate signed distance from Hough voting.
    vote_score : float
        Accumulator peak score from ``hough_vote_peaks`` (passed through to
        the returned ``LineSegment``).
    nms : ndarray, shape (H, W)
        NMS edge magnitudes.
    angle : ndarray, shape (H, W)
        Edge-normal angles (not used during refinement, provided for
        interface uniformity).
    gap_tolerance : float
        Maximum gap in pixels to bridge when finding the longest contiguous
        support run.  Default 2.0 px.
    distance_thresh : float
        Maximum perpendicular distance (pixels) for an edge pixel to be
        considered supporting this line.  Default 1.5 px.

    Returns
    -------
    LineSegment
        Refined line with segment endpoints and vote score.
    """
    H, W = nms.shape

    # ---- collect support ---------------------------------------------------
    ys, xs = np.nonzero(np.asarray(nms))
    strengths = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    dists = np.abs(points @ normal - rho)
    mask = dists < distance_thresh

    support_pts = points[mask]
    support_strengths = strengths[mask]

    if len(support_pts) < 2:
        # Not enough support — return a best-effort degenerate segment.
        return LineSegment(
            normal=normal.copy(),
            rho=rho,
            endpoints=np.zeros((2, 2), dtype=np.float64),
            vote_score=vote_score,
        )

    # ---- weighted TLS fit --------------------------------------------------
    w = support_strengths / support_strengths.sum()
    c = (support_pts * w[:, None]).sum(axis=0)  # weighted centroid
    X = support_pts - c
    Xw = X * np.sqrt(w[:, None])
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)

    direction = vt[0]  # direction *along* the line
    refined_normal = vt[1]  # normal *to* the line

    # Canonicalise rho >= 0.
    refined_rho = float(refined_normal @ c)
    if refined_rho < 0:
        refined_normal = -refined_normal
        refined_rho = -refined_rho
        # direction is still perpendicular (orthonormal basis preserves sign
        # relationship), so it stays as-is.

    # ---- longest contiguous support run ------------------------------------
    proj = support_pts @ direction  # scalar projection onto line direction
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

    # Final run.
    run_len = run_b - run_a
    if run_len > best_len:
        best_len = run_len
        best_a = run_a
        best_b = run_b

    # Convert projection bounds back to (x, y) endpoints on the refined line.
    # Any point on the line can be written as:
    #     p = rho * n + t * d
    ep1 = refined_rho * refined_normal + best_a * direction
    ep2 = refined_rho * refined_normal + best_b * direction

    return LineSegment(
        normal=refined_normal,
        rho=refined_rho,
        endpoints=np.array([ep1, ep2], dtype=np.float64),
        vote_score=vote_score,
    )
