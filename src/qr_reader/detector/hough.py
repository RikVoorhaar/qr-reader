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


def build_hough_accumulator(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 2.0,
    rho_step: float = 1.0,
    theta_window_deg: float = 0.0,
    vote_scheme: str = "onebin",
) -> dict:
    """Build the Hough accumulator and return per-pixel vote data.

    Useful for diagnostic inspection of vote clouds (E2 vote-cloud audit).

    Parameters
    ----------
    nms : ndarray, shape (H, W)
        NMS edge magnitudes.
    angle : ndarray, shape (H, W)
        Edge-normal angles in [-π, π].
    theta_step_deg : float
        Angular bin size in degrees.
    rho_step : float
        Rho bin size in pixels.
    theta_window_deg : float
        Half-window width in degrees for soft angular voting.  0 = one-bin.
    vote_scheme : str
        ``"onebin"``, ``"gaussian"``, or ``"dot"``.

    Returns
    -------
    data : dict
        Keys: ``acc`` (ndarray (n_theta, n_rho)), ``theta_idx``, ``rho_idx``
        (ndarrays of per-pixel bin indices), ``strengths``, ``n_theta``,
        ``n_rho``, ``theta_step_rad``, ``rho_max``.
    """
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

    if vote_scheme == "onebin" or theta_window_deg <= 0:
        flat_idx = theta_idx[valid] * n_rho + rho_idx[valid]
        acc_flat = np.bincount(
            flat_idx, weights=strengths[valid], minlength=n_theta * n_rho
        )
    elif vote_scheme == "gaussian":
        K = max(1, int(np.ceil(theta_window_deg / theta_step_deg)))
        offsets = np.arange(-K, K + 1, dtype=np.int32)
        sigma_deg = theta_window_deg / 3.0
        offset_deg = offsets.astype(np.float64) * theta_step_deg
        weights = np.exp(-0.5 * (offset_deg / sigma_deg) ** 2)
        weights = weights / weights.sum()  # normalise so per-pixel total = 1
        base_ti = theta_idx[valid]
        theta_idx_all = (base_ti[:, None] + offsets[None, :]) % n_theta
        flat_idx = theta_idx_all * n_rho + rho_idx[valid, None]
        sw = strengths[valid, None] * weights[None, :]
        acc_flat = np.bincount(
            flat_idx.ravel(), weights=sw.ravel(), minlength=n_theta * n_rho
        )
    elif vote_scheme == "dot":
        K = max(1, int(np.ceil(theta_window_deg / theta_step_deg)))
        offsets = np.arange(-K, K + 1, dtype=np.int32)
        sigma_deg = theta_window_deg / 3.0
        offset_deg = offsets.astype(np.float64) * theta_step_deg
        weights = np.exp(-0.5 * (offset_deg / sigma_deg) ** 2)
        weights = weights / weights.sum()
        base_ti = theta_idx[valid]
        theta_idx_all = (base_ti[:, None] + offsets[None, :]) % n_theta
        flat_idx = theta_idx_all * n_rho + rho_idx[valid, None]
        sw = strengths[valid, None] * weights[None, :]
        acc_flat = np.bincount(
            flat_idx.ravel(), weights=sw.ravel(), minlength=n_theta * n_rho
        )
    else:
        raise ValueError(f"Unknown vote_scheme: {vote_scheme}")

    acc = acc_flat.reshape(n_theta, n_rho).astype(np.float64)

    theta_idx_full = np.full(len(theta_idx), -1, dtype=np.int32)
    rho_idx_full = np.full(len(rho_idx), -1, dtype=np.int32)
    strengths_full = np.zeros(len(strengths), dtype=np.float64)
    theta_idx_full[valid] = theta_idx[valid]
    rho_idx_full[valid] = rho_idx[valid]
    strengths_full[valid] = strengths[valid]

    return {
        "acc": acc,
        "theta_idx": theta_idx_full,
        "rho_idx": rho_idx_full,
        "strengths": strengths_full,
        "n_theta": n_theta,
        "n_rho": n_rho,
        "theta_step_rad": theta_step_rad,
        "rho_max": rho_max,
    }


def hough_vote_peaks(
    nms: np.ndarray,
    angle: np.ndarray,
    theta_step_deg: float = 2.0,
    rho_step: float = 1.0,
    threshold_rel: float = 0.25,
    max_peaks: int = 20,
    nms_radius_theta: int = 3,
    nms_radius_rho: int = 6,
    return_acc: bool = False,
    theta_window_deg: float = 0.0,
    vote_scheme: str = "onebin",
    acc_smooth: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient-guided one-theta Hough voting followed by peak extraction.

    Each edge pixel votes into the theta bin(s) determined by its
    gradient-normal angle modulo π.  The vote weight is the NMS edge
    magnitude.  Soft angular voting (``theta_window_deg > 0``) spreads each
    pixel's vote into 2K+1 theta bins weighted by an angular kernel.

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
    return_acc : bool
        If True, also returns the accumulator dict from
        ``build_hough_accumulator`` as a fourth element.
    theta_window_deg : float
        Half-window width in degrees for soft angular voting.  ``0`` = one-bin
        (current behaviour).  Default 0.
    vote_scheme : str
        Voting kernel when ``theta_window_deg > 0``:
        ``"gaussian"`` — Gaussian-weighted spreading;
        ``"dot"`` — dot-product-weighted spreading.  Default ``"onebin"``.
    acc_smooth : str or None
        Accumulator smoothing along the rho axis before peak NMS.
        ``None`` (off), ``"1x3_triangular"``, ``"1x5_triangular"``.
        Triangular kernels weight nearby rho bins to reduce fragmentation
        without merging parallel lines in theta.

    Returns
    -------
    normals : ndarray, shape (K, 2)
        Unit normal vectors for each detected line.
    rhos : ndarray, shape (K,)
        Signed distances (≥ 0) from the top-left origin in pixels.
    scores : ndarray, shape (K,)
        Accumulator peak scores (higher = stronger evidence).
    acc_data : dict, optional (when ``return_acc=True``)
        Accumulator data dict from ``build_hough_accumulator``.
    """
    acc_data = build_hough_accumulator(
        nms, angle, theta_step_deg, rho_step,
        theta_window_deg=theta_window_deg, vote_scheme=vote_scheme,
    )
    acc = acc_data["acc"]
    n_theta = acc_data["n_theta"]
    n_rho = acc_data["n_rho"]
    theta_step = acc_data["theta_step_rad"]

    # ---- optional accumulator smoothing along rho axis ------------------------
    if acc_smooth == "1x3_triangular":
        from scipy.ndimage import convolve1d  # noqa: PLC0415
        kernel = np.array([1.0, 2.0, 1.0], dtype=np.float64) / 4.0
        work = convolve1d(acc.astype(np.float64), kernel, axis=1, mode="reflect")
    elif acc_smooth == "1x5_triangular":
        from scipy.ndimage import convolve1d  # noqa: PLC0415
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64) / 9.0
        work = convolve1d(acc.astype(np.float64), kernel, axis=1, mode="reflect")
    else:
        work = acc.copy()

    # ---- iterative peak NMS ------------------------------------------------
    acc_max = work.max()
    if acc_max <= 0:
        if return_acc:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                acc_data,
            )
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
        if return_acc:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                acc_data,
            )
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
    if return_acc:
        return normals[order], rhos[order], scores[order], acc_data
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
    angle_gate_deg: float | None = None,
    gap_angle_gate_deg: float | None = None,
    support_mask: np.ndarray | None = None,
    support_dilate: int = 0,
) -> LineSegment:
    """Refine a Hough candidate line by weighted TLS fit to nearby edge pixels.

    Collects edge pixels within ``distance_thresh`` of the approximate line,
    optionally filtered by gradient-angle consistency with the Hough peak
    normal, then fits a weighted total-least-squares line, then finds the
    longest contiguous support run with gap bridging to determine segment
    endpoints.

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
        Edge-normal angles (gradient directions).  Zero where ``nms == 0``.
    gap_tolerance : float
        Maximum gap in pixels to bridge when finding the longest contiguous
        support run.  Default 2.0 px.
    distance_thresh : float
        Maximum perpendicular distance (pixels) for an edge pixel to be
        considered supporting this line.  Default 1.5 px.
    angle_gate_deg : float, optional
        If provided, only edge pixels whose gradient-normal angle is within
        ``angle_gate_deg`` degrees (modulo π) of the Hough peak normal angle
        are included in the support set.
    gap_angle_gate_deg : float, optional
        If provided, gaps exceeding ``gap_tolerance`` are still bridged if the
        NMS content at the gap midpoint is at a gradient-normal angle
        consistent with the segment normal (within ``gap_angle_gate_deg``
        degrees).  This distinguishes structural gaps (wrong-angle crossing
        edges) from noise dropouts and partial gaps.  Default ``None``
        (always split at ``gap_tolerance``).
    support_mask : ndarray[bool], optional, shape (H, W)
        If provided, only pixels where ``support_mask`` is True and
        ``nms > 0`` are considered for support collection.  Use for hysteresis
        linking: vote from raw NMS, refine from linked mask.
    support_dilate : int
        Binary dilation iterations applied to the ``nms > 0`` mask (or
        ``support_mask`` if provided) before support collection.  ``0`` = no
        dilation.  Default 0.

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

    if support_mask is not None:
        keep = support_mask[ys, xs]
        ys = ys[keep]
        xs = xs[keep]
        strengths = strengths[keep]
        points = points[keep]
    elif support_dilate > 0:
        from scipy.ndimage import binary_dilation  # noqa: PLC0415
        base_mask = np.asarray(nms) > 0
        d_mask = binary_dilation(base_mask, iterations=support_dilate)
        keep = d_mask[ys, xs]
        ys = ys[keep]
        xs = xs[keep]
        strengths = strengths[keep]
        points = points[keep]

    dists = np.abs(points @ normal - rho)
    mask = dists < distance_thresh

    # ---- angle gate --------------------------------------------------------
    if angle_gate_deg is not None:
        hough_theta = float(np.arctan2(normal[1], normal[0]))
        edge_thetas = np.fmod(np.abs(angle[ys, xs]), np.pi)
        theta_diff = np.abs(edge_thetas - (hough_theta % np.pi))
        theta_diff = np.minimum(theta_diff, np.pi - theta_diff)
        mask &= theta_diff < np.deg2rad(angle_gate_deg)

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
    sorted_pts = support_pts[sort_idx]

    # Precompute the segment normal angle (mod π) for gap angle-gating.
    line_theta = None
    if gap_angle_gate_deg is not None:
        line_theta = float(np.arctan2(refined_normal[1], refined_normal[0])) % np.pi

    best_len = 0.0
    best_a = 0.0
    best_b = 0.0

    run_a = float(proj_sorted[0])
    run_b = float(proj_sorted[0])

    for i in range(1, len(proj_sorted)):
        gap = float(proj_sorted[i] - proj_sorted[i - 1])
        if gap <= gap_tolerance:
            run_b = float(proj_sorted[i])
        elif gap_angle_gate_deg is not None:
            # Check NMS pixels whose projection falls in this gap region
            # and which are near the line.  If any have a consistent angle
            # (or none exist at all — dropout), bridge the gap.
            proj_a = float(proj_sorted[i - 1])
            proj_b = float(proj_sorted[i])
            gap_mid = (proj_a + proj_b) / 2.0
            mp = refined_rho * refined_normal + gap_mid * direction  # (x, y)
            cx, cy = int(round(float(mp[0]))), int(round(float(mp[1])))
            bridge_gap = False
            if 0 <= cy < H and 0 <= cx < W:
                # Check 3x3 neighborhood around the gap midpoint.
                y0, y1 = max(0, cy - 1), min(H, cy + 2)
                x0, x1 = max(0, cx - 1), min(W, cx + 2)
                patch_nms = nms[y0:y1, x0:x1]
                patch_angle = angle[y0:y1, x0:x1]
                has_consistent = False
                has_nms = False
                for dy in range(patch_nms.shape[0]):
                    for dx in range(patch_nms.shape[1]):
                        nv = float(patch_nms[dy, dx])
                        if nv > 0:
                            has_nms = True
                            ang = np.fmod(np.abs(float(patch_angle[dy, dx])), np.pi)
                            td = abs(ang - line_theta)
                            td = min(td, np.pi - td)
                            if td < np.deg2rad(gap_angle_gate_deg):
                                has_consistent = True
                                break
                    if has_consistent:
                        break
                # Bridge if: dropout (no NMS at all) OR has consistent-angle NMS.
                if not has_nms or has_consistent:
                    bridge_gap = True
            # NOTE: if gap midpoint falls outside image, we conservatively
            # split (do not bridge).
            if bridge_gap:
                run_b = float(proj_sorted[i])
                continue
            run_len = run_b - run_a
            if run_len > best_len:
                best_len = run_len
                best_a = run_a
                best_b = run_b
            run_a = float(proj_sorted[i])
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
