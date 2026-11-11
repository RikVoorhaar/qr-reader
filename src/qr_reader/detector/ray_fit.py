"""Ray-profile-based finder pattern fitting.

Samples radial intensity profiles from a candidate cluster centre, fits
per-ray module-pitch estimates via a soft template, then derives finder
edges and corners via an edge-clustering pipeline.

Public API
----------
``fit_finder_ray`` — single entry point returning ``RayFitResult``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfc

from qr_reader.detector.edge_fitting import (
    assign_points,
    compute_boundary_points,
    compute_corners,
    fit_finder_edges,
    refine_finder_edges_joint,
    _reorder_to_standard,
)

# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RayFitResult:
    """Result of ray-profile-based finder pattern fitting.

    Attributes
    ----------
    corners : ndarray (4, 2)
        Outer finder corners in ``(x, y)`` ROI-local coordinates.
    score : float
        Quality score in [0, 1]; higher is better.
    valid : bool
        ``True`` if the concentration check passed.
    """

    corners: np.ndarray
    score: float
    valid: bool


# ── Soft template ────────────────────────────────────────────────────────────


def finder_soft_template(
    t: np.ndarray, m: float, sigma: float = 1.0
) -> np.ndarray:
    """Find pattern intensity template along a radial ray.

    The ideal finder pattern cross-section from centre outward (in module
    units) has a 3-module-wide dark centre, then a 1-module white ring, a
    1-module dark ring, and the white quiet zone::

        dark (0) --[1.5]--> white (1) --[2.5]--> dark (0) --[3.5]--> white (1)

    Transitions are smoothed via ``erfc``.

    Parameters
    ----------
    t : ndarray
        Signed distances from centre in pixels.
    m : float
        Module pitch in pixels.
    sigma : float
        Smoothing scale in pixels (default 1.0).

    Returns
    -------
    template : ndarray
        Expected normalised intensities in [0, 1].
    """
    u = np.abs(np.asarray(t, dtype=np.float64)) / m
    s = sigma / m
    sqrt2 = np.sqrt(2.0)

    result = 0.5 * erfc(-(u - 1.5) / (s * sqrt2))
    result -= 0.5 * erfc(-(u - 2.5) / (s * sqrt2))
    result += 0.5 * erfc(-(u - 3.5) / (s * sqrt2))
    return result


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize_roi_intensities(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    sigma_factor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize ROI intensities to [0, 1] using centre-weighted percentiles.

    Pixels near the finder-pattern centre are weighted more heavily so that
    the dark/bright mapping reflects the finder pattern's contrast, not the
    background's.

    Returns
    -------
    roi_norm : ndarray (H, W)
        Normalized intensities in [0, 1].
    dark : float
        Weighted p10 (mapped to 0).
    bright : float
        Weighted p90 (mapped to 1).
    """
    H, W = roi.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt(
        (xs.astype(np.float64) - center_xy[0]) ** 2
        + (ys.astype(np.float64) - center_xy[1]) ** 2
    )
    sigma = sigma_factor * 3.5 * m_est
    weights = np.exp(-0.5 * (dist / sigma) ** 2)

    vals = roi.ravel().astype(np.float64)
    w = weights.ravel()

    order = np.argsort(vals)
    vals_sorted = vals[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    def _weighted_percentile(percentile: float) -> float:
        if total_w == 0.0:
            return float(np.percentile(vals, percentile))
        target = percentile / 100.0 * total_w
        idx = int(np.searchsorted(cum_w, target))
        idx = max(0, min(idx, len(vals_sorted) - 1))
        return float(vals_sorted[idx])

    dark = _weighted_percentile(10.0)
    bright = _weighted_percentile(90.0)

    span = bright - dark
    if span < 1.0:
        span = 1.0
    roi_norm = np.clip((roi.astype(np.float64) - dark) / span, 0.0, 1.0)
    return roi_norm, dark, bright


# ── Ray sampling ──────────────────────────────────────────────────────────────


def sample_ray_profiles(
    roi: np.ndarray,
    center_x: float,
    center_y: float,
    num_rays: int = 36,
    num_samples: int = 120,
    ray_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pixel intensities along half-rays outward from a centre point.

    Parameters
    ----------
    roi : ndarray (H, W)
        Grayscale ROI.
    center_x, center_y : float
        Ray origin in ROI-local (x=col, y=row) coordinates.
    num_rays : int
        Number of equally-spaced half-rays in ``[0, 2π)``.
    num_samples : int
        Number of sample points per half-ray.
    ray_length : float
        Ray extent as a fraction of half the ROI diagonal length.

    Returns
    -------
    profiles : ndarray (num_rays, num_samples)
        Sampled intensities.  Each row is one half-ray from centre outward
        (distance 0 → max_dist).
    max_dist : float
        Maximum sample distance in pixels.
    """
    H_roi, W_roi = roi.shape
    roi_f = roi.astype(np.float64)
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = ray_length * diag_half

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    sample_ts = np.linspace(0, max_dist, num_samples)
    dx = np.cos(angles)[:, None]
    dy = np.sin(angles)[:, None]

    sx = center_x + sample_ts[None, :] * dx  # (R, S)
    sy = center_y + sample_ts[None, :] * dy  # (R, S)

    x0 = np.clip(np.floor(sx).astype(int), 0, W_roi - 1)
    y0 = np.clip(np.floor(sy).astype(int), 0, H_roi - 1)
    x1 = np.clip(x0 + 1, 0, W_roi - 1)
    y1 = np.clip(y0 + 1, 0, H_roi - 1)
    fx = sx - x0.astype(np.float64)
    fy = sy - y0.astype(np.float64)

    profiles = (
        (1 - fy) * ((1 - fx) * roi_f[y0, x0] + fx * roi_f[y0, x1])
        + fy * ((1 - fx) * roi_f[y1, x0] + fx * roi_f[y1, x1])
    )

    profiles = np.clip(profiles, 0, 255)
    return profiles, max_dist


# ── Per-ray m fitting ────────────────────────────────────────────────────────


def _masked_mse(
    t_valid: np.ndarray,
    p_valid: np.ndarray,
    m: float,
    mask_boundary: float,
    sigma: float,
) -> float:
    """MSE per unmasked sample between profile and finder soft-template.

    Masked region (not used in loss): |t| > mask_boundary * m.
    """
    abs_t = np.abs(t_valid)
    inside_mask = abs_t <= mask_boundary * m
    n_inside = int(np.sum(inside_mask))
    if n_inside < 3:
        return np.inf
    template = finder_soft_template(t_valid[inside_mask], m, sigma)
    return float(np.mean((template - p_valid[inside_mask]) ** 2))


def fit_half_ray(
    t_samples: np.ndarray,
    profile: np.ndarray,
    m_est: float,
    mask_boundary: float = 4.5,
    num_grid: int = 50,
    grid_width: float = 2.0,
    sigma: float = 1.0,
) -> dict:
    """Fit *m* to a single half-ray profile via grid search + bounded refine.

    Phase 1 — grid search (masked loss, mask recomputed per grid point).
    Phase 2 — ``minimize_scalar`` within ±1 grid step of winner,
              mask frozen at the winning *m*.
    """
    mask = np.isfinite(profile)
    if np.sum(mask) < 10:
        return {"m_fitted": m_est, "mse": np.inf, "success": False}

    t_valid = t_samples[mask]
    p_valid = profile[mask]

    m_low = m_est / grid_width
    m_high = m_est * grid_width
    m_grid = np.linspace(m_low, m_high, num_grid)
    losses = np.full(num_grid, np.inf, dtype=np.float64)
    for i, m in enumerate(m_grid):
        losses[i] = _masked_mse(t_valid, p_valid, m, mask_boundary, sigma)

    best_idx = np.argmin(losses)
    m_best = m_grid[best_idx]
    best_loss = losses[best_idx]

    if not np.isfinite(best_loss):
        return {"m_fitted": m_est, "mse": best_loss, "success": False}

    abs_t = np.abs(t_valid)
    inside_mask = abs_t <= mask_boundary * m_best
    n_inside = int(np.sum(inside_mask))
    if n_inside < 3:
        return {"m_fitted": m_best, "mse": best_loss, "success": True}

    t_refine = t_valid[inside_mask]
    p_refine = p_valid[inside_mask]

    def cost(m_val):
        template = finder_soft_template(t_refine, m_val, sigma)
        return float(np.mean((template - p_refine) ** 2))

    step = m_grid[1] - m_grid[0] if num_grid > 1 else m_est * 0.05
    result = minimize_scalar(
        cost,
        bounds=(m_best - step, m_best + step),
        method="bounded",
    )
    return {
        "m_fitted": float(result.x),
        "mse": float(result.fun),
        "success": bool(result.success),
    }


def _minimize_bounded_batch(
    t_refine: np.ndarray,
    p_refine: np.ndarray,
    n_real: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: float,
    xatol: float = 1e-5,
    maxiter: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``scipy.optimize.minimize_scalar(method="bounded")``.

    Each row of *t_refine* / *p_refine* is one independent 1-D problem.  Rows
    are zero-padded at the tail; *n_real* holds each row's real sample count.
    The algorithm mirrors SciPy's Brent implementation operation for operation,
    so each row produces the same ``x``, ``fun`` and ``success`` as calling
    ``minimize_scalar`` on that row.

    Returns ``(x, fun, success)`` arrays of shape ``(n_rows,)``.
    """
    abs_t = np.abs(t_refine)
    sqrt2 = np.sqrt(2.0)
    n_rows = t_refine.shape[0]
    n_int = n_real.astype(np.intp)
    widths = np.arange(t_refine.shape[1])[None, :] < n_real[:, None]

    # Flatten the valid (row, sample) pairs so the template is only evaluated
    # where it is actually used.
    row_ids = np.repeat(np.arange(n_rows), n_int)
    col_ids = (
        np.concatenate([np.arange(n) for n in n_int])
        if n_rows
        else np.empty(0, dtype=np.intp)
    )
    t_flat = abs_t[row_ids, col_ids]

    def evaluate(x: np.ndarray) -> np.ndarray:
        m_flat = x[row_ids]
        u = t_flat / m_flat
        s = sigma / m_flat
        template = 0.5 * erfc(-(u - 1.5) / (s * sqrt2))
        template -= 0.5 * erfc(-(u - 2.5) / (s * sqrt2))
        template += 0.5 * erfc(-(u - 3.5) / (s * sqrt2))
        template_full = np.zeros_like(abs_t)
        template_full[row_ids, col_ids] = template
        sq = (template_full - p_refine) ** 2
        # Masked reduction over each row's exact sample count reproduces
        # ``np.mean`` on the unpadded row bit for bit.
        return np.sum(sq, axis=1, where=widths) / n_real

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a = lower.astype(np.float64).copy()
    b = upper.astype(np.float64).copy()
    fulc = a + golden_mean * (b - a)
    nfc = fulc.copy()
    xf = fulc.copy()
    rat = np.zeros(n_rows)
    e = np.zeros(n_rows)
    fx = evaluate(xf)
    num = np.ones(n_rows, dtype=np.int64)
    ffulc = fx.copy()
    fnfc = fx.copy()
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    fu = np.full(n_rows, np.inf)
    flag = np.zeros(n_rows, dtype=np.int64)

    active = np.abs(xf - xm) > (tol2 - 0.5 * (b - a))
    while np.any(active):
        # ── parabolic candidate ──
        cond_e = np.abs(e) > tol1
        r = (xf - nfc) * (fx - ffulc)
        q = (xf - fulc) * (fx - fnfc)
        p = (xf - fulc) * q - (xf - nfc) * r
        q = 2.0 * (q - r)
        q_positive = q > 0.0
        p = np.where(q_positive, -p, p)
        q = np.abs(q)
        r_prev = e
        e_prev = rat
        accept = (
            (np.abs(p) < np.abs(0.5 * q * r_prev))
            & (p > q * (a - xf))
            & (p < q * (b - xf))
        )
        parabolic = active & cond_e & accept
        golden = active & ~parabolic

        with np.errstate(invalid="ignore", divide="ignore"):
            rat_par = (p + 0.0) / q
        x_par = xf + rat_par
        si_near = np.sign(xm - xf) + ((xm - xf) == 0)
        near = ((x_par - a) < tol2) | ((b - x_par) < tol2)
        rat_par = np.where(near, tol1 * si_near, rat_par)
        rat = np.where(parabolic, rat_par, rat)
        e = np.where(parabolic, e_prev, e)

        # ── golden-section step ──
        e_golden = np.where(xf >= xm, a - xf, b - xf)
        rat_golden = golden_mean * e_golden
        e = np.where(golden, e_golden, e)
        rat = np.where(golden, rat_golden, rat)

        si = np.sign(rat) + (rat == 0)
        x_new = xf + si * np.maximum(np.abs(rat), tol1)
        x_eval = np.where(active, x_new, xf)
        fu_new = evaluate(x_eval)
        fu = np.where(active, fu_new, fu)
        num = num + active.astype(np.int64)

        # ── bracket update ──
        better = fu_new <= fx
        a_better = np.where(x_new >= xf, xf, a)
        b_better = np.where(x_new < xf, xf, b)
        nfc_better = xf
        fnfc_better = fx
        xf_better = x_new
        fx_better = fu_new

        a_worse = np.where(x_new < xf, x_new, a)
        b_worse = np.where(x_new >= xf, x_new, b)
        cond1 = (fu_new <= fnfc) | (nfc == xf)
        cond2 = (fu_new <= ffulc) | (fulc == xf) | (fulc == nfc)
        fulc_worse = np.where(cond1, nfc, np.where(cond2, x_new, fulc))
        ffulc_worse = np.where(cond1, fnfc, np.where(cond2, fu_new, ffulc))
        nfc_worse = np.where(cond1, x_new, nfc)
        fnfc_worse = np.where(cond1, fu_new, fnfc)

        a = np.where(active, np.where(better, a_better, a_worse), a)
        b = np.where(active, np.where(better, b_better, b_worse), b)
        fulc = np.where(
            active, np.where(better, nfc, fulc_worse), fulc
        )
        ffulc = np.where(
            active, np.where(better, fnfc, ffulc_worse), ffulc
        )
        nfc = np.where(active, np.where(better, nfc_better, nfc_worse), nfc)
        fnfc = np.where(active, np.where(better, fnfc_better, fnfc_worse), fnfc)
        xf = np.where(active, np.where(better, xf_better, xf), xf)
        fx = np.where(active, np.where(better, fx_better, fx), fx)

        xm = np.where(active, 0.5 * (a + b), xm)
        tol1 = np.where(active, sqrt_eps * np.abs(xf) + xatol / 3.0, tol1)
        tol2 = np.where(active, 2.0 * tol1, tol2)

        converged = active & ~(np.abs(xf - xm) > (tol2 - 0.5 * (b - a)))
        active = active & ~converged
        hit_max = active & (num >= maxiter)
        flag[hit_max] = 1
        active = active & ~hit_max

    nan = np.isnan(xf) | np.isnan(fx) | np.isnan(fu)
    flag[nan] = 2
    success = flag == 0
    return xf, fx, success


def fit_all_rays(
    profiles: np.ndarray,
    m_est: float,
    max_dist: float,
    mask_boundary: float = 4.5,
    num_grid: int = 50,
    grid_width: float = 2.0,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit *m* independently to each half-ray profile.

    Returns
    -------
    m : ndarray (num_rays,)
    mse : ndarray (num_rays,)
    success : ndarray (num_rays,) bool
    """
    n_rays, n_samples = profiles.shape
    t_pos = np.linspace(0, max_dist, n_samples)

    m = np.full(n_rays, np.nan, dtype=np.float64)
    mse_arr = np.full(n_rays, np.nan, dtype=np.float64)
    success = np.full(n_rays, False)

    finite = np.isfinite(profiles)
    enough = np.sum(finite, axis=1) >= 10

    if not (finite.all() and enough.all()):
        # Rare degenerate case: use the original per-ray implementation.
        for i in range(n_rays):
            res = fit_half_ray(
                t_pos, profiles[i], m_est, mask_boundary, num_grid, grid_width, sigma
            )
            m[i] = res["m_fitted"]
            mse_arr[i] = res["mse"]
            success[i] = res["success"]
        return m, mse_arr, success

    m_grid = np.linspace(m_est / grid_width, m_est * grid_width, num_grid)
    abs_t = np.abs(t_pos)  # (S,)
    m_col = m_grid[:, None]  # (K, 1)

    # Prefix length of the mask for each grid m (t is increasing).
    prefix_n = np.sum(abs_t[None, :] <= mask_boundary * m_col, axis=1)  # (K,)
    widths = np.arange(n_samples)[None, :] < prefix_n[:, None]  # (K, S)

    # finder_soft_template for every (grid m, sample) at once.
    u = abs_t[None, :] / m_col
    s = sigma / m_col
    sqrt2 = np.sqrt(2.0)
    template = 0.5 * erfc(-(u - 1.5) / (s * sqrt2))
    template -= 0.5 * erfc(-(u - 2.5) / (s * sqrt2))
    template += 0.5 * erfc(-(u - 3.5) / (s * sqrt2))  # (K, S)

    diff = (template[None, :, :] - profiles[:, None, :]) ** 2  # (R, K, S)
    losses = np.full((n_rays, num_grid), np.inf, dtype=np.float64)
    ok = prefix_n >= 3
    losses[:, ok] = (
        np.sum(diff[:, ok, :], axis=2, where=widths[ok, :]) / prefix_n[ok]
    )

    rows = np.arange(n_rays)
    best_idx = np.argmin(losses, axis=1)
    best_loss = losses[rows, best_idx]
    m_best = m_grid[best_idx]

    step = m_grid[1] - m_grid[0] if num_grid > 1 else m_est * 0.05

    # Rays whose grid fit is unusable mirror fit_half_ray's early exits.
    failed_grid = ~np.isfinite(best_loss)
    m[failed_grid] = m_est
    mse_arr[failed_grid] = best_loss[failed_grid]

    refine = np.nonzero(~failed_grid)[0]
    if refine.size:
        n_refine = np.sum(
            abs_t[None, :] <= mask_boundary * m_best[refine, None], axis=1
        )
        small = n_refine < 3
        for i, n_i in zip(refine[small], n_refine[small]):
            m[i] = m_best[i]
            mse_arr[i] = best_loss[i]
            success[i] = True

        big = refine[~small]
        if big.size:
            n_big = n_refine[~small]
            width = int(n_big.max())
            t_batch = np.zeros((big.size, width), dtype=np.float64)
            p_batch = np.zeros((big.size, width), dtype=np.float64)
            for k, i in enumerate(big):
                n_i = int(n_big[k])
                t_batch[k, :n_i] = t_pos[:n_i]
                p_batch[k, :n_i] = profiles[i, :n_i]
            x_fit, fun_fit, ok_fit = _minimize_bounded_batch(
                t_batch,
                p_batch,
                n_big.astype(np.float64),
                m_best[big] - step,
                m_best[big] + step,
                sigma,
            )
            m[big] = x_fit
            mse_arr[big] = fun_fit
            success[big] = ok_fit

    return m, mse_arr, success


# ── Public API ────────────────────────────────────────────────────────────────


def fit_finder_ray(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    num_rays: int = 36,
    num_samples: int = 120,
    roi_scale: float = 1.5,
    sigma: float = 1.0,
    max_gap: int = 1,
    distance_threshold: float = 0.1,
    min_concentration_ratio: float = 0.5,
    refine_joint: bool = False,
) -> RayFitResult:
    """Fit a finder pattern's four edge corners from ray intensity profiles.

    Pipeline::

        1. normalize_roi_intensities   → [0, 1] ROI
        2. sample_ray_profiles         → profiles (N_rays × N_samples)
        3. fit_all_rays                → per-ray *m* estimates
        4. compute_boundary_points     → 3.5m boundary points
        5. fit_finder_edges            → Phase 0–2 clustering + TLS
        6. concentration-ratio check   → reject non-finder patterns
        7. assign_points               → tie-broken point-to-cluster assignment
        8. [optional] refine_finder_edges_joint  → LM-refined edge lines
        9. compute_corners             → 4 (x, y) outer corners
        10. compute score              → fraction_valid × cluster quality

    Parameters
    ----------
    roi : ndarray (H, W)
        Grayscale ROI cutout (uint8).
    center_xy : ndarray (2,)
        Estimated finder centre in ROI-local (x=col, y=row) coordinates.
    m_est : float
        Estimated module pitch in pixels (from cluster width / 7).
    num_rays : int
        Number of equally-spaced ray directions.
    num_samples : int
        Samples per ray.
    roi_scale : float
        Ignored; for interface compatibility.  ROI scaling is done by
        the caller via ``cluster_to_bbox``.
    sigma : float
        Edge softness for template fitting.
    max_gap : int
        Max cyclic gap in Phase 0 clustering.
    distance_threshold : float
        Single-linkage distance threshold.
    min_concentration_ratio : float
        If the top-4 edge clusters contain fewer than this fraction of all
        valid boundary points, the candidate is rejected.
    refine_joint : bool
        If ``True``, run the optional joint projective refinement.

    Returns
    -------
    RayFitResult
        ``corners`` are in ``(x, y)`` ROI-local coordinates.
        ``valid`` is ``False`` when fewer than 4 valid edge clusters are
        found or the concentration check fails.
    """
    _ = roi_scale  # already accounted for by the caller

    # ── 1. Normalize ROI ──
    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est)

    # ── 2. Sample rays ──
    profiles, max_dist = sample_ray_profiles(
        roi, center_xy[0], center_xy[1],
        num_rays=num_rays, num_samples=num_samples, ray_length=1.0,
    )
    span = bright_val - dark_val
    if span < 1.0:
        span = 1.0
    profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)

    # ── 3. Per-ray m fitting ──
    m, mse, success = fit_all_rays(
        profiles_norm, m_est, max_dist, sigma=sigma,
    )

    # ── 4. Boundary points ──
    theta_rad = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    bp = compute_boundary_points(center_xy, m, theta_rad, pitch_constant=3.5)

    # ── 5. Edge clustering ──
    edge_result = fit_finder_edges(
        bp, max_gap=max_gap, distance_threshold=distance_threshold, k=4,
    )
    if len(edge_result.clusters) < 4:
        return RayFitResult(
            corners=np.full((4, 2), np.nan, dtype=np.float64),
            score=0.0,
            valid=False,
        )

    # ── 6. Concentration-ratio check ──
    n_points = len(edge_result.points)
    top4_support = set()
    for ec in edge_result.clusters:
        top4_support.update(ec.support.tolist())
    concentration = len(top4_support) / max(n_points, 1)
    if concentration < min_concentration_ratio:
        return RayFitResult(
            corners=np.full((4, 2), np.nan, dtype=np.float64),
            score=0.0,
            valid=False,
        )

    # ── 7. Assignment ──
    assignment = assign_points(edge_result.clusters, n_points)

    # ── 8. Optional joint refinement ──
    if refine_joint:
        try:
            half_dirs = np.column_stack(
                [np.cos(theta_rad), np.sin(theta_rad)]
            )
            s_samples = np.linspace(0, max_dist, num_samples)
            refined_clusters, _ = refine_finder_edges_joint(
                edge_result.clusters,
                center_xy,
                profiles_norm,
                half_dirs,
                s_samples,
                sigma=sigma,
            )
            # Reorder to L, R, T, B
            l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(refined_clusters)
            ordered = [
                refined_clusters[l_idx],
                refined_clusters[r_idx],
                refined_clusters[t_idx],
                refined_clusters[b_idx],
            ]
        except Exception:
            l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(edge_result.clusters)
            ordered = [
                edge_result.clusters[l_idx],
                edge_result.clusters[r_idx],
                edge_result.clusters[t_idx],
                edge_result.clusters[b_idx],
            ]
    else:
        l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(edge_result.clusters)
        ordered = [
            edge_result.clusters[l_idx],
            edge_result.clusters[r_idx],
            edge_result.clusters[t_idx],
            edge_result.clusters[b_idx],
        ]

    # ── 9. Compute corners ──
    from qr_reader.detector.edge_fitting import thetarho_to_homogeneous_line

    ell_L = thetarho_to_homogeneous_line(
        float(np.arctan2(ordered[0].normal[1], ordered[0].normal[0])),
        float(ordered[0].rho),
    )
    ell_R = thetarho_to_homogeneous_line(
        float(np.arctan2(ordered[1].normal[1], ordered[1].normal[0])),
        float(ordered[1].rho),
    )
    ell_T = thetarho_to_homogeneous_line(
        float(np.arctan2(ordered[2].normal[1], ordered[2].normal[0])),
        float(ordered[2].rho),
    )
    ell_B = thetarho_to_homogeneous_line(
        float(np.arctan2(ordered[3].normal[1], ordered[3].normal[0])),
        float(ordered[3].rho),
    )
    p_LT, p_RT, p_RB, p_LB = compute_corners(ell_L, ell_R, ell_T, ell_B)
    corners = np.array([p_LT, p_RT, p_RB, p_LB], dtype=np.float64)

    # ── 10. Score ──
    frac_valid = float(int(np.sum(success & np.isfinite(m)))) / float(num_rays)
    mean_sigma_ratio = float(
        np.mean([ec.sigma_ratio for ec in ordered])
    ) if ordered else 1.0
    score = frac_valid * (1.0 - min(mean_sigma_ratio, 1.0))

    return RayFitResult(
        corners=corners,
        score=float(score),
        valid=True,
    )
