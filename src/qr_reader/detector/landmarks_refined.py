"""Optional refined-landmark extraction for a better homography estimate.

The default pipeline fits a homography from only the 12 **outer** finder corners
(three finder patterns × four corners).  This module adds two sources of extra
correspondences so a downstream homography can be over-determined and spread
across the whole symbol:

* **Inner finder corners** — the white-ring square (offset 1..6 of the 7×7
  finder), found by reusing the same ray-profile machinery that locates the
  outer corners, but placing the boundary points at 2.5 modules from the finder
  centre instead of 3.5.

* **Alignment-pattern centres** — the small 5×5 patterns interior to the symbol
  (versions ≥ 2), found by normalized template matching around the position
  predicted by the initial homography.  Each yields one point correspondence
  with a correlation-based confidence score.

Everything works in ``(x, y)`` image coordinates (matching the homography
module's convention), and canonical grid coordinates are likewise ``(x, y)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from qr_reader.decoder.tables import VERSIONS
from qr_reader.detector.edge_fitting import (
    compute_boundary_points,
    compute_corners,
    fit_finder_edges,
    thetarho_to_homogeneous_line,
    _reorder_to_standard,
)
from qr_reader.detector.homography import (
    estimate_homography_dlt,
    project_points,
    project_points_with_jac,
    ransac_homography,
    refine_homography_lm,
)
from qr_reader.detector.ray_fit import (
    fit_all_rays,
    normalize_roi_intensities,
    sample_ray_profiles,
)


# ---------------------------------------------------------------------------
# Inner finder corners
# ---------------------------------------------------------------------------


def fit_finder_inner_corners(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    num_rays: int = 36,
    num_samples: int = 120,
    sigma: float = 1.0,
    max_gap: int = 1,
    distance_threshold: float = 0.1,
) -> np.ndarray | None:
    """Fit the finder pattern's inner (white-ring) corners from ray profiles.

    Mirrors :func:`qr_reader.detector.ray_fit.fit_finder_ray`, but extracts the
    boundary at 2.5 modules from the centre — the white-ring's outer edge — which
    corresponds to the inner square at grid offset 1..6.

    Returns the four corners in ``(x, y)`` ROI-local coordinates, ordered
    ``[TL, TR, BR, BL]``, or ``None`` if fewer than four edge clusters are found.
    """
    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est)

    profiles, max_dist = sample_ray_profiles(
        roi, center_xy[0], center_xy[1],
        num_rays=num_rays, num_samples=num_samples, ray_length=1.0,
    )
    span = bright_val - dark_val
    if span < 1.0:
        span = 1.0
    profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)

    m, _mse, _success = fit_all_rays(profiles_norm, m_est, max_dist, sigma=sigma)

    theta_rad = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    bp = compute_boundary_points(center_xy, m, theta_rad, pitch_constant=2.5)

    edge_result = fit_finder_edges(
        bp, max_gap=max_gap, distance_threshold=distance_threshold, k=4,
    )
    if len(edge_result.clusters) < 4:
        return None

    l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(edge_result.clusters)
    ordered = [
        edge_result.clusters[l_idx],
        edge_result.clusters[r_idx],
        edge_result.clusters[t_idx],
        edge_result.clusters[b_idx],
    ]

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
    # compute_corners already returns [TL, TR, BR, BL] in (x, y).
    return np.array([p_LT, p_RT, p_RB, p_LB], dtype=np.float64)


# ---------------------------------------------------------------------------
# Alignment-pattern template matching
# ---------------------------------------------------------------------------


def alignment_pattern_template(
    module_px: float,
    pad_modules: float = 1.0,
    sigma: float | None = None,
) -> np.ndarray:
    """Render a synthetic 5×5 alignment-pattern template at *module_px*.

    The template covers ``5 + 2*pad_modules`` modules so it includes a white
    quiet ring around the dark border, giving the matcher a distinctive
    signature.  Values are in ``[0, 1]`` (0 = dark, 1 = bright).
    """
    half_modules = 2.5 + pad_modules
    n = int(np.ceil(2.0 * half_modules * module_px))
    if n % 2 == 0:
        n += 1

    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2.0
    u = np.maximum(np.abs(xs - c), np.abs(ys - c)) / module_px

    # dark centre (u < 0.5), white ring (0.5..1.5), dark border (1.5..2.5),
    # white outside (u >= 2.5).
    t = np.where(u < 0.5, 0.0, 1.0)
    t = np.where((u >= 1.5) & (u < 2.5), 0.0, t)

    if sigma is not None and sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        t = gaussian_filter(t, sigma=sigma)

    return t.astype(np.float32)


def _parabolic_peak(res: np.ndarray, mx: int, my: int) -> tuple[float, float]:
    """Sub-pixel refinement of a correlation peak at integer ``(mx, my)``."""
    h, w = res.shape
    dx = 0.0
    dy = 0.0
    if 0 < mx < w - 1:
        f = res[my, mx - 1 : mx + 2].astype(np.float64)
        denom = (f[0] - 2.0 * f[1] + f[2])
        if abs(denom) > 1e-12:
            dx = 0.5 * (f[0] - f[2]) / denom
    if 0 < my < h - 1:
        f = res[my - 1 : my + 2, mx].astype(np.float64)
        denom = (f[0] - 2.0 * f[1] + f[2])
        if abs(denom) > 1e-12:
            dy = 0.5 * (f[0] - f[2]) / denom
    return float(dx), float(dy)


def match_alignment_pattern(
    img_gray: np.ndarray,
    center_xy: np.ndarray,
    module_px: float,
    search_radius_px: float | None = None,
) -> tuple[np.ndarray, float]:
    """Locate an alignment-pattern centre via normalized template matching.

    Parameters
    ----------
    img_gray : ndarray (H, W)
        Grayscale image.
    center_xy : ndarray (2,)
        Initial centre estimate in ``(x, y)`` image coordinates (e.g. projected
        through an initial homography).
    module_px : float
        Local module pitch in pixels at the pattern location.
    search_radius_px : float or None
        Half-extent of the search window (defaults to 2 modules).

    Returns
    -------
    (image_xy, confidence) where ``image_xy`` is the sub-pixel centre in
    ``(x, y)`` and ``confidence`` is the TM_CCOEFF_NORMED correlation in
    ``[0, 1]``.
    """
    if search_radius_px is None:
        search_radius_px = 2.0 * module_px

    template = alignment_pattern_template(module_px)
    th, tw = template.shape
    ctr_x = (tw - 1) / 2.0
    ctr_y = (th - 1) / 2.0

    half = int(np.ceil(search_radius_px)) + max(ctr_x, ctr_y).__int__() + 1
    cx, cy = float(center_xy[0]), float(center_xy[1])

    h, w = img_gray.shape
    x0 = int(round(cx - half))
    x1 = int(round(cx + half)) + 1
    y0 = int(round(cy - half))
    y1 = int(round(cy + half)) + 1

    # Pad the crop so the template can be matched even near image borders.
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    window = img_gray[sy0:sy1, sx0:sx1].astype(np.float32)
    if pad_l or pad_t or pad_r or pad_b:
        window = np.pad(
            window,
            ((pad_t, pad_b), (pad_l, pad_r)),
            mode="edge",
        )

    if window.shape[0] < th or window.shape[1] < tw:
        return np.array([cx, cy]), 0.0

    res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)

    mx, my = int(max_loc[0]), int(max_loc[1])
    dx, dy = _parabolic_peak(res, mx, my)

    image_x = (x0 - pad_l) + (mx + dx) + ctr_x
    image_y = (y0 - pad_t) + (my + dy) + ctr_y

    confidence = float(max(0.0, min(1.0, max_val)))
    return np.array([image_x, image_y]), confidence


# ---------------------------------------------------------------------------
# Alignment-pattern canonical positions
# ---------------------------------------------------------------------------


def alignment_centers_grid(version: int) -> list[tuple[float, float]]:
    """Return the canonical ``(x, y)`` grid coordinates of alignment centres.

    Excludes the three corner patterns that coincide with the finder patterns.
    Returns an empty list for version 1 (no alignment patterns).
    """
    positions = VERSIONS[version].alignment_positions
    if not positions:
        return []

    centers: list[tuple[float, float]] = []
    for row_pos in positions:
        for col_pos in positions:
            # Skip the three corners occupied by finder patterns.
            first = positions[0]
            last = positions[-1]
            if (row_pos == first and col_pos == first) or (
                row_pos == first and col_pos == last
            ) or (row_pos == last and col_pos == first):
                continue
            centers.append((col_pos + 0.5, row_pos + 0.5))
    return centers


# ---------------------------------------------------------------------------
# Weighted homography fitting
# ---------------------------------------------------------------------------


def refine_homography_lm_weighted(
    H_init: np.ndarray,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    weights: np.ndarray | None = None,
    loss: str = "linear",
) -> np.ndarray:
    """Levenberg-Marquardt homography refinement with per-point weights.

    ``weights`` scales each point's reprojection residual.  When ``None`` this
    is equivalent to :func:`qr_reader.detector.homography.refine_homography_lm`.
    """
    if weights is None:
        weights = np.ones(len(src_xy), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.mean()), 1e-12)
    sqrt_w = np.sqrt(w)
    w_rep = np.repeat(sqrt_w, 2)[:, None]  # (2n, 1)

    x0 = H_init.ravel()[:8].copy()

    def fun(params: np.ndarray) -> np.ndarray:
        H = np.eye(3)
        H.ravel()[:8] = params
        projected, _ = project_points_with_jac(H, src_xy)
        return ((projected - dst_xy) * w_rep).ravel()

    def jac(params: np.ndarray) -> np.ndarray:
        H = np.eye(3)
        H.ravel()[:8] = params
        _, J = project_points_with_jac(H, src_xy)
        return J * w_rep

    result = least_squares(
        fun, x0, jac=jac, method="lm", loss=loss, max_nfev=2000,
    )
    H_refined = np.eye(3)
    H_refined.ravel()[:8] = result.x
    return H_refined


@dataclass
class Correspondence:
    """One grid↔image point pair used to fit a homography."""

    grid_xy: np.ndarray      # (2,) canonical (x, y)
    image_xy: np.ndarray     # (2,) image (x, y)
    confidence: float        # in [0, 1]
    label: str               # e.g. "outer-TL-0", "inner-TR-2", "align-1"


def fit_homography_from_correspondences(
    correspondences: list[Correspondence],
    weighted: bool = False,
    robust: bool = False,
) -> tuple[np.ndarray, float]:
    """Fit a homography from a list of correspondences.

    Always runs DLT + LM.  When *weighted* is true the LM stage down-weights
    low-confidence points; the DLT initialisation is unweighted.  When *robust*
    is true a RANSAC stage rejects outlier correspondences before the final
    least-squares refit.
    """
    src = np.array([c.grid_xy for c in correspondences], dtype=np.float64)
    dst = np.array([c.image_xy for c in correspondences], dtype=np.float64)

    if robust and len(correspondences) >= 8:
        H, inliers = ransac_homography(src, dst, threshold=2.0, iters=2000)
        if np.sum(inliers) >= 4:
            src = src[inliers]
            dst = dst[inliers]
            if weighted:
                w = np.array(
                    [c.confidence for c, k in zip(correspondences, inliers) if k],
                    dtype=np.float64,
                )
            else:
                w = None
            H = refine_homography_lm_weighted(H, src, dst, weights=w) if weighted else refine_homography_lm(H, src, dst, loss="linear")
            proj = project_points(H, src)
            err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
            return H, err

    H = estimate_homography_dlt(src, dst)
    if weighted:
        w = np.array([c.confidence for c in correspondences], dtype=np.float64)
        try:
            H = refine_homography_lm_weighted(H, src, dst, weights=w)
        except Exception:
            H = refine_homography_lm(H, src, dst, loss="linear")
    else:
        H = refine_homography_lm(H, src, dst, loss="linear")

    proj = project_points(H, src)
    err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
    return H, err


# ---------------------------------------------------------------------------
# High-level: build a refined homography for one (triplet, version)
# ---------------------------------------------------------------------------


def finder_grid_coords_xy(N: int) -> dict[str, dict[str, np.ndarray]]:
    """Canonical outer + inner finder corners in ``(x, y)``, [TL, TR, BR, BL]."""
    outer = {
        "TL": np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64),
        "TR": np.array([[N - 7, 0], [N, 0], [N, 7], [N - 7, 7]], dtype=np.float64),
        "BL": np.array([[0, N - 7], [7, N - 7], [7, N], [0, N]], dtype=np.float64),
    }
    inner = {
        "TL": np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float64),
        "TR": np.array([[N - 6, 1], [N - 1, 1], [N - 1, 6], [N - 6, 6]], dtype=np.float64),
        "BL": np.array([[1, N - 6], [6, N - 6], [6, N - 1], [1, N - 1]], dtype=np.float64),
    }
    return {"outer": outer, "inner": inner}


def _canonicalize_xy(
    corners_xy: np.ndarray, right: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Order four finder corners as [TL, TR, BR, BL] using a triplet basis."""
    centre = corners_xy.mean(axis=0)
    uv = corners_xy - centre
    u = uv @ right
    v = uv @ down
    idx_tl = int(np.argmin(u + v))
    idx_tr = int(np.argmax(u - v))
    idx_br = int(np.argmax(u + v))
    idx_bl = int(np.argmin(u - v))
    return corners_xy[np.array([idx_tl, idx_tr, idx_br, idx_bl])]


def _triplet_axes_xy(
    global_outer_xy: dict, triplet
) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, down) unit vectors in (x, y) for a triplet."""
    tl = global_outer_xy[triplet.top_left_idx].mean(axis=0)
    tr = global_outer_xy[triplet.top_right_idx].mean(axis=0)
    bl = global_outer_xy[triplet.bottom_left_idx].mean(axis=0)
    right = tr - tl
    right = right / (float(np.linalg.norm(right)) + 1e-12)
    down = bl - tl
    down = down / (float(np.linalg.norm(down)) + 1e-12)
    return right, down


def _local_module_px(H: np.ndarray, gx: float, gy: float) -> float:
    p0 = project_points(H, np.array([[gx, gy]]))[0]
    px = project_points(H, np.array([[gx + 1.0, gy]]))[0]
    py = project_points(H, np.array([[gx, gy + 1.0]]))[0]
    return float((np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) / 2.0)


def build_refined_homography(
    img_gray: np.ndarray,
    H_base: np.ndarray,
    triplet,
    fps: list,
    score_map: dict | None = None,
    version: int = 1,
    include_inner: bool = True,
    include_align: bool = True,
    robust: bool = False,
    weighted: bool = False,
) -> tuple[np.ndarray, dict]:
    """Build a refined homography from outer + inner + alignment landmarks.

    Parameters
    ----------
    img_gray : grayscale image.
    H_base : initial homography (from the 12 outer corners) used to seed the
        alignment-pattern search.
    triplet : Triplet naming the three finder patterns.
    fps : list of :class:`FinderPattern`.  Outer and (when populated) inner
        corners are read directly from these — inner corners are produced by the
        finder fit itself, so this step adds no re-fitting.
    score_map : optional per-cluster quality scores for outer-corner confidence.
    version : estimated QR version.
    include_inner / include_align : toggle the two extra landmark sources.
    robust / weighted : passed to :func:`fit_homography_from_correspondences`.

    Returns
    -------
    (H, stats) where *stats* reports ``n_outer``, ``n_inner``, ``n_align``,
    ``n_corr`` and the mean reprojection error ``err``.
    """
    score_map = score_map or {}
    fp_map = {fp.cluster_idx: fp for fp in fps}
    global_outer_xy = {ci: fp.outer_corners[:, ::-1] for ci, fp in fp_map.items()}

    N = 4 * version + 17
    right, down = _triplet_axes_xy(global_outer_xy, triplet)
    grid = finder_grid_coords_xy(N)
    idx = {
        "TL": triplet.top_left_idx,
        "TR": triplet.top_right_idx,
        "BL": triplet.bottom_left_idx,
    }

    corr: list[Correspondence] = []
    for role in ["TL", "TR", "BL"]:
        ci = idx[role]
        corners = _canonicalize_xy(global_outer_xy[ci], right, down)
        g = grid["outer"][role]
        for k in range(4):
            corr.append(
                Correspondence(
                    grid_xy=g[k], image_xy=corners[k],
                    confidence=float(score_map.get(ci, 1.0)),
                    label=f"outer-{role}-{k}",
                )
            )

    if include_inner:
        for role in ["TL", "TR", "BL"]:
            ci = idx[role]
            inner_rc = fp_map[ci].inner_corners
            if inner_rc is None:
                continue
            inner = _canonicalize_xy(inner_rc[:, ::-1], right, down)
            g = grid["inner"][role]
            for k in range(4):
                corr.append(
                    Correspondence(
                        grid_xy=g[k], image_xy=inner[k], confidence=0.9,
                        label=f"inner-{role}-{k}",
                    )
                )

    if include_align:
        for (gx, gy) in alignment_centers_grid(version):
            center0 = project_points(H_base, np.array([[gx, gy]]))[0]
            module_px = _local_module_px(H_base, gx, gy)
            if module_px < 0.5:
                continue
            image_xy, conf = match_alignment_pattern(img_gray, center0, module_px)
            corr.append(
                Correspondence(
                    grid_xy=np.array([gx, gy]), image_xy=image_xy, confidence=conf,
                    label=f"align-{gx:.1f}-{gy:.1f}",
                )
            )

    H, err = fit_homography_from_correspondences(
        corr, weighted=weighted, robust=robust
    )
    stats = {
        "n_outer": sum(1 for c in corr if c.label.startswith("outer")),
        "n_inner": sum(1 for c in corr if c.label.startswith("inner")),
        "n_align": sum(1 for c in corr if c.label.startswith("align")),
        "n_corr": len(corr),
        "err": err,
    }
    return H, stats
