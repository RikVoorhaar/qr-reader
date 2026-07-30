"""Finder-pattern edge fitting from per-ray boundary points (plan v3).

Given N boundary points in cyclic order (one per ray direction, computed
from per-ray module-pitch estimates at 3.5 modules from the finder centre),
estimate the finder pattern's 4 edges:

Phase 0
    Form N initial clusters, one per adjacent point pair ``(i, i+1 mod N)``.
    For every pair of initial clusters within ``MAX_GAP`` cyclic steps,
    compute the TLS degeneracy ratio sigma2/sigma1 on the union of their
    points.  Pairs on the same edge are near-colinear (low ratio); pairs on
    different edges are not (high ratio).  Pairs further apart than
    ``MAX_GAP`` get distance 1.0.

Phase 1
    Single-linkage agglomerative clustering (sklearn) on the precomputed
    distance matrix with a fixed ``distance_threshold``.  Single linkage is
    required because distant same-edge pairs have distance 1.0 by
    construction; only the chain of adjacent low-distance pairs connects
    them.

Phase 2
    Keep the 4 largest clusters, fit a TLS line to each cluster's support
    set (union of member-pair points), and tie-break shared points to the
    cluster whose full-support fit has the lowest sigma2/sigma1.

See ``docs/plan-finder-edge-fitting.md`` for the full design history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

# ── Parameters (plan v3) ─────────────────────────────────────────────────────
MAX_GAP = 1             # Max cyclic index gap between mergeable pairs
DISTANCE_THRESHOLD = 0.1  # sigma2/sigma1 above which pairs are incomparable
PITCH_CONSTANT = 3.5    # Finder outer boundary in module units
FAR_DISTANCE = 1.0      # Distance assigned to pairs beyond MAX_GAP


def compute_boundary_points(
    center_xy: np.ndarray,
    m: np.ndarray,
    theta_rad: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
) -> np.ndarray:
    """One boundary point per half-ray direction.

    ``theta_rad`` must be in ``[0, 2π)`` with one entry per half-ray.
    Returns ``(k, 2)`` points where ``k = len(theta_rad)``, NaN rows for
    failed fits.
    """
    k = len(theta_rad)
    points = np.full((k, 2), np.nan, dtype=np.float64)
    for i in range(k):
        if np.isfinite(m[i]):
            d = np.array([np.cos(theta_rad[i]), np.sin(theta_rad[i])])
            points[i] = center_xy + pitch_constant * m[i] * d
    return points


def tls_line(
    pts: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Total-least-squares line fit via SVD on centred points.

    Returns
    -------
    normal : ndarray (2,)
        Unit normal, oriented so that ``rho >= 0``.
    rho : float
        Signed distance of the line from the origin (non-negative).
    direction : ndarray (2,)
        Unit direction along the line.
    singular_values : ndarray (2,)
        ``(sigma1, sigma2)`` with ``sigma1 >= sigma2``.
    """
    mu = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - mu, full_matrices=False)
    normal = vt[1]
    direction = vt[0]
    rho = float(mu @ normal)
    if rho < 0:
        normal = -normal
        rho = -rho
    return normal, rho, direction, s


def sigma_ratio(pts: np.ndarray) -> float:
    """TLS degeneracy ratio sigma2/sigma1 for a set of >= 2 points."""
    _, _, _, s = tls_line(pts)
    if s[0] <= 1e-12:
        return FAR_DISTANCE
    return float(s[1] / s[0])


def _min_cyclic_gap(
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    orig_indices: np.ndarray,
    n_total: int,
) -> int:
    """Number of original boundary points strictly between two point pairs.

    The gap is the minimal cyclic distance (in original ray-index space)
    between any member of ``pair_a`` and any member of ``pair_b``, minus 1,
    clamped at 0 (pairs that share a point or are directly adjacent have
    gap 0).
    """
    best = n_total
    for u in pair_a:
        for v in pair_b:
            d = abs(int(orig_indices[u]) - int(orig_indices[v])) % n_total
            d = min(d, n_total - d)
            best = min(best, d)
    return max(best - 1, 0)


def build_pair_distance_matrix(
    points: np.ndarray,
    orig_indices: np.ndarray,
    n_total: int,
    max_gap: int = MAX_GAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 0 — pairwise sigma2/sigma1 distances between adjacent pairs.

    Parameters
    ----------
    points : ndarray (M, 2)
        Valid boundary points in cyclic order.
    orig_indices : ndarray (M,)
        Original ray index of each point (for cyclic-gap computation).
    n_total : int
        Total number of ray directions (N).
    max_gap : int
        Maximum number of missing original points allowed between two
        mergeable pairs.

    Returns
    -------
    distance_matrix : ndarray (M, M)
        Symmetric; ``FAR_DISTANCE`` beyond ``max_gap``, 0 on the diagonal.
    pairs : ndarray (M, 2)
        Point indices of each initial cluster: pair ``j = (j, j+1 mod M)``.
    """
    m = len(points)
    pairs = np.stack([np.arange(m), (np.arange(m) + 1) % m], axis=1)
    dist = np.full((m, m), FAR_DISTANCE, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for i in range(m):
        for j in range(i + 1, m):
            gap = _min_cyclic_gap(pairs[i], pairs[j], orig_indices, n_total)
            if gap > max_gap:
                continue
            union = np.unique(np.concatenate([pairs[i], pairs[j]]))
            dist[i, j] = dist[j, i] = sigma_ratio(points[union])
    return dist, pairs


def cluster_pairs(
    distance_matrix: np.ndarray,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Phase 1 — single-linkage agglomerative clustering on the matrix."""
    if len(distance_matrix) < 2:
        return np.zeros(len(distance_matrix), dtype=int)
    ac = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="single",
        distance_threshold=distance_threshold,
    )
    return ac.fit_predict(distance_matrix)


@dataclass
class EdgeCluster:
    """One fitted finder edge: TLS line + supporting points."""

    label: int                 # sklearn cluster label
    pair_indices: np.ndarray   # indices into the pairs array
    support: np.ndarray        # point indices (union of member pairs)
    normal: np.ndarray         # unit normal, rho >= 0
    rho: float                 # distance from origin
    direction: np.ndarray      # unit direction along the line
    sigma_ratio: float         # sigma2/sigma1 of the full-support TLS fit


def extract_top_clusters(
    labels: np.ndarray,
    pairs: np.ndarray,
    points: np.ndarray,
    k: int = 4,
) -> list[EdgeCluster]:
    """Phase 2 — the k largest clusters with TLS lines on their supports.

    Returned in descending size order.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    clusters: list[EdgeCluster] = []
    for rank in order[:k]:
        label = int(unique_labels[rank])
        member_pairs = np.flatnonzero(labels == label)
        support = np.unique(pairs[member_pairs].ravel())
        support_pts = points[support]
        normal, rho, direction, s = tls_line(support_pts)
        ratio = float(s[1] / s[0]) if s[0] > 1e-12 else FAR_DISTANCE
        clusters.append(
            EdgeCluster(
                label=label,
                pair_indices=member_pairs,
                support=support,
                normal=normal,
                rho=rho,
                direction=direction,
                sigma_ratio=ratio,
            )
        )
    return clusters


def assign_points(clusters: list[EdgeCluster], n_points: int) -> np.ndarray:
    """Tie-broken point-to-edge assignment.

    A point appearing in multiple clusters' support sets goes to the
    cluster with the lowest full-support sigma2/sigma1.  Unsupported
    points get -1.

    Returns
    -------
    assignment : ndarray (n_points,) int
        Index into ``clusters`` (not the sklearn label), or -1.
    """
    assignment = np.full(n_points, -1, dtype=int)
    best_ratio = np.full(n_points, np.inf)
    for ci, ec in enumerate(clusters):
        for p in ec.support:
            if ec.sigma_ratio < best_ratio[p]:
                best_ratio[p] = ec.sigma_ratio
                assignment[p] = ci
    return assignment


# ── Segment-refinement helpers (plan Step 2–3) ───────────────────────────────

from scipy.special import erfc
from scipy.optimize import least_squares, OptimizeResult


def _finder_template(
    t: np.ndarray,
    m: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """Soft finder-pattern intensity template along a radial ray."""
    u = np.abs(np.asarray(t, dtype=np.float64)) / m
    inv_s_sqrt2 = 1.0 / ((sigma / m) * np.sqrt(2.0))
    return (
        0.5 * erfc(-(u - 1.5) * inv_s_sqrt2)
        - 0.5 * erfc(-(u - 2.5) * inv_s_sqrt2)
        + 0.5 * erfc(-(u - 3.5) * inv_s_sqrt2)
    )


def _template_dm(
    t: np.ndarray,
    m: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """Derivative of ``_finder_template`` w.r.t. *m*."""
    abs_t = np.abs(np.asarray(t, dtype=np.float64))
    inv_factor = 1.0 / (sigma * np.sqrt(2.0))
    z1 = -(abs_t - 1.5 * m) * inv_factor
    z2 = -(abs_t - 2.5 * m) * inv_factor
    z3 = -(abs_t - 3.5 * m) * inv_factor
    prefactor = -1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return prefactor * (
        1.5 * np.exp(-z1 * z1)
        - 2.5 * np.exp(-z2 * z2)
        + 3.5 * np.exp(-z3 * z3)
    )


def assign_half_rays_to_segments(
    center_xy: np.ndarray,
    half_dirs: np.ndarray,
    segments: list[EdgeCluster],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each half-ray to the segment with the smallest positive t.

    Parameters
    ----------
    center_xy : ndarray (2,)
        Finder centre in ROI-local (x=col, y=row) coordinates.
    half_dirs : ndarray (N, 2)
        Unit direction vectors, one per half-ray.
    segments : list[EdgeCluster]
        The 4 fitted edge segments with ``.normal`` and ``.rho``.

    Returns
    -------
    segment_idx : ndarray (N,) int
        Index into ``segments`` (0..3); -1 if no segment has positive t.
    t_int : ndarray (N,) float
        Intersection distance from centre; NaN for unassigned half-rays.
    """
    n_rays = len(half_dirs)
    segment_idx = np.full(n_rays, -1, dtype=int)
    t_int = np.full(n_rays, np.nan, dtype=np.float64)

    for i in range(n_rays):
        d = half_dirs[i]
        best_t = np.inf
        best_idx = -1
        for si, seg in enumerate(segments):
            denom = seg.normal @ d
            if abs(denom) < 1e-12:
                continue
            t = (seg.rho - seg.normal @ center_xy) / denom
            if 0 < t < best_t:
                best_t = t
                best_idx = si
        if best_idx >= 0:
            segment_idx[i] = best_idx
            t_int[i] = best_t

    return segment_idx, t_int


def segment_refinement_residuals(
    x: np.ndarray,
    center_xy: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    t_samples: np.ndarray,
    segment_mask: np.ndarray,
    m_mask: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
    mask_boundary: float = 4.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Residual vector for one segment's LM refinement.

    Returns a 1-D float64 array of length ``N_assigned * N_S`` where
    ``N_S = len(t_samples)``.  Unmasked samples contribute
    ``template - profile``; masked samples contribute ``0.0``.

    The masking boundary uses ``m_mask`` (pre-computed from the initial
    estimate), NOT the current *m* value, to keep the residual smooth.
    """
    theta, rho = x[0], x[1]
    n = np.array([np.cos(theta), np.sin(theta)])

    assigned = np.flatnonzero(segment_mask)
    n_assigned = len(assigned)
    n_s = len(t_samples)
    residuals = np.zeros(n_assigned * n_s, dtype=np.float64)

    for k, i in enumerate(assigned):
        d = half_dirs[i]
        denom = n @ d
        if abs(denom) < 1e-12:
            continue
        t_int_i = (rho - n @ center_xy) / denom
        m_i = t_int_i / pitch_constant
        if m_i <= 0:
            continue
        template = _finder_template(t_samples, m_i, sigma)
        mask = np.abs(t_samples) <= mask_boundary * m_mask[i]
        row_start = k * n_s
        residuals[row_start:row_start + n_s][mask] = \
            template[mask] - half_profiles[i, mask]

    return residuals


def segment_refinement_jacobian(
    x: np.ndarray,
    center_xy: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    t_samples: np.ndarray,
    segment_mask: np.ndarray,
    m_mask: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
    mask_boundary: float = 4.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Jacobian of ``segment_refinement_residuals`` w.r.t. ``x = [theta, rho]``.

    Returns an ``(R, 2)`` float64 array where ``R = N_assigned * N_S``.
    Column 0 = ∂r/∂θ, column 1 = ∂r/∂ρ.
    """
    theta, rho = x[0], x[1]
    n_vec = np.array([np.cos(theta), np.sin(theta)])
    n_perp = np.array([-np.sin(theta), np.cos(theta)])

    assigned = np.flatnonzero(segment_mask)
    n_assigned = len(assigned)
    n_s = len(t_samples)
    Jac = np.zeros((n_assigned * n_s, 2), dtype=np.float64)

    n_dot_C = n_vec @ center_xy
    nperp_dot_C = n_perp @ center_xy

    for k, i in enumerate(assigned):
        d = half_dirs[i]
        nd = n_vec @ d
        if abs(nd) < 1e-12:
            continue
        nperp_d = n_perp @ d
        rho_minus_nC = rho - n_dot_C

        t_int_i = rho_minus_nC / nd
        m_i = t_int_i / pitch_constant
        if m_i <= 0:
            continue

        dm_drho = 1.0 / (pitch_constant * nd)
        dm_dtheta = (
            -(nperp_dot_C * nd)
            - (rho_minus_nC * nperp_d)
        ) / (pitch_constant * nd * nd)

        dT_dm = _template_dm(t_samples, m_i, sigma)
        mask = np.abs(t_samples) <= mask_boundary * m_mask[i]

        row_start = k * n_s
        Jac[row_start:row_start + n_s, 0][mask] = dT_dm[mask] * dm_dtheta
        Jac[row_start:row_start + n_s, 1][mask] = dT_dm[mask] * dm_drho

    return Jac


def check_segment_jacobian(
    x0: np.ndarray,
    center_xy: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    t_samples: np.ndarray,
    segment_mask: np.ndarray,
    m_mask: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
    mask_boundary: float = 4.5,
    sigma: float = 1.0,
    eps: float = 5e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compare analytical Jacobian to finite-difference approximation.

    Returns
    -------
    J_analytical : ndarray
    J_fd : ndarray
    max_rel_error : float
        Maximum relative error across non-negligible entries (|J| > 1e-8).
    """
    base = segment_refinement_residuals(
        x0, center_xy, half_profiles, half_dirs, t_samples,
        segment_mask, m_mask, pitch_constant, mask_boundary, sigma,
    )
    J = segment_refinement_jacobian(
        x0, center_xy, half_profiles, half_dirs, t_samples,
        segment_mask, m_mask, pitch_constant, mask_boundary, sigma,
    )
    J_fd = np.zeros_like(J)
    for k in range(2):
        h = np.zeros(2)
        h[k] = eps
        f_plus = segment_refinement_residuals(
            x0 + h, center_xy, half_profiles, half_dirs, t_samples,
            segment_mask, m_mask, pitch_constant, mask_boundary, sigma,
        )
        f_minus = segment_refinement_residuals(
            x0 - h, center_xy, half_profiles, half_dirs, t_samples,
            segment_mask, m_mask, pitch_constant, mask_boundary, sigma,
        )
        J_fd[:, k] = (f_plus - f_minus) / (2.0 * eps)

    denom = np.maximum(np.abs(J), 1e-12)
    rel_errors = np.abs(J - J_fd) / denom
    significant = np.abs(J) > 1e-8
    if np.any(significant):
        max_err = float(np.max(rel_errors[significant]))
    else:
        max_err = float(np.max(rel_errors))
    return J, J_fd, max_err


@dataclass
class EdgeFitResult:
    """Full output of the edge-fitting pipeline for one finder candidate."""

    valid_indices: np.ndarray        # (M,) original ray indices of valid points
    points: np.ndarray              # (M, 2) valid boundary points, cyclic order
    pairs: np.ndarray               # (M, 2) initial point-pair clusters
    distance_matrix: np.ndarray     # (M, M) sigma2/sigma1 distances
    labels: np.ndarray              # (M,) sklearn cluster label per pair
    clusters: list[EdgeCluster] = field(default_factory=list)
    assignment: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def fit_finder_edges(
    boundary_points: np.ndarray,
    max_gap: int = MAX_GAP,
    distance_threshold: float = DISTANCE_THRESHOLD,
    k: int = 4,
) -> EdgeFitResult:
    """Run Phases 0–2 on a cyclic array of boundary points (NaN allowed).

    Parameters
    ----------
    boundary_points : ndarray (N, 2)
        One point per ray direction in cyclic order; NaN rows are skipped.

    Returns
    -------
    EdgeFitResult
        ``clusters`` is empty when fewer than 4 valid points exist.
    """
    boundary_points = np.asarray(boundary_points, dtype=np.float64)
    n_total = len(boundary_points)
    valid = np.all(np.isfinite(boundary_points), axis=1)
    valid_indices = np.flatnonzero(valid)
    points = boundary_points[valid]
    m = len(points)

    if m < 4:
        return EdgeFitResult(
            valid_indices=valid_indices,
            points=points,
            pairs=np.empty((0, 2), dtype=int),
            distance_matrix=np.empty((0, 0)),
            labels=np.empty(0, dtype=int),
            clusters=[],
            assignment=np.full(m, -1, dtype=int),
        )

    dist, pairs = build_pair_distance_matrix(
        points, valid_indices, n_total, max_gap=max_gap
    )
    labels = cluster_pairs(dist, distance_threshold=distance_threshold)
    clusters = extract_top_clusters(labels, pairs, points, k=k)
    assignment = assign_points(clusters, m)
    return EdgeFitResult(
        valid_indices=valid_indices,
        points=points,
        pairs=pairs,
        distance_matrix=dist,
        labels=labels,
        clusters=clusters,
        assignment=assignment,
    )


# ── Projective 4-line refinement helpers (plan-projective-refinement.md Phases 1–2)

# Distances from the projective center to the finder-pattern transitions, in units
# of the outer-edge distance (3.5 modules).  These are the physically standard
# positions: 1.5m, 2.5m, 3.5m (finder edge), 4.5m (quiet-zone edge).  Using the
# projective-center line as the inner reference makes this the fraction from the
# center toward the physical side.
_TRANSITION_ALPHAS = np.array([3.0, 5.0, 7.0, 9.0]) / 7.0


def thetarho_to_homogeneous_line(theta: float, rho: float) -> np.ndarray:
    """Return (3,) homogeneous line vector [a, b, e] with a·x + b·y + e = 0."""
    return np.array([np.cos(theta), np.sin(theta), -rho], dtype=np.float64)


def homogeneous_line_to_thetarho(ell: np.ndarray) -> tuple[float, float]:
    """Inverse of ``thetarho_to_homogeneous_line``; returns (theta, rho)."""
    a, b, e = np.asarray(ell, dtype=np.float64).ravel()
    norm = np.hypot(a, b)
    if norm < 1e-15:
        return float(np.arctan2(b, a)), float(-e)
    a, b, e = a / norm, b / norm, e / norm
    theta = float(np.arctan2(b, a))
    rho = float(-e)
    return theta, rho


def _homog_point(p: np.ndarray) -> np.ndarray:
    """Convert a Euclidean (x, y) point to homogeneous coordinates."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if len(p) == 3:
        return p
    return np.array([p[0], p[1], 1.0], dtype=np.float64)


def _euclidean(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """De-homogenise a 3-vector, returning NaNs if w == 0."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if abs(p[2]) < eps:
        return np.full(2, np.nan, dtype=np.float64)
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def compute_corners(
    ell_L: np.ndarray,
    ell_R: np.ndarray,
    ell_T: np.ndarray,
    ell_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the four finder corners from its four side lines.

    The ordering is (p_LT, p_RT, p_RB, p_LB) in Euclidean (x, y) coordinates.
    """
    ell_L = np.asarray(ell_L, dtype=np.float64).ravel()
    ell_R = np.asarray(ell_R, dtype=np.float64).ravel()
    ell_T = np.asarray(ell_T, dtype=np.float64).ravel()
    ell_B = np.asarray(ell_B, dtype=np.float64).ravel()

    p_LT = _euclidean(np.cross(ell_L, ell_T))
    p_RT = _euclidean(np.cross(ell_R, ell_T))
    p_RB = _euclidean(np.cross(ell_R, ell_B))
    p_LB = _euclidean(np.cross(ell_L, ell_B))
    return p_LT, p_RT, p_RB, p_LB


def compute_projective_center(
    p_LT: np.ndarray,
    p_RT: np.ndarray,
    p_RB: np.ndarray,
    p_LB: np.ndarray,
) -> np.ndarray:
    """Return the projective center of the finder-pattern quadrilateral.

    The center is the intersection of the two diagonal lines.
    """
    p_LT = _homog_point(p_LT)
    p_RT = _homog_point(p_RT)
    p_RB = _homog_point(p_RB)
    p_LB = _homog_point(p_LB)

    diag1 = np.cross(p_LT, p_RB)
    diag2 = np.cross(p_RT, p_LB)
    return _euclidean(np.cross(diag1, diag2))


def compute_kappa(
    ell_L: np.ndarray,
    ell_R: np.ndarray,
    ell_T: np.ndarray,
    ell_B: np.ndarray,
    c: np.ndarray,
) -> tuple[float, float]:
    """Return the pair of opposite-line scale factors (kappa_u, kappa_v)."""
    ell_L = np.asarray(ell_L, dtype=np.float64).ravel()
    ell_R = np.asarray(ell_R, dtype=np.float64).ravel()
    ell_T = np.asarray(ell_T, dtype=np.float64).ravel()
    ell_B = np.asarray(ell_B, dtype=np.float64).ravel()
    c_h = _homog_point(c)

    dot_L = float(ell_L @ c_h)
    dot_R = float(ell_R @ c_h)
    dot_T = float(ell_T @ c_h)
    dot_B = float(ell_B @ c_h)

    kappa_u = -dot_L / dot_R if abs(dot_R) > 1e-15 else np.nan
    kappa_v = -dot_T / dot_B if abs(dot_B) > 1e-15 else np.nan
    return float(kappa_u), float(kappa_v)


def interpolate_line(
    alpha: float,
    ell_inner: np.ndarray,
    ell_outer: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Interpolate two homogeneous lines: ``ℓ(α) = (1-α)ℓ_inner + α·κ·ℓ_outer``."""
    ell_inner = np.asarray(ell_inner, dtype=np.float64).ravel()
    ell_outer = np.asarray(ell_outer, dtype=np.float64).ravel()
    return (1.0 - alpha) * ell_inner + alpha * kappa * ell_outer


def ray_line_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    line: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Distance ``s >= 0`` where ``origin + s·direction`` meets ``line``.

    The ``line`` may be given either as a homogeneous 3-vector ``[a, b, e]`` or
    a Euclidean 2-vector ``[a, b]`` (``e`` is taken as 0).  Returns NaN if the
    ray is parallel to the line.
    """
    origin = np.asarray(origin, dtype=np.float64).ravel()[:2]
    direction = np.asarray(direction, dtype=np.float64).ravel()[:2]
    line = np.asarray(line, dtype=np.float64).ravel()

    a, b = line[0], line[1]
    e = line[2] if line.size > 2 else 0.0

    denom = a * direction[0] + b * direction[1]
    if abs(denom) < eps:
        return float(np.nan)

    numer = -(a * origin[0] + b * origin[1] + e)
    return float(numer / denom)


def canonical_uv(
    point: np.ndarray,
    ell_L: np.ndarray,
    ell_R: np.ndarray,
    ell_T: np.ndarray,
    ell_B: np.ndarray,
    kappa_u: float,
    kappa_v: float,
) -> tuple[float, float]:
    """Map a Euclidean point to canonical finder coordinates ``(u, v)``.

    The left and top sides map to ``u = 0`` and ``v = 0`` respectively; the right
    and bottom sides map to ``u = 1`` and ``v = 1``.
    """
    p = _homog_point(point)
    ell_L = np.asarray(ell_L, dtype=np.float64).ravel()
    ell_R = np.asarray(ell_R, dtype=np.float64).ravel()
    ell_T = np.asarray(ell_T, dtype=np.float64).ravel()
    ell_B = np.asarray(ell_B, dtype=np.float64).ravel()

    dot_L = float(ell_L @ p)
    dot_R = float(ell_R @ p)
    dot_T = float(ell_T @ p)
    dot_B = float(ell_B @ p)

    denom_u = dot_L - kappa_u * dot_R
    denom_v = dot_T - kappa_v * dot_B
    u = np.nan if abs(denom_u) < 1e-15 else dot_L / denom_u
    v = np.nan if abs(denom_v) < 1e-15 else dot_T / denom_v
    return float(u), float(v)


def _central_line(line: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the line through ``c`` parallel to ``line``.

    The returned line has the same (a, b) coefficients as ``line``.
    """
    line = np.asarray(line, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()[:2]
    a, b = line[0], line[1]
    return np.array([a, b, -(a * c[0] + b * c[1])], dtype=np.float64)


def compute_transition_distances(
    centerpoint: np.ndarray,
    direction: np.ndarray,
    ell_L: np.ndarray,
    ell_R: np.ndarray,
    ell_T: np.ndarray,
    ell_B: np.ndarray,
    kappa_u: float,
    kappa_v: float,
    side_idx: int | None = None,
) -> np.ndarray:
    """Return the four sorted transition distances ``s₁..s₄`` for one half-ray.

    Parameters
    ----------
    side_idx : int or None
        If provided (0=L, 1=R, 2=T, 3=B), compute transitions from that
        side only.  If None, use the 4 smallest positive intersections
        from all sides (legacy behaviour).
    """
    centerpoint = np.asarray(centerpoint, dtype=np.float64).ravel()[:2]
    direction = np.asarray(direction, dtype=np.float64).ravel()[:2]

    side_lines = [
        np.asarray(ell_L, dtype=np.float64).ravel(),
        np.asarray(ell_R, dtype=np.float64).ravel(),
        np.asarray(ell_T, dtype=np.float64).ravel(),
        np.asarray(ell_B, dtype=np.float64).ravel(),
    ]

    # Projective center is computed from the four side lines.
    corners = compute_corners(*side_lines)
    c = compute_projective_center(*corners)
    if not np.all(np.isfinite(c)):
        return np.full(4, np.nan, dtype=np.float64)

    if side_idx is not None:
        # Use only the specified side's 4 interpolated lines.
        line = side_lines[side_idx]
        central = _central_line(line, c)
        dists: list[float] = []
        for alpha in _TRANSITION_ALPHAS:
            ell_alpha = interpolate_line(alpha, central, line, kappa=1.0)
            s = ray_line_intersection(centerpoint, direction, ell_alpha)
            if np.isfinite(s) and s > 1e-9:
                dists.append(s)
        dists = np.sort(np.asarray(dists, dtype=np.float64))
        if len(dists) >= 4:
            return dists[:4]
        out = np.full(4, np.nan, dtype=np.float64)
        out[: len(dists)] = dists
        return out

    positive: list[float] = []
    for line in side_lines:
        central = _central_line(line, c)
        for alpha in _TRANSITION_ALPHAS:
            ell_alpha = interpolate_line(alpha, central, line, kappa=1.0)
            s = ray_line_intersection(centerpoint, direction, ell_alpha)
            if np.isfinite(s) and s > 1e-9:
                positive.append(s)

    positive = np.sort(np.asarray(positive, dtype=np.float64))
    if len(positive) >= 4:
        return positive[:4]

    out = np.full(4, np.nan, dtype=np.float64)
    out[: len(positive)] = positive
    return out


def synthesize_template(
    s_samples: np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Synthesise the smooth finder-pattern intensity template.

    Parameters
    ----------
    s_samples
        Sample distances along the half-ray.
    s_junctions
        The four transition distances ``s₁ < s₂ < s₃ < s₄``.
        Only the first three are used for the template (the 4th is the
        mask boundary, not a physical intensity transition).
    sigma
        Edge softness in pixels.

    Returns
    -------
    template : ndarray
        Values in ``[0, 1]`` with the transition sign pattern
        ``(+1, -1, +1)`` — staying bright beyond ``s₃`` (quiet zone).
    """
    s_samples = np.asarray(s_samples, dtype=np.float64)
    s_junctions = np.asarray(s_junctions, dtype=np.float64).ravel()
    if len(s_junctions) < 3:
        return np.full_like(s_samples, np.nan, dtype=np.float64)

    template = np.zeros_like(s_samples, dtype=np.float64)
    signs = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    for j in range(3):
        template += signs[j] * 0.5 * erfc(-(s_samples - s_junctions[j]) / sigma)
    return template


def precompute_mask(
    s_samples: np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return a smooth float mask ∼1 inside the pattern, decaying to 0.

    The mask is ``≈ 1`` up to the midpoint of the quiet zone (``α = 8/7``)
    and decays smoothly to ``≈ 0`` by the quiet-zone boundary
    (``α = 9/7``, stored as ``s_junctions[3]``).  Uses an erfc transition.
    """
    s_samples = np.asarray(s_samples, dtype=np.float64)
    s_junctions = np.asarray(s_junctions, dtype=np.float64).ravel()
    if len(s_junctions) < 4:
        return np.ones_like(s_samples, dtype=np.float64)

    s_7 = s_junctions[2]
    s_9 = s_junctions[3]
    s_8 = 0.5 * (s_7 + s_9)
    s_mid = 0.5 * (s_8 + s_9)
    width = s_9 - s_8
    sigma_mask = max(width / 4.0, 1e-6)
    return 0.5 * erfc((s_samples - s_mid) / sigma_mask)


# ── Phase 3 — Joint refinement residual and Jacobian ──────────────────────


def _cross_mat(v: np.ndarray) -> np.ndarray:
    """Return the (3, 3) cross-product matrix for a 3-vector."""
    x, y, z = np.asarray(v, dtype=np.float64).ravel()
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def _dehomog_jac(p_h: np.ndarray) -> np.ndarray:
    """Return the (2, 3) Jacobian of Euclidean homogenisation w.r.t. *p_h*."""
    x, y, w = np.asarray(p_h, dtype=np.float64).ravel()
    if abs(w) < 1e-15:
        return np.zeros((2, 3), dtype=np.float64)
    iw = 1.0 / w
    iw2 = iw * iw
    return np.array(
        [[iw, 0.0, -x * iw2], [0.0, iw, -y * iw2]], dtype=np.float64
    )


def _cross_deriv(a: np.ndarray, b: np.ndarray,
                da: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Return ``d(a × b)/dp`` given the point-wise derivatives *da*, *db*."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = np.asarray(da, dtype=np.float64).ravel()
    db = np.asarray(db, dtype=np.float64).ravel()
    return _cross_mat(da) @ b + _cross_mat(a) @ db


def _template_deriv_wrt_junctions(
    s_sample: float | np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    r"""Return ``(4,)`` or ``(N, 4)`` array of :math:`\partial T/\partial s_j`.

    Template definition: ``T(s) = Σ Δ_j·½·erfc(-(s - s_j)/σ)`` with
    ``Δ = [+1, -1, +1, 0]`` — the 4th junction is the mask boundary and
    does not contribute to the template.  The derivative w.r.t. a single
    junction is

    .. math::
        \frac{\partial T}{\partial s_j}
        = -\Delta_j\,\frac{1}{\sigma\sqrt{\pi}}\,
          \exp\!\Bigl(-\bigl(\tfrac{s-s_j}{\sigma}\bigr)^2\Bigr).
    """
    s_sample = np.asarray(s_sample, dtype=np.float64)
    s_junctions = np.asarray(s_junctions, dtype=np.float64).ravel()
    signs = np.array([1.0, -1.0, 1.0, 0.0], dtype=np.float64)
    z = -(s_sample[..., None] - s_junctions[None, :]) / sigma
    derfc_z = -(2.0 / np.sqrt(np.pi)) * np.exp(-z * z)
    result = signs[None, :] * 0.5 * derfc_z / sigma
    if result.ndim == 2 and result.shape[0] == 1:
        result = result[0]
    return result


def _line_deriv_wrt_theta(theta: float) -> np.ndarray:
    """Return ``(3,)`` ``∂ℓ/∂θ`` for ``ℓ = [cos θ, sin θ, -ρ]``."""
    return np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=np.float64)


def _line_deriv_wrt_rho() -> np.ndarray:
    """Return ``(3,)`` ``∂ℓ/∂ρ``."""
    return np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _projective_center_deriv(
    ell_L: np.ndarray, ell_R: np.ndarray,
    ell_T: np.ndarray, ell_B: np.ndarray,
    dL: np.ndarray, dR: np.ndarray,
    dT: np.ndarray, dB: np.ndarray,
) -> np.ndarray:
    """Chain-rule derivative of the 2-D projective center w.r.t. one parameter.

    Parameters
    ----------
    ell_L … ell_B : ndarray (3,)
        Current homogeneous side lines.
    dL … dB : ndarray (3,)
        ``∂ℓ/∂param`` for each of the four side lines.  Zero for lines that
        are independent of the parameter.

    Returns
    -------
    dc_dp : ndarray (2,)
        ``∂c/∂param`` where *c* is the Euclidean projective centre.
    """
    ell_L = np.asarray(ell_L, dtype=np.float64).ravel()
    ell_R = np.asarray(ell_R, dtype=np.float64).ravel()
    ell_T = np.asarray(ell_T, dtype=np.float64).ravel()
    ell_B = np.asarray(ell_B, dtype=np.float64).ravel()
    dL = np.asarray(dL, dtype=np.float64).ravel()
    dR = np.asarray(dR, dtype=np.float64).ravel()
    dT = np.asarray(dT, dtype=np.float64).ravel()
    dB = np.asarray(dB, dtype=np.float64).ravel()

    p_LT = np.cross(ell_L, ell_T)
    p_RT = np.cross(ell_R, ell_T)
    p_RB = np.cross(ell_R, ell_B)
    p_LB = np.cross(ell_L, ell_B)

    dp_LT = _cross_deriv(ell_L, ell_T, dL, dT)
    dp_RT = _cross_deriv(ell_R, ell_T, dR, dT)
    dp_RB = _cross_deriv(ell_R, ell_B, dR, dB)
    dp_LB = _cross_deriv(ell_L, ell_B, dL, dB)

    diag1 = np.cross(p_LT, p_RB)
    diag2 = np.cross(p_RT, p_LB)
    d_diag1 = _cross_deriv(p_LT, p_RB, dp_LT, dp_RB)
    d_diag2 = _cross_deriv(p_RT, p_LB, dp_RT, dp_LB)

    dc_h = _cross_deriv(diag1, diag2, d_diag1, d_diag2)
    c_h = np.cross(diag1, diag2)
    return _dehomog_jac(c_h) @ dc_h

# ── Transition-distance chain-rule helpers ────────────────────────────────


def _all_candidate_info(
    centerpoint: np.ndarray,
    direction: np.ndarray,
    ell_L: np.ndarray, ell_R: np.ndarray,
    ell_T: np.ndarray, ell_B: np.ndarray,
    c: np.ndarray,
    side_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted transition distances, side indices, and alpha indices.

    Parameters
    ----------
    side_idx : int or None
        If provided (0=L, 1=R, 2=T, 3=B), only return candidates from
        that side.  If None, return candidates from all sides.

    Returns
    -------
    s_vals : ndarray (K,)
        Sorted positive intersection distances.  ``K <= 16`` (or ``K <= 4``
        when ``side_idx`` is given).
    side_idx_arr : ndarray (K,) int
        Which side line produced each distance (0=L, 1=R, 2=T, 3=B).
    alpha_idx : ndarray (K,) int
        Which ``_TRANSITION_ALPHAS`` index (0..3) produced each distance.
    """
    centerpoint = np.asarray(centerpoint, dtype=np.float64).ravel()[:2]
    direction = np.asarray(direction, dtype=np.float64).ravel()[:2]

    side_lines = [
        np.asarray(ell_L, dtype=np.float64).ravel(),
        np.asarray(ell_R, dtype=np.float64).ravel(),
        np.asarray(ell_T, dtype=np.float64).ravel(),
        np.asarray(ell_B, dtype=np.float64).ravel(),
    ]
    s_all: list[float] = []
    side_all: list[int] = []
    alpha_all: list[int] = []

    sides_to_iter = [side_idx] if side_idx is not None else range(4)
    for si in sides_to_iter:
        line = side_lines[si]
        central = _central_line(line, c)
        for ai, alpha in enumerate(_TRANSITION_ALPHAS):
            ell_alpha = interpolate_line(alpha, central, line, kappa=1.0)
            s_val = ray_line_intersection(centerpoint, direction, ell_alpha)
            if np.isfinite(s_val) and s_val > 1e-9:
                s_all.append(s_val)
                side_all.append(si)
                alpha_all.append(ai)

    order = np.argsort(s_all)
    return (
        np.asarray(s_all, dtype=np.float64)[order],
        np.asarray(side_all, dtype=np.int64)[order],
        np.asarray(alpha_all, dtype=np.int64)[order],
    )


def _ds_dparams_one_candidate(
    centerpoint: np.ndarray,
    direction: np.ndarray,
    ell_side: np.ndarray,
    d_side: np.ndarray,
    c: np.ndarray,
    dc: np.ndarray,
    alpha: float,
) -> float:
    """Return ``ds/dp`` for one interpolated transition line.

    Parameters
    ----------
    d_side : ndarray (3,)
        ``partial ell_side / partial p``.
    dc : ndarray (2,)
        ``partial c / partial p`` for the projective centre.
    """
    a_s, b_s, _ = np.asarray(ell_side, dtype=np.float64).ravel()
    ox, oy = np.asarray(centerpoint, dtype=np.float64).ravel()[:2]
    dx, dy = np.asarray(direction, dtype=np.float64).ravel()[:2]
    cx, cy = np.asarray(c, dtype=np.float64).ravel()[:2]
    da_s, db_s, de_s = np.asarray(d_side, dtype=np.float64).ravel()
    dcx, dcy = np.asarray(dc, dtype=np.float64).ravel()[:2]

    # Reference (a, b) — same for side line and central line, constant through
    # the interpolation because both use the same (a_s, b_s).
    a_r = a_s
    b_r = b_s
    # e for the interpolated line: (1-alpha)*(-a_r*cx - b_r*cy) + alpha*e_s
    e_r = (1.0 - alpha) * (-a_r * cx - b_r * cy) + alpha * float(
        np.asarray(ell_side, dtype=np.float64).ravel()[2]
    )

    den = float(a_r * dx + b_r * dy)
    if abs(den) < 1e-15:
        return float(np.nan)
    num = float(-(a_r * ox + b_r * oy + e_r))
    inv_den2 = 1.0 / (den * den)

    # Chain: d(e_r)/dp
    de_r = (1.0 - alpha) * (
        -da_s * cx - a_r * dcx - db_s * cy - b_r * dcy
    ) + alpha * de_s

    # d(num)/dp = -(da*ox + db*oy + de_r)
    dnum = -(da_s * ox + db_s * oy + de_r)
    # d(den)/dp = da*dx + db*dy
    dden = da_s * dx + db_s * dy

    # ds/dp = (dnum * den - num * dden) / den^2
    return float((dnum * den - num * dden) * inv_den2)


def _transition_derivs_one_ray(
    centerpoint: np.ndarray,
    direction: np.ndarray,
    ell_L: np.ndarray, ell_R: np.ndarray,
    ell_T: np.ndarray, ell_B: np.ndarray,
    c: np.ndarray,
    dlines_dparams: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    side_idx: int | None = None,
) -> np.ndarray:
    """Return ``(4, 8)`` ``partial s_j / partial p`` for one half-ray.

    Parameters
    ----------
    dlines_dparams : list of 8 items
        Each item is ``(dL, dR, dT, dB)`` — the ``(3,)`` derivative of
        each side line w.r.t. that parameter.  Unaffected lines get zeros.
    side_idx : int or None
        If provided, only use that side's candidates (0=L, 1=R, 2=T, 3=B).
    """
    s_vals, side_idx_arr, alpha_idx = _all_candidate_info(
        centerpoint, direction, ell_L, ell_R, ell_T, ell_B, c,
        side_idx=side_idx,
    )
    if len(s_vals) < 4:
        return np.full((4, 8), np.nan, dtype=np.float64)

    side_lines = [
        np.asarray(ell_L, dtype=np.float64).ravel(),
        np.asarray(ell_R, dtype=np.float64).ravel(),
        np.asarray(ell_T, dtype=np.float64).ravel(),
        np.asarray(ell_B, dtype=np.float64).ravel(),
    ]

    ds_dp = np.zeros((4, 8), dtype=np.float64)
    for p in range(8):
        dL, dR, dT, dB = dlines_dparams[p]
        dlines_list = [dL, dR, dT, dB]
        dc_dp = _projective_center_deriv(
            ell_L, ell_R, ell_T, ell_B, dL, dR, dT, dB,
        )
        for j in range(4):
            si = int(side_idx_arr[j])
            ai = int(alpha_idx[j])
            d_side = np.asarray(dlines_list[si], dtype=np.float64).ravel()
            ds_dp[j, p] = _ds_dparams_one_candidate(
                centerpoint, direction,
                side_lines[si], d_side,
                c, dc_dp,
                float(_TRANSITION_ALPHAS[ai]),
            )
    return ds_dp


# ── Residual ──────────────────────────────────────────────────────────────


def _assign_rays_to_sides(
    centerpoint: np.ndarray,
    half_dirs: np.ndarray,
    ell_L: np.ndarray,
    ell_R: np.ndarray,
    ell_T: np.ndarray,
    ell_B: np.ndarray,
) -> np.ndarray:
    """Assign each half-ray to its nearest side (0=L, 1=R, 2=T, 3=B).

    The nearest side is the one whose line the ray intersects at the
    smallest positive distance.
    """
    centerpoint = np.asarray(centerpoint, dtype=np.float64).ravel()[:2]
    half_dirs = np.asarray(half_dirs, dtype=np.float64)
    side_lines = [
        np.asarray(ell_L, dtype=np.float64).ravel(),
        np.asarray(ell_R, dtype=np.float64).ravel(),
        np.asarray(ell_T, dtype=np.float64).ravel(),
        np.asarray(ell_B, dtype=np.float64).ravel(),
    ]
    n_rays = len(half_dirs)
    assignment = np.full(n_rays, -1, dtype=np.int64)
    for k in range(n_rays):
        best_t = np.inf
        best_si = -1
        for si, line in enumerate(side_lines):
            t = ray_line_intersection(centerpoint, half_dirs[k], line)
            if np.isfinite(t) and t > 1e-9 and t < best_t:
                best_t = t
                best_si = si
        assignment[k] = best_si
    return assignment


def _fit_ols_params(
    centerpoint: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    pre_masks: np.ndarray,
    ell_L: np.ndarray, ell_R: np.ndarray,
    ell_T: np.ndarray, ell_B: np.ndarray,
    kappa_u: float, kappa_v: float,
    sigma: float = 1.0,
    per_ray_side: np.ndarray | None = None,
    ray_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute the OLS brightness/contrast pair ``(a, b)``.

    Fits ``template ≈ a·profile + b`` on all unmasked samples and returns
    the least-squares estimates.

    Parameters
    ----------
    per_ray_side : ndarray (N_rays,) or None
        Side assignment per ray (0=L, 1=R, 2=T, 3=B).  If provided,
        transitions are computed from the assigned side only.
    ray_weights : ndarray (N_rays,) or None
        Per-ray weight (e.g. ``|d·n|``).  If provided, each sample is
        weighted by its ray's weight in the OLS fit.
    """
    T_list: list[np.ndarray] = []
    P_list: list[np.ndarray] = []
    W_list: list[np.ndarray] = []
    for k in range(half_profiles.shape[0]):
        mask_w = pre_masks[k]
        if not np.any(mask_w > 1e-9):
            continue
        si = int(per_ray_side[k]) if per_ray_side is not None else None
        s_j = compute_transition_distances(
            centerpoint, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si,
        )
        if not np.all(np.isfinite(s_j)) or len(s_j) < 4:
            continue
        T_k = synthesize_template(s_samples, s_j, sigma)
        T_list.append(T_k)
        P_list.append(half_profiles[k])
        w = mask_w.copy()
        if ray_weights is not None:
            w = w * float(ray_weights[k])
        W_list.append(w)
    if len(T_list) == 0:
        return 1.0, 0.0
    T_all = np.concatenate(T_list)
    P_all = np.concatenate(P_list)
    W_all = np.concatenate(W_list)
    A = np.column_stack([P_all, np.ones(len(P_all), dtype=np.float64)])
    W_sqrt = np.sqrt(W_all)
    A = A * W_sqrt[:, None]
    T_all = T_all * W_sqrt
    ab, _, _, _ = np.linalg.lstsq(A, T_all, rcond=None)
    return float(ab[0]), float(ab[1])


def joint_refinement_residuals(
    x: np.ndarray,
    centerpoint: np.ndarray,
    R: float,
    theta0: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    pre_masks: np.ndarray,
    sigma: float = 1.0,
    ab_fixed: tuple[float, float] | None = None,
    per_ray_side: np.ndarray | None = None,
    ray_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Residual vector for joint refinement of 4 finder edges.

    Parameters
    ----------
    x : ndarray (8,)
        State ``[phi0/R, phi1/R, phi2/R, phi3/R, rho0, rho1, rho2, rho3]``.
    ab_fixed : tuple of float or None
        If provided, use these fixed ``(a, b)`` for the residual.  When
        ``None``, the pair is re-fitted from the current state via OLS.
        Passing the pair computed from ``x0`` makes the residual function
        smoothly differentiable and keeps the analytical Jacobian exact.
    per_ray_side : ndarray (N_rays,) or None
        Side assignment per ray (0=L, 1=R, 2=T, 3=B).  If provided,
        transitions are computed from the assigned side only.
    ray_weights : ndarray (N_rays,) or None
        Per-ray weights.  If provided, each sample in ray *k* is weighted
        by ``ray_weights[k]``.

    Returns
    -------
    residuals : ndarray (N_rays * N_S,)
        ``(a * profile + b - template) * w``; masked entries are 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    theta0 = np.asarray(theta0, dtype=np.float64).ravel()
    half_profiles = np.asarray(half_profiles, dtype=np.float64)
    half_dirs = np.asarray(half_dirs, dtype=np.float64)
    s_samples = np.asarray(s_samples, dtype=np.float64)
    pre_masks = np.asarray(pre_masks, dtype=np.float64)

    theta = theta0 + x[:4] * R
    rho = x[4:8]

    ell_L = thetarho_to_homogeneous_line(float(theta[0]), float(rho[0]))
    ell_R = thetarho_to_homogeneous_line(float(theta[1]), float(rho[1]))
    ell_T = thetarho_to_homogeneous_line(float(theta[2]), float(rho[2]))
    ell_B = thetarho_to_homogeneous_line(float(theta[3]), float(rho[3]))

    corners = compute_corners(ell_L, ell_R, ell_T, ell_B)
    c = compute_projective_center(*corners)
    kappa_u, kappa_v = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)

    N_rays, N_S = half_profiles.shape
    n_total = N_rays * N_S

    if ab_fixed is not None:
        a_val, b_val = ab_fixed
    else:
        a_val, b_val = _fit_ols_params(
            centerpoint, half_profiles, half_dirs, s_samples, pre_masks,
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v, sigma,
            per_ray_side=per_ray_side, ray_weights=ray_weights,
        )

    residuals = np.zeros(n_total, dtype=np.float64)
    for k in range(N_rays):
        mask_w = pre_masks[k]
        if not np.any(mask_w > 1e-9):
            continue
        si = int(per_ray_side[k]) if per_ray_side is not None else None
        s_j = compute_transition_distances(
            centerpoint, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si,
        )
        if not np.all(np.isfinite(s_j)) or len(s_j) < 4:
            continue
        T_k = synthesize_template(s_samples, s_j, sigma)
        raw = a_val * half_profiles[k] + b_val - T_k
        if ray_weights is not None:
            raw = raw * float(ray_weights[k])
        raw = raw * mask_w
        idx_start = k * N_S
        residuals[idx_start:idx_start + N_S] = raw

    return residuals


# ── Jacobian ──────────────────────────────────────────────────────────────


def joint_refinement_jacobian(
    x: np.ndarray,
    centerpoint: np.ndarray,
    R: float,
    theta0: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    pre_masks: np.ndarray,
    sigma: float = 1.0,
    per_ray_side: np.ndarray | None = None,
    ray_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Analytical Jacobian of ``joint_refinement_residuals``.

    The residual is ``r = w·(a*P + b - T)``.  We treat *a*, *b* and *w* as
    constant, so ``partial r / partial p = -w · partial T / partial p``.

    Parameters
    ----------
    per_ray_side : ndarray (N_rays,) or None
        Side assignment per ray (0=L, 1=R, 2=T, 3=B).  If provided,
        transitions are computed from the assigned side only.
    ray_weights : ndarray (N_rays,) or None
        Per-ray weights.  If provided, the Jacobian rows for ray *k*
        are multiplied by ``ray_weights[k]``.

    Returns
    -------
    Jac : ndarray (R_total, 8)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    theta0 = np.asarray(theta0, dtype=np.float64).ravel()
    half_profiles = np.asarray(half_profiles, dtype=np.float64)
    half_dirs = np.asarray(half_dirs, dtype=np.float64)
    s_samples = np.asarray(s_samples, dtype=np.float64)
    pre_masks = np.asarray(pre_masks, dtype=np.float64)

    theta = theta0 + x[:4] * R
    rho = x[4:8]

    ell_L = thetarho_to_homogeneous_line(float(theta[0]), float(rho[0]))
    ell_R = thetarho_to_homogeneous_line(float(theta[1]), float(rho[1]))
    ell_T = thetarho_to_homogeneous_line(float(theta[2]), float(rho[2]))
    ell_B = thetarho_to_homogeneous_line(float(theta[3]), float(rho[3]))

    corners = compute_corners(ell_L, ell_R, ell_T, ell_B)
    c = compute_projective_center(*corners)
    kappa_u, kappa_v = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)

    N_rays, N_S = half_profiles.shape
    n_total = N_rays * N_S
    Jac = np.zeros((n_total, 8), dtype=np.float64)

    # Line derivatives per parameter.
    # Params 0-3 are scaled angle: d(theta)/dp = R.
    # Params 4-7 are rho: d(rho)/dp = 1.
    zero3 = np.zeros(3, dtype=np.float64)
    dlines_dparams: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for p in range(8):
        dL = dR = dT = dB = zero3.copy()
        if p < 4:
            d_val = _line_deriv_wrt_theta(float(theta[p])) * R
        else:
            d_val = _line_deriv_wrt_rho()
        if p == 0 or p == 4:
            dL = d_val.copy()
        elif p == 1 or p == 5:
            dR = d_val.copy()
        elif p == 2 or p == 6:
            dT = d_val.copy()
        elif p == 3 or p == 7:
            dB = d_val.copy()
        dlines_dparams.append((dL, dR, dT, dB))

    for k in range(N_rays):
        mask_w = pre_masks[k]
        if not np.any(mask_w > 1e-9):
            continue
        si = int(per_ray_side[k]) if per_ray_side is not None else None
        s_j = compute_transition_distances(
            centerpoint, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si,
        )
        if not np.all(np.isfinite(s_j)) or len(s_j) < 4:
            continue

        ds_dp = _transition_derivs_one_ray(
            centerpoint, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B,
            c, dlines_dparams,
            side_idx=si,
        )
        if not np.all(np.isfinite(ds_dp)):
            continue

        dT_ds = _template_deriv_wrt_junctions(s_samples, s_j, sigma)
        dT_dp = dT_ds @ ds_dp
        if ray_weights is not None:
            dT_dp = dT_dp * float(ray_weights[k])

        row_start = k * N_S
        Jac[row_start:row_start + N_S, :] = -dT_dp * mask_w[:, None]

    return Jac


# ── FD verification ───────────────────────────────────────────────────────


def check_joint_refinement_jacobian(
    x0: np.ndarray,
    centerpoint: np.ndarray,
    R: float,
    theta0: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    pre_masks: np.ndarray,
    sigma: float = 1.0,
    eps: float = 5e-6,
    per_ray_side: np.ndarray | None = None,
    ray_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compare analytical Jacobian to central-difference FD.

    The OLS pair ``(a, b)`` is computed from *x0* once and held fixed
    so that FD and analytical derivatives are consistent.

    Parameters
    ----------
    per_ray_side : ndarray (N_rays,) or None
        Side assignment per ray (0=L, 1=R, 2=T, 3=B).  If provided,
        transitions are computed from the assigned side only.
    ray_weights : ndarray (N_rays,) or None
        Per-ray weights passed through to the residual and Jacobian.

    Returns
    -------
    J_analytical : ndarray (R_total, 8)
    J_fd : ndarray (R_total, 8)
    max_rel_error : float
        Maximum relative error on entries where ``|J| > 1e-8``.
    """
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    half_profiles = np.asarray(half_profiles, dtype=np.float64)
    half_dirs = np.asarray(half_dirs, dtype=np.float64)
    s_samples = np.asarray(s_samples, dtype=np.float64)
    pre_masks = np.asarray(pre_masks, dtype=np.float64)

    theta = np.asarray(theta0, dtype=np.float64).ravel()
    rho_val = x0[4:8]

    ell_L = thetarho_to_homogeneous_line(float(theta[0]), float(rho_val[0]))
    ell_R = thetarho_to_homogeneous_line(float(theta[1]), float(rho_val[1]))
    ell_T = thetarho_to_homogeneous_line(float(theta[2]), float(rho_val[2]))
    ell_B = thetarho_to_homogeneous_line(float(theta[3]), float(rho_val[3]))

    corners = compute_corners(ell_L, ell_R, ell_T, ell_B)
    c = compute_projective_center(*corners)
    kappa_u, kappa_v = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)

    ab_fixed = _fit_ols_params(
        centerpoint, half_profiles, half_dirs, s_samples, pre_masks,
        ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v, sigma,
        per_ray_side=per_ray_side, ray_weights=ray_weights,
    )

    J = joint_refinement_jacobian(
        x0, centerpoint, R, theta0, half_profiles, half_dirs,
        s_samples, pre_masks, sigma, per_ray_side=per_ray_side,
        ray_weights=ray_weights,
    )
    J_fd = np.zeros_like(J)
    for col in range(8):
        h = np.zeros(8, dtype=np.float64)
        h[col] = eps
        fp = joint_refinement_residuals(
            x0 + h, centerpoint, R, theta0, half_profiles, half_dirs,
            s_samples, pre_masks, sigma, ab_fixed=ab_fixed,
            per_ray_side=per_ray_side, ray_weights=ray_weights,
        )
        fm = joint_refinement_residuals(
            x0 - h, centerpoint, R, theta0, half_profiles, half_dirs,
            s_samples, pre_masks, sigma, ab_fixed=ab_fixed,
            per_ray_side=per_ray_side, ray_weights=ray_weights,
        )
        J_fd[:, col] = (fp - fm) / (2.0 * eps)

    denom = np.maximum(np.abs(J), 1e-12)
    rel_errors = np.abs(J - J_fd) / denom
    significant = np.abs(J) > 1e-8
    if np.any(significant):
        max_err = float(np.max(rel_errors[significant]))
    else:
        max_err = float(np.max(rel_errors))
    return J, J_fd, max_err


# ── Phase 4 — Joint LM refinement ────────────────────────────────────────


def _reorder_to_standard(
    segments: list[EdgeCluster],
) -> tuple[int, int, int, int]:
    """Return indices of *segments* ordered (L, R, T, B).

    Classification is based on the dominant component of each edge's normal and
    the **geometric position** of the line (derived from *rho* and *normal*).
    This handles conventions where opposite sides share a normal direction.

    When the dominant-component split fails (e.g. rotated finder with diagonally
    oriented edges), falls back to pairing normals by dot product.
    """
    normals = np.array([s.normal for s in segments])
    rhos = np.array([s.rho for s in segments])
    nx = normals[:, 0]
    ny = normals[:, 1]

    is_lr = np.abs(nx) >= np.abs(ny)
    is_tb = ~is_lr

    lr_idx = np.flatnonzero(is_lr)
    tb_idx = np.flatnonzero(is_tb)

    if len(lr_idx) != 2 or len(tb_idx) != 2:
        # Fallback: pair by dot product; look for two opposite-sign pairs
        pairs = _pair_opposite_normals(normals)
        if len(pairs) != 2:
            raise ValueError(
                f"Could not pair normals into two opposite pairs; normals:\n{normals}"
            )
        # Classify each pair as horizontal or vertical by dominant component
        pair_a = (pairs[0][0], pairs[0][1])
        pair_b = (pairs[1][0], pairs[1][1])
        avg_a = normals[pair_a[0]] + normals[pair_a[1]]
        avg_b = normals[pair_b[0]] + normals[pair_b[1]]
        if abs(avg_a[0]) >= abs(avg_b[0]):
            lr_pair, tb_pair = pair_a, pair_b
        else:
            lr_pair, tb_pair = pair_b, pair_a
        lr_idx = np.array(list(lr_pair))
        tb_idx = np.array(list(tb_pair))

    x_pos = rhos[lr_idx] / nx[lr_idx]
    left_idx = lr_idx[np.argmin(x_pos)]
    right_idx = lr_idx[np.argmax(x_pos)]
    if left_idx == right_idx:
        left_idx = lr_idx[np.argmin(nx[lr_idx])]
        right_idx = lr_idx[np.argmax(nx[lr_idx])]

    y_pos = rhos[tb_idx] / ny[tb_idx]
    top_idx = tb_idx[np.argmin(y_pos)]
    bottom_idx = tb_idx[np.argmax(y_pos)]
    if top_idx == bottom_idx:
        top_idx = tb_idx[np.argmin(ny[tb_idx])]
        bottom_idx = tb_idx[np.argmax(ny[tb_idx])]

    return int(left_idx), int(right_idx), int(top_idx), int(bottom_idx)


def _pair_opposite_normals(
    normals: np.ndarray,
) -> list[tuple[int, int]]:
    """Pair 4 normals into two opposite-sign pairs by greedy dot-product matching."""
    used = [False] * 4
    pairs: list[tuple[int, int]] = []
    for i in range(4):
        if used[i]:
            continue
        best_j = -1
        best_dot = 1.0
        for j in range(i + 1, 4):
            if used[j]:
                continue
            d = float(normals[i] @ normals[j])
            if d < best_dot:
                best_dot = d
                best_j = j
        if best_j >= 0:
            pairs.append((i, best_j))
            used[i] = used[best_j] = True
    return pairs


def refine_finder_edges_joint(
    segments: list[EdgeCluster],
    centerpoint: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    sigma: float = 1.0,
) -> tuple[list[EdgeCluster], OptimizeResult]:
    """Jointly refine 4 finder-pattern edge lines with a projective model.

    Parameters
    ----------
    segments : list of EdgeCluster
        Exactly 4 edge segments in any order.  Internally sorted to
        LEFT/RIGHT/TOP/BOTTOM.
    centerpoint : ndarray (2,)
        Ray origin in ``(x, y)`` coordinates.
    half_profiles : ndarray (N_rays, N_S)
        Normalised intensity profiles for each half-ray.
    half_dirs : ndarray (N_rays, 2)
        Unit direction vectors for each ray.
    s_samples : ndarray (N_S,)
        Signed distances from *centerpoint* to each profile sample.
    sigma : float
        Edge softness in pixels (default 1.0).

    Returns
    -------
    refined_segments : list of EdgeCluster
        New ``EdgeCluster`` objects with updated ``.normal``, ``.rho``, and
        ``.direction``.  Other attributes (``label``, ``support``,
        ``pair_indices``, ``sigma_ratio``) are copied from the originals.
    result : OptimizeResult
        Full scipy result from ``least_squares``.
    """
    if len(segments) != 4:
        raise ValueError(f"Expected 4 segments, got {len(segments)}")

    l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(segments)
    ordered = [segments[l_idx], segments[r_idx],
               segments[t_idx], segments[b_idx]]

    theta0 = np.array([np.arctan2(s.normal[1], s.normal[0])
                       for s in ordered], dtype=np.float64)
    rho0 = np.array([s.rho for s in ordered], dtype=np.float64)

    ell_L = thetarho_to_homogeneous_line(float(theta0[0]), float(rho0[0]))
    ell_R = thetarho_to_homogeneous_line(float(theta0[1]), float(rho0[1]))
    ell_T = thetarho_to_homogeneous_line(float(theta0[2]), float(rho0[2]))
    ell_B = thetarho_to_homogeneous_line(float(theta0[3]), float(rho0[3]))

    corners = compute_corners(ell_L, ell_R, ell_T, ell_B)
    c = compute_projective_center(*corners)

    R = float(np.mean([np.linalg.norm(corner - c) for corner in corners]))

    kappa_u, kappa_v = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)

    # Assign each ray to its nearest side (frozen from initial lines).
    per_ray_side = _assign_rays_to_sides(
        centerpoint, half_dirs, ell_L, ell_R, ell_T, ell_B,
    )

    N_rays, N_S = half_profiles.shape

    # Per-ray angular weight: |d·n| — down-weights diagonal rays where
    # the 1-D finder template doesn't apply.
    ray_weights = np.zeros(N_rays, dtype=np.float64)
    for k in range(N_rays):
        si = int(per_ray_side[k])
        if si >= 0:
            w = abs(float(ordered[si].normal @ half_dirs[k]))
            ray_weights[k] = max(w, 0.1)

    pre_masks = np.zeros((N_rays, N_S), dtype=np.float64)
    for k in range(N_rays):
        si = int(per_ray_side[k]) if per_ray_side[k] >= 0 else None
        s_j = compute_transition_distances(
            centerpoint, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si,
        )
        pre_masks[k] = precompute_mask(s_samples, s_j, sigma)

    x0 = np.zeros(8, dtype=np.float64)
    x0[4:8] = rho0

    ab_fixed = _fit_ols_params(
        centerpoint, half_profiles, half_dirs, s_samples, pre_masks,
        ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v, sigma,
        per_ray_side=per_ray_side, ray_weights=ray_weights,
    )

    result = least_squares(
        fun=lambda x, *args: joint_refinement_residuals(
            x, *args, ab_fixed=ab_fixed, per_ray_side=per_ray_side,
            ray_weights=ray_weights),
        x0=x0,
        jac=lambda x, *args: joint_refinement_jacobian(
            x, *args, per_ray_side=per_ray_side, ray_weights=ray_weights),
        method="lm",
        args=(centerpoint, R, theta0, half_profiles, half_dirs,
              s_samples, pre_masks, sigma),
        xtol=1e-6,
        ftol=1e-6,
        max_nfev=200,
    )

    x_opt = result.x
    theta_opt = theta0 + x_opt[:4] * R
    rho_opt = x_opt[4:8]

    refined = []
    for i in range(4):
        n_opt = np.array([np.cos(theta_opt[i]), np.sin(theta_opt[i])])
        d_opt = np.array([-n_opt[1], n_opt[0]])
        refined.append(EdgeCluster(
            label=ordered[i].label,
            pair_indices=ordered[i].pair_indices,
            support=ordered[i].support,
            normal=n_opt,
            rho=float(rho_opt[i]),
            direction=d_opt,
            sigma_ratio=ordered[i].sigma_ratio,
        ))

    return refined, result
