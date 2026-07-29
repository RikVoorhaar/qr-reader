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
) -> np.ndarray:
    """Return the four sorted transition distances ``s₁..s₄`` for one half-ray.
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
    sigma
        Edge softness in pixels.

    Returns
    -------
    template : ndarray
        Values in ``[0, 1]`` with the transition sign pattern
        ``(+1, -1, +1, -1)``.
    """
    s_samples = np.asarray(s_samples, dtype=np.float64)
    s_junctions = np.asarray(s_junctions, dtype=np.float64).ravel()
    if len(s_junctions) < 4:
        return np.full_like(s_samples, np.nan, dtype=np.float64)

    template = np.zeros_like(s_samples, dtype=np.float64)
    signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    for j in range(4):
        template += signs[j] * 0.5 * erfc(-(s_samples - s_junctions[j]) / sigma)
    return template


def precompute_mask(
    s_samples: np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask that is True inside the quiet zone / pattern.

    Samples beyond ``s₄ + 2·σ`` are masked out (returned False).
    """
    s_samples = np.asarray(s_samples, dtype=np.float64)
    s_junctions = np.asarray(s_junctions, dtype=np.float64).ravel()
    if len(s_junctions) < 4:
        return np.ones_like(s_samples, dtype=bool)
    return s_samples <= s_junctions[-1] + 2.0 * sigma
