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
    m_pos: np.ndarray,
    m_neg: np.ndarray,
    theta_rad: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
) -> np.ndarray:
    """1N-circuit boundary points: one point per ray direction.

    For direction ``theta_rad[i]``, prefer the positive half-ray fit
    ``m_pos[i]``; fall back to the negative half of the opposite ray
    ``m_neg[(i + N/2) % N]`` (which points in the same direction).  Rows
    with no valid fit are NaN.

    Returns
    -------
    points : ndarray (N, 2)
        Boundary points in (x, y), NaN where no fit succeeded.
    """
    n = len(theta_rad)
    half_n = n // 2
    points = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        dir_vec = np.array([np.cos(theta_rad[i]), np.sin(theta_rad[i])])
        if np.isfinite(m_pos[i]):
            points[i] = center_xy + pitch_constant * m_pos[i] * dir_vec
        else:
            neg_idx = (i + half_n) % n
            if np.isfinite(m_neg[neg_idx]):
                points[i] = center_xy + pitch_constant * m_neg[neg_idx] * dir_vec
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
