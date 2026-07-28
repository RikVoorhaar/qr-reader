"""Tests for finder-pattern edge fitting (v3 plan).

Covers: 1N boundary-point construction, TLS line fits, the pairwise
sigma2/sigma1 distance matrix (including the cyclic-gap guard), sklearn
single-linkage clustering, top-4 extraction, and tie-broken point
assignment.  Synthetic data is a perfect (or lightly perturbed) square.
"""

import numpy as np
import pytest

from qr_reader.detector.edge_fitting import (
    DISTANCE_THRESHOLD,
    MAX_GAP,
    assign_points,
    build_pair_distance_matrix,
    cluster_pairs,
    compute_boundary_points,
    extract_top_clusters,
    fit_finder_edges,
    tls_line,
)


def square_boundary_points(
    n_rays: int = 36, half: float = 10.0, center=(0.0, 0.0)
) -> np.ndarray:
    """Boundary points of an axis-aligned square, sampled by equal-angle rays."""
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = half / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    return np.stack(
        [center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)], axis=1
    )


# ── compute_boundary_points ──────────────────────────────────────────────────


class TestComputeBoundaryPoints:
    def test_output_shape(self):
        k = 4
        theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
        center = np.array([5.0, 7.0])
        m = np.full(k, 2.0)
        pts = compute_boundary_points(center, m, theta, pitch_constant=3.5)
        assert pts.shape == (k, 2)
        for i in range(k):
            d = np.array([np.cos(theta[i]), np.sin(theta[i])])
            np.testing.assert_allclose(pts[i], center + 3.5 * 2.0 * d, atol=1e-12)

    def test_nan_rows(self):
        k = 4
        theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
        m = np.array([1.0, np.nan, 3.0, np.nan])
        pts = compute_boundary_points(np.zeros(2), m, theta)
        assert pts.shape == (4, 2)
        assert np.isfinite(pts[0]).all(); assert np.isnan(pts[1]).all()
        assert np.isfinite(pts[2]).all(); assert np.isnan(pts[3]).all()

    def test_all_nan_when_no_fits(self):
        k = 6
        theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pts = compute_boundary_points(
            np.zeros(2), np.full(k, np.nan), theta,
        )
        assert pts.shape == (6, 2)
        assert np.isnan(pts).all()


# ── tls_line ─────────────────────────────────────────────────────────────────


class TestTlsLine:
    def test_horizontal_line(self):
        pts = np.array([[0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [5.0, 3.0]])
        n, rho, d, s = tls_line(pts)
        np.testing.assert_allclose(np.abs(n), [0.0, 1.0], atol=1e-12)
        assert rho == pytest.approx(3.0)
        assert rho >= 0
        np.testing.assert_allclose(np.abs(d), [1.0, 0.0], atol=1e-12)
        assert s[1] == pytest.approx(0.0, abs=1e-12)

    def test_diagonal_line(self):
        t = np.linspace(-2, 2, 7)
        pts = np.stack([t, t + 1.0], axis=1)  # y = x + 1
        n, rho, d, s = tls_line(pts)
        # Normal proportional to (-1, 1)/sqrt(2), rho = 1/sqrt(2), forced positive.
        np.testing.assert_allclose(np.abs(n @ np.array([1.0, 1.0])), 0.0, atol=1e-12)
        assert rho == pytest.approx(1.0 / np.sqrt(2.0))
        assert s[1] == pytest.approx(0.0, abs=1e-12)

    def test_rho_nonnegative(self):
        pts = np.array([[0.0, -3.0], [1.0, -3.0], [2.0, -3.0]])
        _, rho, _, _ = tls_line(pts)
        assert rho == pytest.approx(3.0)


# ── build_pair_distance_matrix ───────────────────────────────────────────────


class TestPairDistanceMatrix:
    def test_cyclic_gap_guard_on_colinear_points(self):
        # 8 colinear points; every union is perfectly colinear, so any
        # non-1.0 entry must be ~0.  The gap guard alone decides which
        # entries are 1.0.
        pts = np.stack([np.arange(8, dtype=float), np.zeros(8)], axis=1)
        orig = np.arange(8)
        D, pairs = build_pair_distance_matrix(pts, orig, n_total=8, max_gap=1)
        assert D.shape == (8, 8)
        np.testing.assert_allclose(np.diag(D), 0.0)
        assert D[0, 1] == pytest.approx(0.0, abs=1e-9)  # share a point
        assert D[0, 2] == pytest.approx(0.0, abs=1e-9)  # gap 0
        assert D[0, 3] == pytest.approx(0.0, abs=1e-9)  # gap 1 == MAX_GAP
        assert D[0, 4] == pytest.approx(1.0)  # gap 2 > MAX_GAP
        # Symmetry
        np.testing.assert_allclose(D, D.T)

    def test_gap_uses_original_indices(self):
        # 4 valid points with original ray indices [0, 1, 3, 4] out of 10:
        # point "2" is missing.  Pairs: (0,1), (1,3), (3,4), (4,...0 wrap).
        pts = np.stack([np.arange(4, dtype=float), np.zeros(4)], axis=1)
        orig = np.array([0, 1, 3, 4])
        D, pairs = build_pair_distance_matrix(pts, orig, n_total=10, max_gap=1)
        # Pair 0 = points {0,1} (orig 0,1); pair 2 = points {2,3} (orig 3,4).
        # Closest members orig 1 and 3 → one point (2) between → gap 1 → allowed.
        assert D[0, 2] == pytest.approx(0.0, abs=1e-9)
        # Wrap-around pair 3 = orig {4, 0}: shares orig 0 with pair 0 → allowed.
        assert D[0, 3] == pytest.approx(0.0, abs=1e-9)

    def test_cross_edge_pairs_have_high_ratio(self):
        pts = square_boundary_points(n_rays=36)
        orig = np.arange(36)
        D, pairs = build_pair_distance_matrix(pts, orig, n_total=36, max_gap=1)
        # Pairs 1 and 2 both lie fully on the right edge (x = +10):
        # rays 10°..40° hit the right edge for 36 rays? Right edge covers
        # theta in (-45°, 45°) → rays 0..4 and 32..35.
        assert D[0, 2] == pytest.approx(0.0, abs=1e-9)
        # Pair 4 = points (4,5): point 4 (40°) is on the right edge, point 5
        # (50°) is on the top edge → union with pair 2 is non-colinear.
        assert D[2, 4] > 0.05


# ── clustering + extraction on a perfect square ──────────────────────────────


class TestSquarePipeline:
    def test_four_clusters_on_perfect_square(self):
        pts = square_boundary_points(n_rays=36)
        result = fit_finder_edges(pts)
        assert len(result.clusters) == 4
        # Each fitted line should be one of x=±10 / y=±10: axis-aligned
        # normal, rho ≈ 10.
        for ec in result.clusters:
            assert ec.rho == pytest.approx(10.0, abs=1e-6)
            assert np.max(np.abs(ec.normal)) == pytest.approx(1.0, abs=1e-9)
        # All four sides present: normals must be distinct.
        normals = np.array([ec.normal for ec in result.clusters])
        signed = normals.round(6)
        assert len({tuple(row) for row in signed}) == 4

    def test_support_points_lie_on_their_line(self):
        pts = square_boundary_points(n_rays=36)
        result = fit_finder_edges(pts)
        for ec in result.clusters:
            support_pts = result.points[ec.support]
            dist = np.abs(support_pts @ ec.normal - ec.rho)
            assert np.max(dist) < 1e-6

    def test_survives_missing_points(self):
        pts = square_boundary_points(n_rays=36)
        # Knock out one ray per edge (non-adjacent): gaps of 1 are bridgeable.
        pts[2] = np.nan
        pts[11] = np.nan
        pts[20] = np.nan
        pts[29] = np.nan
        result = fit_finder_edges(pts)
        assert len(result.clusters) == 4
        for ec in result.clusters:
            assert ec.rho == pytest.approx(10.0, abs=1e-6)

    def test_noisy_square(self):
        rng = np.random.default_rng(0)
        pts = square_boundary_points(n_rays=36, half=10.0)
        pts += rng.normal(scale=0.05, size=pts.shape)
        result = fit_finder_edges(pts)
        assert len(result.clusters) == 4
        for ec in result.clusters:
            assert ec.rho == pytest.approx(10.0, abs=0.2)

    def test_assignment_is_unique_and_covers_supports(self):
        pts = square_boundary_points(n_rays=36)
        result = fit_finder_edges(pts)
        assignment = result.assignment
        assert assignment.shape == (len(result.points),)
        all_support = set()
        for ec in result.clusters:
            all_support.update(ec.support.tolist())
        # Every supported point is assigned to exactly one of the 4 clusters.
        for p in range(len(result.points)):
            if p in all_support:
                assert 0 <= assignment[p] < 4
            else:
                assert assignment[p] == -1

    def test_too_few_points_returns_no_clusters(self):
        pts = np.full((36, 2), np.nan)
        pts[0] = [1.0, 0.0]
        pts[9] = [0.0, 1.0]
        pts[18] = [-1.0, 0.0]
        result = fit_finder_edges(pts)
        assert result.clusters == []


# ── unit pieces ──────────────────────────────────────────────────────────────


class TestClusterPairs:
    def test_single_linkage_bridges_far_cyclic_pairs(self):
        # Distance matrix where consecutive pairs are close but distant
        # pairs are 1.0: single linkage must still form one cluster.
        n = 6
        D = np.ones((n, n))
        np.fill_diagonal(D, 0.0)
        for i in range(n - 1):
            D[i, i + 1] = D[i + 1, i] = 0.01
        labels = cluster_pairs(D, distance_threshold=DISTANCE_THRESHOLD)
        assert len(set(labels.tolist())) == 1


class TestExtractTopClusters:
    def test_top_k_by_size(self):
        pts = square_boundary_points(n_rays=36)
        orig = np.arange(36)
        D, pairs = build_pair_distance_matrix(pts, orig, n_total=36, max_gap=MAX_GAP)
        labels = cluster_pairs(D)
        clusters = extract_top_clusters(labels, pairs, pts, k=4)
        assert len(clusters) == 4
        sizes = [len(ec.pair_indices) for ec in clusters]
        assert sizes == sorted(sizes, reverse=True)
        # The 4 edge clusters are all larger than any leftover corner cluster.
        from collections import Counter

        counts = Counter(labels.tolist())
        leftover = sorted(counts.values(), reverse=True)[4:]
        if leftover:
            assert min(sizes) >= max(leftover)


class TestAssignPoints:
    def test_tie_break_prefers_lower_sigma_ratio(self):
        from qr_reader.detector.edge_fitting import EdgeCluster

        eps = np.zeros(2)
        c0 = EdgeCluster(
            label=0,
            pair_indices=np.array([0]),
            support=np.array([0, 1]),
            normal=np.array([0.0, 1.0]),
            rho=0.0,
            direction=np.array([1.0, 0.0]),
            sigma_ratio=0.05,
        )
        c1 = EdgeCluster(
            label=1,
            pair_indices=np.array([1]),
            support=np.array([1, 2]),
            normal=np.array([1.0, 0.0]),
            rho=0.0,
            direction=np.array([0.0, 1.0]),
            sigma_ratio=0.02,
        )
        assignment = assign_points([c0, c1], n_points=3)
        assert assignment[0] == 0
        assert assignment[1] == 1  # shared → lower sigma_ratio wins
        assert assignment[2] == 1
