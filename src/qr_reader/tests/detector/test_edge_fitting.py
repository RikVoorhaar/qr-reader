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
    EdgeCluster,
    _fit_ols_params,
    _reorder_to_standard,
    _template_deriv_wrt_junctions,
    assign_points,
    build_pair_distance_matrix,
    canonical_uv,
    cluster_pairs,
    compute_boundary_points,
    compute_corners,
    compute_kappa,
    compute_projective_center,
    compute_transition_distances,
    extract_top_clusters,
    fit_finder_edges,
    homogeneous_line_to_thetarho,
    interpolate_line,
    joint_refinement_jacobian,
    joint_refinement_residuals,
    precompute_mask,
    ray_line_intersection,
    refine_finder_edges_joint,
    synthesize_template,
    thetarho_to_homogeneous_line,
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




# ── helpers for projective tests ──────────────────────────────────────────────


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


def _square_lines(side_radius: float = 3.5):
    """Return (ell_L, ell_R, ell_T, ell_B) for an axis-aligned square.

    The square is centred at the origin with sides ``x = ±side_radius`` and
    ``y = ±side_radius``.  The line convention here uses theta/rho such that
    opposite sides share the same normal direction, giving ``κ_u = κ_v = 1``.
    """
    L = thetarho_to_homogeneous_line(0.0, -side_radius)          # x = -side_radius
    R = thetarho_to_homogeneous_line(0.0, side_radius)           # x = +side_radius
    T = thetarho_to_homogeneous_line(np.pi / 2.0, -side_radius)  # y = -side_radius
    B = thetarho_to_homogeneous_line(np.pi / 2.0, side_radius)   # y = +side_radius
    return L, R, T, B


# ── projective geometry (Phase 1) ────────────────────────────────────────────


class TestHomogeneousLines:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_roundtrip(self, seed):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-np.pi, np.pi)
        rho = rng.uniform(-50.0, 50.0)
        ell = thetarho_to_homogeneous_line(theta, rho)
        theta2, rho2 = homogeneous_line_to_thetarho(ell)
        # angles are normalised to (-π, π]
        assert theta2 == pytest.approx(np.arctan2(np.sin(theta), np.cos(theta)))
        assert rho2 == pytest.approx(rho)


class TestComputeCorners:
    def test_axis_aligned_square(self):
        L, R, T, B = _square_lines(10.0)
        p_LT, p_RT, p_RB, p_LB = compute_corners(L, R, T, B)
        np.testing.assert_allclose(p_LT, [-10.0, -10.0], atol=1e-12)
        np.testing.assert_allclose(p_RT, [10.0, -10.0], atol=1e-12)
        np.testing.assert_allclose(p_RB, [10.0, 10.0], atol=1e-12)
        np.testing.assert_allclose(p_LB, [-10.0, 10.0], atol=1e-12)


class TestComputeProjectiveCenter:
    def test_square_centroid(self):
        lines = _square_lines(10.0)
        corners = compute_corners(*lines)
        c = compute_projective_center(*corners)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(c, np.mean(corners, axis=0), atol=1e-12)


class TestComputeKappa:
    def test_square_same_orientation(self):
        L, R, T, B = _square_lines(10.0)
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        assert kappa_u == pytest.approx(1.0)
        assert kappa_v == pytest.approx(1.0)

    def test_outward_normals_give_minus_one(self):
        # Lines with outward normals: left normal points left, right points right.
        L = thetarho_to_homogeneous_line(np.pi, 10.0)   # x = -10
        R = thetarho_to_homogeneous_line(0.0, 10.0)     # x = +10
        T = thetarho_to_homogeneous_line(-np.pi / 2, 10.0)  # y = -10
        B = thetarho_to_homogeneous_line(np.pi / 2, 10.0)   # y = +10
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        assert kappa_u == pytest.approx(-1.0)
        assert kappa_v == pytest.approx(-1.0)


class TestInterpolateLine:
    def test_endpoints(self):
        L, R, _, _ = _square_lines(7.0)
        np.testing.assert_allclose(interpolate_line(0.0, L, R, 1.0), L, atol=1e-12)
        # α = 1 is the same line as the outer side up to a non-zero scale.
        ell_1 = interpolate_line(1.0, L, R, 1.0)
        # Line scale is irrelevant; compare point sets by cross-product with R.
        assert np.linalg.norm(np.cross(ell_1, R)[:2]) < 1e-12

    def test_central_line_at_half(self):
        L, R, _, _ = _square_lines(7.0)
        ell_half = interpolate_line(0.5, L, R, 1.0)
        # Central line should be x = 0.
        assert abs(ell_half[2]) < 1e-12


class TestRayLineIntersection:
    def test_axis_aligned_hits(self):
        _, R, _, _ = _square_lines(5.0)  # x = 5
        origin = np.array([0.0, 0.0])
        s = ray_line_intersection(origin, np.array([1.0, 0.0]), R)
        assert s == pytest.approx(5.0)

    def test_parallel_returns_nan(self):
        _, R, _, _ = _square_lines(5.0)
        origin = np.array([0.0, 0.0])
        s = ray_line_intersection(origin, np.array([0.0, 1.0]), R)
        assert np.isnan(s)

    def test_behind_is_negative(self):
        _, R, _, _ = _square_lines(5.0)
        s = ray_line_intersection(np.zeros(2), np.array([-1.0, 0.0]), R)
        assert s < 0.0
        assert np.isfinite(s)


class TestCanonicalUv:
    def test_square_corners(self):
        lines = _square_lines(3.5)
        L, R, T, B = lines
        corners = compute_corners(*lines)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(*lines, c)
        expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for corner, (ue, ve) in zip(corners, expected):
            u, v = canonical_uv(corner, L, R, T, B, kappa_u, kappa_v)
            assert u == pytest.approx(ue, abs=1e-12)
            assert v == pytest.approx(ve, abs=1e-12)

    def test_center_is_half(self):
        lines = _square_lines(10.0)
        L, R, T, B = lines
        corners = compute_corners(*lines)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(*lines, c)
        u, v = canonical_uv(c, L, R, T, B, kappa_u, kappa_v)
        assert u == pytest.approx(0.5, abs=1e-12)
        assert v == pytest.approx(0.5, abs=1e-12)


# ── template synthesis (Phase 2) ─────────────────────────────────────────────


class TestComputeTransitionDistances:
    def test_square_rightward(self):
        L, R, T, B = _square_lines(3.5)
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        s = compute_transition_distances(
            np.zeros(2), np.array([1.0, 0.0]), L, R, T, B, kappa_u, kappa_v
        )
        np.testing.assert_allclose(s, [1.5, 2.5, 3.5, 4.5], atol=1e-12)

    def test_square_leftward(self):
        L, R, T, B = _square_lines(3.5)
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        s = compute_transition_distances(
            np.zeros(2), np.array([-1.0, 0.0]), L, R, T, B, kappa_u, kappa_v
        )
        np.testing.assert_allclose(s, [1.5, 2.5, 3.5, 4.5], atol=1e-12)

    def test_offset_centerpoint_differs(self):
        L, R, T, B = _square_lines(3.5)
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        s_center = compute_transition_distances(
            np.zeros(2), np.array([1.0, 0.0]), L, R, T, B, kappa_u, kappa_v
        )
        s_offset = compute_transition_distances(
            np.array([0.5, 0.0]), np.array([1.0, 0.0]),
            L, R, T, B, kappa_u, kappa_v,
        )
        np.testing.assert_allclose(s_center, [1.5, 2.5, 3.5, 4.5], atol=1e-12)
        assert not np.allclose(s_offset, s_center, atol=1e-9)

    def test_returns_four_valid_distances(self):
        L, R, T, B = _square_lines(3.5)
        corners = compute_corners(L, R, T, B)
        c = compute_projective_center(*corners)
        kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
        rng = np.random.default_rng(3)
        for _ in range(20):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            d = np.array([np.cos(theta), np.sin(theta)])
            s = compute_transition_distances(
                np.zeros(2), d, L, R, T, B, kappa_u, kappa_v
            )
            assert s.shape == (4,)
            assert np.all(np.isfinite(s))
            assert np.all(s[:-1] <= s[1:])
            assert np.all(s > 0.0)


class TestSynthesizeTemplate:
    def test_junction_values_are_half(self):
        s_j = np.array([10.0, 20.0, 30.0, 40.0])
        out = synthesize_template(s_j, s_j, sigma=1.0)
        # First 3 junctions each contribute 0.5; 4th is mask boundary (unchanged).
        np.testing.assert_allclose(out[:3], np.full(3, 0.5), atol=1e-3)
        assert out[3] > 0.9  # stays bright in quiet zone

    def test_alternating_regions(self):
        s_j = np.array([10.0, 20.0, 30.0, 40.0])
        sigma = 1.0
        val_dark_ring = synthesize_template(np.array([25.0]), s_j, sigma=sigma)
        val_light_quiet = synthesize_template(np.array([35.0]), s_j, sigma=sigma)
        val_outside = synthesize_template(np.array([50.0]), s_j, sigma=sigma)
        assert val_dark_ring[0] < 0.1
        assert val_light_quiet[0] > 0.9
        assert val_outside[0] > 0.9  # stays bright past quiet zone

    def test_baseline_normalized(self):
        s_j = np.array([10.0, 20.0, 30.0, 40.0])
        s = np.linspace(-10.0, 50.0, 601)
        out = synthesize_template(s, s_j, sigma=1.0)
        assert np.min(out) < 0.05
        assert np.max(out) > 0.95


class TestPrecomputeMask:
    def test_inside_and_beyond(self):
        s_j = np.array([1.5, 2.5, 3.5, 4.5])
        sigma = 1.0
        mask = precompute_mask(np.array([0.0, 4.0, 10.0]), s_j, sigma)
        assert mask[0]
        assert mask[1]
        assert not mask[2]


# ── test data helpers for Phase 3 ──────────────────────────────────────────


def _build_joint_test_data(
    side_radius: float = 3.5,
    n_rays: int = 36,
    n_samples: int = 100,
    sigma: float = 1.0,
    rng_seed: int = 42,
) -> dict:
    """Synthetic test data for the joint refinement residual and Jacobian.

    Returns a dict with all the positional args for ``joint_refinement_residuals``
    and ``joint_refinement_jacobian``, plus the natural *x0*.
    """
    rng = np.random.default_rng(rng_seed)
    L, R, T, B = _square_lines(side_radius)
    theta_L, rho_L = homogeneous_line_to_thetarho(L)
    theta_R, rho_R = homogeneous_line_to_thetarho(R)
    theta_T, rho_T = homogeneous_line_to_thetarho(T)
    theta_B, rho_B = homogeneous_line_to_thetarho(B)

    theta0 = np.array([theta_L, theta_R, theta_T, theta_B], dtype=np.float64)
    rho0 = np.array([rho_L, rho_R, rho_T, rho_B], dtype=np.float64)
    x0 = np.concatenate([np.zeros(4, dtype=np.float64), rho0])

    centerpoint = np.zeros(2, dtype=np.float64)
    corners = compute_corners(L, R, T, B)
    c = compute_projective_center(*corners)
    R_val = float(np.mean([np.linalg.norm(p) for p in corners]))

    half_dirs = np.column_stack([
        np.cos(np.linspace(0, 2 * np.pi, n_rays, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, n_rays, endpoint=False)),
    ])

    max_s = side_radius * 2.0
    s_samples = np.linspace(0.0, max_s, n_samples, dtype=np.float64)

    # Build "perfect" profiles from the template, then add tiny noise so that a~1,b~0.
    half_profiles = np.zeros((n_rays, n_samples), dtype=np.float64)
    pre_masks = np.zeros((n_rays, n_samples), dtype=bool)
    kappa_u, kappa_v = compute_kappa(L, R, T, B, c)
    for k in range(n_rays):
        s_j = compute_transition_distances(
            centerpoint, half_dirs[k], L, R, T, B, kappa_u, kappa_v,
        )
        half_profiles[k] = synthesize_template(s_samples, s_j, sigma)
        pre_masks[k] = precompute_mask(s_samples, s_j, sigma)

    # No noise — perfect match for Jacobian FD check
    return {
        "centerpoint": centerpoint,
        "R": R_val,
        "theta0": theta0,
        "half_profiles": half_profiles,
        "half_dirs": half_dirs,
        "s_samples": s_samples,
        "pre_masks": pre_masks,
        "sigma": sigma,
        "x0": x0,
        "rho0": rho0,
    }


# ── Phase 3 — Joint refinement ────────────────────────────────────────────


class TestTemplateDerivWrtJunctions:
    def test_shape_scalar(self):
        s_j = np.array([1.5, 2.5, 3.5, 4.5])
        out = _template_deriv_wrt_junctions(0.0, s_j, sigma=1.0)
        assert out.shape == (4,)

    def test_shape_vector(self):
        s_j = np.array([1.5, 2.5, 3.5, 4.5])
        s = np.linspace(0, 5, 10)
        out = _template_deriv_wrt_junctions(s, s_j, sigma=1.0)
        assert out.shape == (10, 4)

    def test_fd_check(self):
        s_j = np.array([1.5, 2.5, 3.5, 4.5])
        s = 2.0
        eps = 1e-6
        analytic = _template_deriv_wrt_junctions(s, s_j, sigma=1.0)
        for j in range(4):
            h = np.zeros(4)
            h[j] = eps
            Tp = synthesize_template(np.array([s]), s_j + h, sigma=1.0)[0]
            Tm = synthesize_template(np.array([s]), s_j - h, sigma=1.0)[0]
            fd = (Tp - Tm) / (2.0 * eps)
            assert analytic[j] == pytest.approx(fd, rel=1e-3, abs=1e-6)


class TestJointRefinementResiduals:
    @pytest.fixture(scope="class")
    def data(self):
        return _build_joint_test_data()

    def test_residual_shape(self, data):
        res = joint_refinement_residuals(
            data["x0"], data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        n_rays, n_s = data["half_profiles"].shape
        assert res.shape == (n_rays * n_s,)
        assert res.dtype == np.float64

    def test_residual_near_zero_at_identity(self, data):
        res = joint_refinement_residuals(
            data["x0"], data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        nonzero_mask = np.abs(res) > 0.0
        if np.any(nonzero_mask):
            assert np.max(np.abs(res[nonzero_mask])) < 1e-4

    def test_residual_nonzero_when_shifted(self, data):
        res0 = joint_refinement_residuals(
            data["x0"], data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        x_shifted = data["x0"].copy()
        x_shifted[0] += 0.01  # Perturb theta_L
        res1 = joint_refinement_residuals(
            x_shifted, data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        assert np.max(np.abs(res1)) > np.max(np.abs(res0))


class TestJointRefinementJacobian:
    @pytest.fixture(scope="class")
    def data(self):
        return _build_joint_test_data()

    def test_jacobian_shape(self, data):
        J = joint_refinement_jacobian(
            data["x0"], data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        n_total = data["half_profiles"].shape[0] * data["half_profiles"].shape[1]
        assert J.shape == (n_total, 8)
        assert J.dtype == np.float64

    def test_jacobian_vs_fd(self, data):
        J, J_fd, max_err = check_joint_refinement_jacobian(
            data["x0"], data["centerpoint"], data["R"],
            data["theta0"], data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["pre_masks"], data["sigma"],
        )
        assert max_err <= 1e-3


# ── Phase 4 — Joint LM refinement ────────────────────────────────────────


def _make_edge_clusters(
    side_radius: float = 3.5,
) -> list[EdgeCluster]:
    """Return 4 ``EdgeCluster`` objects for a square centred at the origin.

    Order is the natural top-4 order (L, R, T, B by construction), though
    tests should not rely on this.
    """
    L, R, T, B = _square_lines(side_radius)
    clusters = []
    for label, ell in enumerate([L, R, T, B]):
        theta, rho = homogeneous_line_to_thetarho(ell)
        normal = np.array([np.cos(theta), np.sin(theta)])
        direction = np.array([-normal[1], normal[0]])
        clusters.append(EdgeCluster(
            label=label,
            pair_indices=np.array([label], dtype=int),
            support=np.array([0], dtype=int),
            normal=normal,
            rho=rho,
            direction=direction,
            sigma_ratio=1.0,
        ))
    return clusters


class TestReorderToStandard:
    def test_axis_aligned(self):
        clusters = _make_edge_clusters()
        l, r, t, b = _reorder_to_standard(clusters)
        assert len({l, r, t, b}) == 4
        assert clusters[l].rho / clusters[l].normal[0] < 0   # LEFT
        assert clusters[r].rho / clusters[r].normal[0] > 0   # RIGHT
        assert clusters[t].rho / clusters[t].normal[1] < 0   # TOP
        assert clusters[b].rho / clusters[b].normal[1] > 0   # BOTTOM

    def test_permuted_input(self):
        clusters = _make_edge_clusters()
        rng = np.random.default_rng(123)
        perm = rng.permutation(len(clusters))
        shuffled = [clusters[i] for i in perm]
        l, r, t, b = _reorder_to_standard(shuffled)
        assert len({l, r, t, b}) == 4
        assert shuffled[l].rho / shuffled[l].normal[0] < 0
        assert shuffled[r].rho / shuffled[r].normal[0] > 0
        assert shuffled[t].rho / shuffled[t].normal[1] < 0
        assert shuffled[b].rho / shuffled[b].normal[1] > 0

    def test_all_four_sides_present(self):
        """Tilted normals (3:1 dominant) should split 2 L/R + 2 T/B."""
        normals = [
            np.array([-3.0, -1.0]) / np.sqrt(10),   # TL-ish
            np.array([ 3.0, -1.0]) / np.sqrt(10),   # TR-ish
            np.array([ 1.0,  3.0]) / np.sqrt(10),   # BR-ish
            np.array([-1.0,  3.0]) / np.sqrt(10),   # BL-ish
        ]
        clusters = [EdgeCluster(i, np.array([i]), np.array([0]),
                                n, float(i + 1), np.array([-n[1], n[0]]), 1.0)
                    for i, n in enumerate(normals)]
        l, r, t, b = _reorder_to_standard(clusters)
        assert len({l, r, t, b}) == 4


class TestRefineFinderEdgesJoint:
    @pytest.fixture(scope="class")
    def data(self):
        return _build_joint_test_data()

    @pytest.fixture(scope="class")
    def clusters(self):
        return _make_edge_clusters()

    def test_returns_four_segments(self, data, clusters):
        refined, result = refine_finder_edges_joint(
            clusters, data["centerpoint"],
            data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["sigma"],
        )
        assert len(refined) == 4
        for ec in refined:
            assert isinstance(ec, EdgeCluster)
            assert ec.normal.shape == (2,)
            assert ec.direction.shape == (2,)
            assert isinstance(ec.rho, float)

    def test_converges_on_synthetic(self, data, clusters):
        refined, result = refine_finder_edges_joint(
            clusters, data["centerpoint"],
            data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["sigma"],
        )
        assert result.success or result.cost < 1e-4

    def test_small_shift_on_perfect_data(self, data, clusters):
        refined, result = refine_finder_edges_joint(
            clusters, data["centerpoint"],
            data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["sigma"],
        )
        for i, ec in enumerate(refined):
            assert np.allclose(ec.normal, clusters[i].normal, atol=1e-2)
            assert abs(ec.rho - clusters[i].rho) < 0.5

    def test_improves_on_perturbed(self, data):
        clusters = _make_edge_clusters()
        theta_orig = np.array([np.arctan2(s.normal[1], s.normal[0])
                               for s in clusters], dtype=np.float64)
        rho_orig = np.array([s.rho for s in clusters], dtype=np.float64)

        theta_T, rho_T = homogeneous_line_to_thetarho(
            _square_lines()[2])
        n_T = np.array([np.cos(theta_T + 0.1), np.sin(theta_T + 0.1)])
        d_T = np.array([-n_T[1], n_T[0]])
        clusters[2] = EdgeCluster(2, np.array([2]), np.array([0]),
                                  n_T, rho_T, d_T, 1.0)

        refined, result = refine_finder_edges_joint(
            clusters, data["centerpoint"],
            data["half_profiles"], data["half_dirs"],
            data["s_samples"], data["sigma"],
        )
        assert result.success
        theta_refined = np.array([np.arctan2(ec.normal[1], ec.normal[0])
                                  for ec in refined])
        assert np.allclose(theta_refined[0], theta_orig[0], atol=0.02)
        assert np.allclose(theta_refined[1], theta_orig[1], atol=0.02)
        assert np.allclose(theta_refined[2], theta_orig[2], atol=0.06)
        assert np.allclose(theta_refined[3], theta_orig[3], atol=0.02)

    def test_rejects_wrong_segment_count(self, data):
        clusters = _make_edge_clusters()
        with pytest.raises(ValueError, match="Expected 4"):
            refine_finder_edges_joint(
                clusters[:3], data["centerpoint"],
                data["half_profiles"], data["half_dirs"],
                data["s_samples"], data["sigma"],
            )
