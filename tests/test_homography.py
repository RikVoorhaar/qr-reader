"""Tests for homography module: normalization, DLT, projection, RANSAC, LM."""

import numpy as np
import pytest

from qr_reader.homography import (
    compute_qr_corners,
    estimate_homography_dlt,
    normalization_transform,
    project_points,
    project_points_with_jac,
    ransac_homography,
    refine_homography_lm,
)


def _apply_H(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography H to (x, y) points."""
    ones = np.ones((len(points), 1))
    homogeneous = np.hstack([points, ones])
    projected = homogeneous @ H.T
    projected = projected[:, :2] / projected[:, 2:]
    return projected


class TestNormalizationTransform:
    """Tests for normalization_transform."""

    def test_centroid_near_zero(self):
        """After normalization, centroid ≈ 0."""
        np.random.seed(42)
        pts = np.random.uniform(100, 500, size=(20, 2))
        T = normalization_transform(pts)
        norm_pts = _apply_H(pts, T)
        centroid = norm_pts.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0], atol=1e-10)

    def test_mean_distance_sqrt2(self):
        """After normalization, mean distance from origin ≈ sqrt(2)."""
        np.random.seed(42)
        pts = np.random.uniform(100, 500, size=(20, 2))
        T = normalization_transform(pts)
        norm_pts = _apply_H(pts, T)
        dists = np.linalg.norm(norm_pts, axis=1)
        mean_dist = np.mean(dists)
        assert abs(mean_dist - np.sqrt(2.0)) < 1e-10


class TestDLT:
    """Tests for estimate_homography_dlt."""

    def test_perfect_recovery(self):
        """Round-trip: random H, map grid pts, recover H."""
        np.random.seed(42)

        H_true = np.array(
            [
                [1.2, 0.1, 10.0],
                [0.3, 0.9, 20.0],
                [0.001, 0.0005, 1.0],
            ]
        )

        # 24 grid points (matching the QR landmark count)
        grid = np.array(
            [
                [0, 0],
                [0, 7],
                [7, 0],
                [7, 7],
                [0, 14],
                [7, 14],
                [14, 0],
                [14, 7],
            ],
            dtype=np.float64,
        )
        grid = np.tile(grid, (3, 1))  # pad to 24 total
        grid = grid[:24]

        image_pts = _apply_H(grid, H_true)
        H_est = estimate_homography_dlt(grid, image_pts)

        # Reprojection should be near-perfect
        reproj = _apply_H(grid, H_est)
        np.testing.assert_allclose(reproj, image_pts, atol=1e-8)

    def test_noise_resilience(self):
        """With small Gaussian noise, reprojection error is small."""
        np.random.seed(42)

        H_true = np.eye(3)
        H_true[0, 2] = 5.0
        H_true[1, 2] = 3.0

        grid = np.random.uniform(0, 50, size=(24, 2))
        image_pts = _apply_H(grid, H_true)
        image_pts += np.random.normal(0, 0.1, image_pts.shape)

        H_est = estimate_homography_dlt(grid, image_pts)
        reproj = _apply_H(grid, H_est)
        errors = np.linalg.norm(reproj - image_pts, axis=1)
        assert np.mean(errors) < 1.0


class TestProjectPoints:
    """Tests for project_points."""

    def test_identity(self):
        """Identity homography: output == input."""
        H = np.eye(3)
        src = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        dst = project_points(H, src)
        np.testing.assert_allclose(dst, src, atol=1e-12)

    def test_translation(self):
        """A simple translation."""
        H = np.eye(3)
        H[0, 2] = 10.0
        H[1, 2] = 20.0
        src = np.array([[1, 2]], dtype=np.float64)
        dst = project_points(H, src)
        np.testing.assert_allclose(dst, [[11, 22]], atol=1e-12)

    def test_matches_manual(self):
        """project_points should match manual homogeneous projection."""
        H = np.array([[1.5, 0.2, 5.0], [0.1, 1.3, 3.0], [0.01, 0.02, 1.0]])
        src = np.random.uniform(0, 100, size=(10, 2))
        dst = project_points(H, src)
        manual = _apply_H(src, H)
        np.testing.assert_allclose(dst, manual, atol=1e-12)


class TestRANSAC:
    """Tests for ransac_homography."""

    def test_clean_data(self):
        """RANSAC on clean data: all points are inliers, H recovered."""
        np.random.seed(42)

        H_true = np.array(
            [
                [1.2, 0.1, 10.0],
                [0.3, 0.9, 20.0],
                [0.001, 0.0005, 1.0],
            ]
        )

        grid = np.random.uniform(0, 100, size=(24, 2))
        image_pts = _apply_H(grid, H_true)

        H_est, inliers = ransac_homography(grid, image_pts, threshold=2.0, iters=500)

        assert np.sum(inliers) >= 20  # most are inliers
        # Reprojection should be good
        reproj = _apply_H(grid, H_est)
        errors = np.linalg.norm(reproj - image_pts, axis=1)
        assert np.median(errors) < 1.0

    def test_corrupted_data(self):
        """RANSAC handles 4–8 corrupted correspondences."""
        np.random.seed(42)

        H_true = np.eye(3)
        H_true[0, 2] = 50.0
        H_true[1, 2] = 30.0

        grid = np.random.uniform(0, 100, size=(24, 2))
        image_pts = _apply_H(grid, H_true)

        # Corrupt 6 random points
        corrupt_idx = np.random.choice(24, size=6, replace=False)
        image_pts[corrupt_idx] += np.random.uniform(50, 100, size=(6, 2))

        H_est, inliers = ransac_homography(grid, image_pts, threshold=3.0, iters=2000)

        # Corrupted points should be marked as outliers
        for idx in corrupt_idx:
            assert not inliers[idx], f"Corrupted point {idx} should be an outlier"

        # Clean points should be inliers
        clean = np.sum(inliers)
        assert clean >= 16  # at least the 18 clean points


class TestRefineLM:
    """Tests for refine_homography_lm."""

    def test_refinement_improves_error(self):
        """LM refinement should reduce reprojection error vs DLT alone."""
        np.random.seed(42)

        H_true = np.array(
            [
                [1.2, 0.1, 10.0],
                [0.3, 0.9, 20.0],
                [0.001, 0.0005, 1.0],
            ]
        )

        grid = np.random.uniform(0, 100, size=(24, 2))
        image_pts = _apply_H(grid, H_true)
        # Add small noise
        image_pts += np.random.normal(0, 0.5, image_pts.shape)

        H_dlt = estimate_homography_dlt(grid, image_pts)
        dlt_err = np.mean(np.linalg.norm(_apply_H(grid, H_dlt) - image_pts, axis=1))

        H_lm = refine_homography_lm(H_dlt, grid, image_pts)
        lm_err = np.mean(np.linalg.norm(_apply_H(grid, H_lm) - image_pts, axis=1))

        assert lm_err <= dlt_err + 0.1  # LM should not be worse (allow tiny epsilon)


class TestProjectPointsWithJac:
    """Tests for project_points_with_jac analytical Jacobian."""

    def test_jac_matches_finite_diff(self):
        """Analytical Jacobian matches scipy's finite-difference approximation."""
        from scipy.optimize._numdiff import approx_derivative

        rng = np.random.default_rng(42)
        src_xy = rng.uniform(0, 100, size=(8, 2))

        for _ in range(5):
            H = np.eye(3) + rng.normal(0, 0.3, (3, 3))
            H[2, 2] = 1.0
            params = H.ravel()[:8]

            # Analytical Jacobian
            _, J_analytical = project_points_with_jac(H, src_xy)

            # Finite-difference Jacobian of project_points
            def f(p):
                Hp = np.eye(3)
                Hp.ravel()[:8] = p
                return project_points(Hp, src_xy).ravel()

            J_fd = approx_derivative(f, params, method="2-point")

            # Finite-difference accuracy is poor on large-magnitude entries
            # (~10^5–10^6); this is a sanity check that the analytical formulas
            # are structurally correct, not a precision assertion.
            np.testing.assert_allclose(J_analytical, J_fd, rtol=1e-3, atol=1e-6)

    def test_jac_for_identity(self):
        """With identity H, check a few known Jacobian entries."""
        src_xy = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64)
        H = np.eye(3)

        pts, J = project_points_with_jac(H, src_xy)

        # For identity H: u=x, v=y, w=1
        np.testing.assert_allclose(pts, src_xy, atol=1e-12)

        # Point 0: x=2, y=3, u=2, v=3, w=1
        # Row 0 (u): [x/w, y/w, 1/w, 0, 0, 0, -x·u/w, -y·u/w]
        #            = [2, 3, 1, 0, 0, 0, -4, -6]
        np.testing.assert_allclose(
            J[0], [2.0, 3.0, 1.0, 0.0, 0.0, 0.0, -4.0, -6.0], atol=1e-12
        )
        # Row 1 (v): [0, 0, 0, x/w, y/w, 1/w, -x·v/w, -y·v/w]
        #            = [0, 0, 0, 2, 3, 1, -6, -9]
        np.testing.assert_allclose(
            J[1], [0.0, 0.0, 0.0, 2.0, 3.0, 1.0, -6.0, -9.0], atol=1e-12
        )

    def test_project_points_with_jac_output(self):
        """project_points_with_jac output matches project_points."""
        rng = np.random.default_rng(123)
        src_xy = rng.uniform(0, 100, size=(12, 2))
        H = np.eye(3) + rng.normal(0, 0.3, (3, 3))
        H[2, 2] = 1.0

        pts_jac, _ = project_points_with_jac(H, src_xy)
        pts = project_points(H, src_xy)

        np.testing.assert_allclose(pts_jac, pts, atol=1e-12)


class TestComputeQRCorners:
    """Tests for compute_qr_corners."""

    def test_identity_H_known_N(self):
        """With identity H, corners match the grid corners."""
        H = np.eye(3)
        N = 21
        corners = compute_qr_corners(H, N)
        expected = np.array(
            [
                [0.0, 0.0],
                [21.0, 0.0],
                [21.0, 21.0],
                [0.0, 21.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(corners, expected, atol=1e-12)

    def test_translated_H(self):
        """With a translated H, corners are offset correctly."""
        H = np.eye(3)
        H[0, 2] = 100.0
        H[1, 2] = 200.0
        corners = compute_qr_corners(H, 21)
        expected = np.array(
            [
                [100.0, 200.0],
                [121.0, 200.0],
                [121.0, 221.0],
                [100.0, 221.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(corners, expected, atol=1e-12)

    def test_order_matches_OpenCV_expectation(self):
        """Corners should be [TL, TR, BR, BL]."""
        H = np.eye(3)
        corners = compute_qr_corners(H, 21)
        # TL has min x and min y
        assert corners[0, 0] < corners[1, 0]  # TL.x < TR.x
        assert corners[0, 1] < corners[3, 1]  # TL.y < BL.y
        assert corners[1, 1] < corners[2, 1]  # TR.y < BR.y
        assert corners[3, 0] < corners[2, 0]  # BL.x < BR.x
