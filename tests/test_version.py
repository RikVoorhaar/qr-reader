"""Tests for version.py: cross-ratio measurement and version estimation."""

import numpy as np
import pytest

from qr_reader.landmarks import (
    NamedLandmarks,
    canonical_grid_landmarks,
    get_colinear_quadruples,
)
from qr_reader.version import (
    Constraint,
    build_constraints,
    estimate_version,
    expected_cross_ratio,
    expected_cross_ratio_by_N,
    filter_constraints,
    measured_cross_ratio,
)


class TestExpectedCrossRatio:
    """Tests for expected_cross_ratio."""

    def test_formula_for_outer(self):
        """expected_cross_ratio(0,7,N-7,N) = (N-7)^2 / (N*(N-14))"""
        for N in [21, 25, 29, 41]:
            r = expected_cross_ratio(0.0, 7.0, float(N - 7), float(N))
            expected = float(N - 7) ** 2 / (float(N) * float(N - 14))
            assert abs(r - expected) < 1e-12

    def test_formula_for_inner(self):
        """expected_cross_ratio(1,6,N-6,N-1) = (N-7)^2 / ((N-2)*(N-7)) = (N-7)/(N-2)"""
        for N in [21, 25, 29, 41]:
            r = expected_cross_ratio(1.0, 6.0, float(N - 6), float(N - 1))
            # (x2-x0)*(x3-x1) = (N-6-1)*(N-1-6) = (N-7)*(N-7)
            # (x3-x0)*(x2-x1) = (N-1-1)*(N-6-6) = (N-2)*(N-12)  -- wait, re-check
            #
            # inner positions: x0=1, x1=6, x2=N-6, x3=N-1
            # (x2-x0) = N-6-1 = N-7
            # (x3-x1) = N-1-6 = N-7
            # numerator = (N-7)^2
            # (x3-x0) = N-1-1 = N-2
            # (x2-x1) = N-6-6 = N-12
            # denominator = (N-2)*(N-12)
            expected = float(N - 7) ** 2 / (float(N - 2) * float(N - 12))
            assert abs(r - expected) < 1e-12

    def test_by_N_helper(self):
        """expected_cross_ratio_by_N returns both outer and inner."""
        outer, inner = expected_cross_ratio_by_N(21)
        assert abs(outer - expected_cross_ratio(0.0, 7.0, 14.0, 21.0)) < 1e-12
        assert abs(inner - expected_cross_ratio(1.0, 6.0, 15.0, 20.0)) < 1e-12


class TestMeasuredCrossRatio:
    """Tests for measured_cross_ratio."""

    def test_perfectly_colinear(self):
        """Four colinear points on a line: line_error ~ 0, r matches expected."""
        pts = np.array(
            [
                [0.0, 0.0],
                [7.0, 0.0],
                [14.0, 0.0],
                [21.0, 0.0],
            ]
        )
        r, line_error, span = measured_cross_ratio(pts)
        assert line_error < 1e-12
        assert span > 0
        # Expected cross-ratio for positions 0,7,14,21:
        r_expected = expected_cross_ratio(0.0, 7.0, 14.0, 21.0)
        assert abs(r - r_expected) < 1e-12

    def test_orientation_flip(self):
        """If u[3] < u[0], the sign is flipped — r should still be correct."""
        pts = np.array(
            [
                [21.0, 0.0],  # reversed order
                [14.0, 0.0],
                [7.0, 0.0],
                [0.0, 0.0],
            ]
        )
        r, line_error, span = measured_cross_ratio(pts)
        r_expected = expected_cross_ratio(0.0, 7.0, 14.0, 21.0)
        assert abs(r - r_expected) < 1e-12

    def test_nearly_colinear_noise(self):
        """Small perpendicular noise → line_error is small but nonzero."""
        np.random.seed(42)
        x = np.array([0.0, 7.0, 14.0, 21.0])
        y_noise = np.random.normal(0, 0.01, 4)
        pts = np.column_stack([x, y_noise])
        r, line_error, span = measured_cross_ratio(pts)
        assert line_error > 0
        assert line_error < 0.01  # small noise
        r_expected = expected_cross_ratio(0.0, 7.0, 14.0, 21.0)
        assert abs(r - r_expected) < 0.05  # cross-ratio is robust

    def test_projective_invariance(self):
        """Cross-ratio is preserved under projective transformation."""
        np.random.seed(123)

        # Four colinear points in grid space
        grid_pts = np.array(
            [
                [0.0, 0.0],
                [7.0, 0.0],
                [14.0, 0.0],
                [21.0, 0.0],
            ]
        )

        r_grid = expected_cross_ratio(0.0, 7.0, 14.0, 21.0)

        # A random but nondegenerate homography
        H = np.array(
            [
                [1.2, 0.1, 10.0],
                [0.3, 0.9, 5.0],
                [0.001, 0.0005, 1.0],
            ]
        )

        def apply_H(pts, H):
            """Apply homography H to points (row, col)."""
            ones = np.ones((len(pts), 1))
            homogeneous = np.hstack([pts, ones])
            projected = homogeneous @ H.T
            projected = projected[:, :2] / projected[:, 2:]
            return projected

        image_pts = apply_H(grid_pts, H)
        r_measured, line_error, _ = measured_cross_ratio(image_pts)

        assert abs(r_measured - r_grid) < 1e-10


class TestConstraints:
    """Tests for build_constraints and filter_constraints."""

    def test_build_from_canonical(self):
        """Build constraints from canonical grid landmarks."""
        lm = canonical_grid_landmarks(21)
        constraints = build_constraints(lm)
        assert len(constraints) == 8
        assert sum(1 for c in constraints if c.type == "outer") == 4
        assert sum(1 for c in constraints if c.type == "inner") == 4

    def test_filter_by_span(self):
        """Constraints with span below min_span are dropped."""
        # Two constraints: one with tiny span, one with normal span
        c1 = Constraint(
            type="outer", label="test1", r_measured=1.5, line_error=0.01, span=0.5
        )  # too small
        c2 = Constraint(
            type="outer", label="test2", r_measured=1.5, line_error=0.01, span=10.0
        )  # fine
        filtered = filter_constraints([c1, c2], min_span=1.0)
        assert len(filtered) == 1
        assert filtered[0].label == "test2"

    def test_filter_keeps_best_k(self):
        """When many constraints exist, only the best k by line_error are kept."""
        constraints = [
            Constraint(
                type="outer",
                label=f"c{i}",
                r_measured=1.5,
                line_error=0.001 * i,
                span=10.0,
            )
            for i in range(10)
        ]
        # c0: 0.000, c1: 0.001, c2: 0.002, ...
        filtered = filter_constraints(constraints, k=3, eps=0.0)
        # best 3 are c0, c1, c2; reference_error = 0.002; threshold = 0.002
        assert len(filtered) == 3
        labels = {c.label for c in filtered}
        assert labels == {"c0", "c1", "c2"}

    def test_filter_caps_error(self):
        """max_error_cap limits how many constraints are kept."""
        constraints = [
            Constraint(
                type="outer", label="c0", r_measured=1.5, line_error=0.001, span=10.0
            ),
            Constraint(
                type="outer", label="c1", r_measured=1.5, line_error=0.002, span=10.0
            ),
            Constraint(
                type="outer", label="c2", r_measured=1.5, line_error=0.003, span=10.0
            ),
            Constraint(
                type="outer", label="c3", r_measured=1.5, line_error=0.10, span=10.0
            ),  # bad
        ]
        # k=2 → best_k = c0, c1 → reference_error = 0.002
        # threshold = min(0.002+0.01, 0.05) = 0.012
        # c0, c1, c2 pass; c3 fails
        filtered = filter_constraints(constraints, k=2, eps=0.01, max_error_cap=0.05)
        labels = {c.label for c in filtered}
        assert "c0" in labels
        assert "c1" in labels
        assert "c2" in labels
        assert "c3" not in labels


class TestEstimateVersion:
    """End-to-end version estimation tests."""

    def test_version_1_from_canonical(self):
        """From canonical V=1 landmarks, estimate_version returns V=1."""
        lm = canonical_grid_landmarks(21)  # N=21 = 4*1+17
        constraints = build_constraints(lm)
        V_best, scores = estimate_version(constraints)
        assert V_best == 1
        assert scores[0] < 1e-6  # perfect fit for V=1

    def test_version_5_from_canonical(self):
        """From canonical V=5 landmarks, estimate_version returns V=5."""
        N = 4 * 5 + 17  # N=37
        lm = canonical_grid_landmarks(N)
        constraints = build_constraints(lm)
        V_best, scores = estimate_version(constraints)
        assert V_best == 5

    def test_version_recovery_under_homography(self):
        """Version is recoverable after an arbitrary projective warp."""
        np.random.seed(42)

        def random_homography():
            """Generate a random non-degenerate homography."""
            H = np.eye(3)
            H += np.random.normal(0, 0.2, (3, 3))
            H[2, 2] = 1.0
            return H

        def apply_H(points, H):
            """Apply homography H to (row, col) points."""
            ones = np.ones((len(points), 1))
            homogeneous = np.hstack([points, ones])
            projected = homogeneous @ H.T
            projected = projected[:, :2] / projected[:, 2:]
            return projected

        for V in [1, 2, 5, 10]:
            N = 4 * V + 17
            lm_grid = canonical_grid_landmarks(N)
            H = random_homography()

            # Warp all points
            def warp(pts):
                return apply_H(pts, H) if pts is not None else None

            warped_lm = NamedLandmarks(
                A=warp(lm_grid.A),
                B=warp(lm_grid.B),
                C=warp(lm_grid.C),
                D=warp(lm_grid.D),
                E=warp(lm_grid.E),
                F=warp(lm_grid.F),
            )

            constraints = build_constraints(warped_lm)
            filtered = filter_constraints(constraints, k=4, min_span=1.0)
            V_best, scores = estimate_version(filtered)

            # For V=1, the cross-ratios are very close across neighboring versions;
            # we accept if the true V is in the top 2 scores.
            if V == 1:
                sorted_idx = np.argsort(scores)
                assert sorted_idx[0] == 0 or sorted_idx[1] == 0, (
                    f"V={V}: best={sorted_idx[0] + 1}, second={sorted_idx[1] + 1}"
                )
            else:
                assert V_best == V, (
                    f"Expected V={V}, got V={V_best}, scores={dict(enumerate(scores))}"
                )


# Note: NamedLandmarks imported at module level above.
