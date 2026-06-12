"""Tests for landmarks module: corner ordering and named landmarks."""

import numpy as np

from qr_reader.detector.finder_pattern import FinderPattern, Triplet
from qr_reader.detector.landmarks import (
    NamedLandmarks,
    canonical_grid_landmarks,
    get_colinear_quadruples,
    local_basis,
    order_square_corners,
)


class TestOrderSquareCorners:
    """Tests for order_square_corners with the local basis."""

    def test_axis_aligned_square_clockwise(self):
        """Axis-aligned square with known TL/BL/BR/TR corners."""
        # In (row, col): TL=(0,0), BL=(7,0), BR=(7,7), TR=(0,7)
        points4 = np.array(
            [
                [7, 7],  # BR
                [0, 7],  # TR
                [7, 0],  # BL
                [0, 0],  # TL
            ],
            dtype=np.float64,
        )
        right = np.array([0.0, 1.0])  # (row, col): col increases = right
        down = np.array([1.0, 0.0])  # (row, col): row increases = down

        ordered = order_square_corners(points4, right, down)
        expected = np.array(
            [
                [0, 0],  # TL
                [7, 0],  # BL
                [7, 7],  # BR
                [0, 7],  # TR
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(ordered, expected, atol=1e-10)

    def test_rotated_square(self):
        """Rotated square: ordering is basis-relative and still correct."""
        # A square centered at (5,5), side length 10, rotated 30°.
        theta = np.radians(30)
        c = 5.0
        s = 10.0

        # Canonical corners (row, col) before rotation:
        # TL=(0,0), BL=(s,0), BR=(s,s), TR=(0,s)
        canonical = np.array(
            [
                [c - s / 2, c - s / 2],  # TL
                [c + s / 2, c - s / 2],  # BL
                [c + s / 2, c + s / 2],  # BR
                [c - s / 2, c + s / 2],  # TR
            ],
            dtype=np.float64,
        )

        # Rotate by theta around center
        center = np.array([c, c])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = (canonical - center) @ R.T + center

        # Basis vectors (also rotated): right = (0,1) rotated, down = (1,0) rotated
        right = np.array([-sin_t, cos_t])
        down = np.array([cos_t, sin_t])

        # Shuffle to simulate unordered input
        idx = np.array([2, 0, 3, 1])  # BR, TL, TR, BL
        shuffled = rotated[idx]

        ordered = order_square_corners(shuffled, right, down)

        # Should recover TL, BL, BR, TR
        np.testing.assert_allclose(ordered[0], rotated[0], atol=1e-10)  # TL
        np.testing.assert_allclose(ordered[1], rotated[1], atol=1e-10)  # BL
        np.testing.assert_allclose(ordered[2], rotated[2], atol=1e-10)  # BR
        np.testing.assert_allclose(ordered[3], rotated[3], atol=1e-10)  # TR


class TestLocalBasis:
    """Tests for local_basis."""

    def test_axis_aligned_triplet(self):
        """Simple axis-aligned triplet → right is (0,1), down is (1,0)."""
        fps = [
            FinderPattern(
                cluster_idx=0,
                outer_corners=np.array(
                    [[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64
                ),
            ),  # TL
            FinderPattern(
                cluster_idx=1,
                outer_corners=np.array(
                    [[0, 20], [7, 20], [7, 27], [0, 27]], dtype=np.float64
                ),
            ),  # TR
            FinderPattern(
                cluster_idx=2,
                outer_corners=np.array(
                    [[20, 0], [27, 0], [27, 7], [20, 7]], dtype=np.float64
                ),
            ),  # BL
        ]
        t = Triplet(top_left_idx=0, top_right_idx=1, bottom_left_idx=2)

        right, down = local_basis(t, fps)

        assert np.dot(right, np.array([0.0, 1.0])) > 0.9  # roughly +col
        assert np.dot(down, np.array([1.0, 0.0])) > 0.9  # roughly +row


class TestCanonicalGridLandmarks:
    """Tests for canonical_grid_landmarks."""

    def test_N_21_coords(self):
        """For N=21 (V=1), spot-check canonical coordinates."""
        lm = canonical_grid_landmarks(21)

        # C is top-right outer: col=N-7..N, row=0..7
        # C[2] = BR = (7, N) = (7, 21)
        np.testing.assert_allclose(lm.C[2], [7, 21], atol=1e-12)

        # B is top-left inner: (1,1)(6,1)(6,6)(1,6)
        expected_B = np.array(
            [
                [1, 1],
                [6, 1],
                [6, 6],
                [1, 6],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(lm.B, expected_B, atol=1e-12)

    def test_all_B_are_offset_1(self):
        """Inner squares B/D/F all use positions 1..6, not the center."""
        for N in [21, 25, 29]:
            lm = canonical_grid_landmarks(N)
            np.testing.assert_allclose(lm.B[0], [1, 1], atol=1e-12)  # TL
            np.testing.assert_allclose(lm.B[2], [6, 6], atol=1e-12)  # BR


class TestColinearQuadruples:
    """Tests for get_colinear_quadruples."""

    def test_canonical_outer_quadruples_colinear(self):
        """Each canonical outer quadruple should be colinear in grid space."""
        lm = canonical_grid_landmarks(21)
        quads = get_colinear_quadruples(lm)
        outer_quads = [q for q in quads if q.type == "outer"]

        for q in outer_quads:
            pts = q.points
            # Check colinearity: the 4 points should have negligible
            # perpendicular spread. Use the same SVD approach.
            center = pts.mean(axis=0)
            centered = pts - center
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            sigma2 = S[1] if len(S) > 1 else 0.0
            sigma1 = S[0]
            line_error = sigma2 / (sigma1 + 1e-12)
            assert line_error < 1e-12, f"{q.label}: line_error={line_error}"

    def test_canonical_inner_quadruples_colinear(self):
        """Each canonical inner quadruple should be colinear in grid space."""
        lm = canonical_grid_landmarks(21)
        quads = get_colinear_quadruples(lm)
        inner_quads = [q for q in quads if q.type == "inner"]

        for q in inner_quads:
            pts = q.points
            center = pts.mean(axis=0)
            centered = pts - center
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            sigma2 = S[1] if len(S) > 1 else 0.0
            sigma1 = S[0]
            line_error = sigma2 / (sigma1 + 1e-12)
            assert line_error < 1e-12, f"{q.label}: line_error={line_error}"

    def test_quadruple_count(self):
        """We get exactly 8 quadruples when all inner corners are available."""
        lm = canonical_grid_landmarks(21)
        quads = get_colinear_quadruples(lm)
        assert len(quads) == 8
        assert sum(1 for q in quads if q.type == "outer") == 4
        assert sum(1 for q in quads if q.type == "inner") == 4

    def test_quadruple_count_no_inner(self):
        """When inner corners are None, we get only 4 outer quadruples."""
        A = np.eye(4)[:4, :2]  # dummy
        lm = NamedLandmarks(
            A=A,
            B=None,
            C=A,
            D=None,
            E=A,
            F=None,
        )
        quads = get_colinear_quadruples(lm)
        assert len(quads) == 4
        assert all(q.type == "outer" for q in quads)
