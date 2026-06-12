"""Tests for the high-level detector API (detector.py)."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.detector.detector import detect_corners, detect_homography, detect_sample
from qr_reader.qr_gen import generate_test_image, make_qr_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_distorted_image(
    version: int = 1,
    content: str = "test",
    seed: int = 42,
) -> np.ndarray:
    """Generate a noisy, rotated, perspective-warped QR image."""
    return generate_test_image(
        version=version,
        content=content,
        rotation_angle_deg=20.0,
        perspective_max_shift=30.0,
        noise_std=40.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# detect_corners
# ---------------------------------------------------------------------------


class TestDetectCorners:
    """Tests for detect_corners."""

    def test_clean_image_returns_four_corners(self):
        """A clean QR image should yield exactly 4 corners."""
        img = make_qr_image(content="hi", version=1, box_size=10, border=4)
        corners, version = detect_corners(img)

        assert corners.shape == (4, 2)
        assert corners.dtype == np.float64
        assert version == 1

    def test_distorted_image_returns_four_corners(self):
        """A distorted image should still yield 4 corners."""
        img = _generate_distorted_image(version=1, content="distort", seed=1)
        corners, version = detect_corners(img)

        assert corners.shape == (4, 2)
        assert version == 1

    def test_corner_order(self):
        """Corners should be [TL, TR, BR, BL] in (x, y)."""
        img = make_qr_image(content="order", version=1, box_size=10, border=4)
        corners, _version = detect_corners(img)

        tl, tr, br, bl = corners
        # For an axis-aligned image the expected corners are roughly:
        # TL=(40,40), TR=(290,40), BR=(290,290), BL=(40,290)
        # Allow a few pixels of detection jitter.
        assert tl[0] < tr[0] + 5.0  # TL.x roughly < TR.x
        assert tl[1] < bl[1] - 200.0  # TL.y well above BL.y
        assert tr[1] < br[1] - 200.0  # TR.y well above BR.y
        assert bl[0] < br[0] - 200.0  # BL.x well left of BR.x

        # Spot-check that the four points form a large square
        width = np.linalg.norm(tr - tl)
        height = np.linalg.norm(bl - tl)
        assert width > 200.0
        assert height > 200.0

    def test_different_versions(self):
        """detect_corners works for versions 1, 2, and 3."""
        for version in (1, 2, 3):
            img = make_qr_image(
                content=f"v{version}", version=version, box_size=10, border=4
            )
            corners, detected_version = detect_corners(img)
            assert corners.shape == (4, 2)
            assert detected_version == version

    def test_colour_input(self):
        """BGR colour images are accepted and converted internally."""
        gray = make_qr_image(content="colour", version=1, box_size=10, border=4)
        bgr = np.stack([gray] * 3, axis=-1)
        corners, version = detect_corners(bgr)
        assert corners.shape == (4, 2)
        assert version == 1


# ---------------------------------------------------------------------------
# detect_homography
# ---------------------------------------------------------------------------


class TestDetectHomography:
    """Tests for detect_homography."""

    def test_returns_3x3_matrix(self):
        """A clean image should yield a 3×3 homography."""
        img = make_qr_image(content="h", version=1, box_size=10, border=4)
        H, version = detect_homography(img)

        assert H.shape == (3, 3)
        assert H.dtype == np.float64
        assert abs(H[2, 2] - 1.0) < 1e-6  # scaled to unit bottom-right
        assert version == 1

    def test_identity_like_for_clean_image(self):
        """For a clean, axis-aligned QR the homography is roughly identity + offset."""
        img = make_qr_image(content="id", version=1, box_size=10, border=4)
        H, _version = detect_homography(img)

        # The QR grid starts at (border*box_size, border*box_size) in image space
        border_px = 4 * 10  # 40
        # Project grid origin (0,0)
        origin = np.array([[0.0, 0.0, 1.0]])
        projected = (H @ origin.T).T
        projected = projected[:, :2] / projected[:, 2:]

        # Should be near the top-left corner of the QR region
        assert np.linalg.norm(projected[0] - np.array([border_px, border_px])) < 5.0

    def test_distorted_image(self):
        """A distorted image still yields a valid homography."""
        img = _generate_distorted_image(version=1, content="warp", seed=2)
        H, version = detect_homography(img)
        assert H.shape == (3, 3)
        assert version == 1

    def test_colour_input(self):
        """BGR colour images are accepted."""
        gray = make_qr_image(content="col", version=1, box_size=10, border=4)
        bgr = np.stack([gray] * 3, axis=-1)
        H, version = detect_homography(bgr)
        assert H.shape == (3, 3)
        assert version == 1


# ---------------------------------------------------------------------------
# detect_sample
# ---------------------------------------------------------------------------


class TestDetectSample:
    """Tests for detect_sample."""

    def test_clean_image_shape(self):
        """A clean V=1 image yields a 21×21 boolean matrix."""
        img = make_qr_image(content="s", version=1, box_size=10, border=4)
        bits = detect_sample(img)

        assert bits.shape == (21, 21)
        assert bits.dtype == bool

    def test_clean_image_has_finder_patterns(self):
        """The sampled matrix should have black finder patterns in the corners."""
        img = make_qr_image(content="fp", version=1, box_size=10, border=4)
        bits = detect_sample(img)

        # TL finder pattern: rows 0..6, cols 0..6 should be mostly dark
        tl = bits[:7, :7]
        assert tl.sum() > 30  # more than half the 49 cells are dark

        # TR finder pattern: rows 0..6, cols 14..20
        tr = bits[:7, 14:21]
        assert tr.sum() > 30

        # BL finder pattern: rows 14..20, cols 0..6
        bl = bits[14:21, :7]
        assert bl.sum() > 30

    def test_version_2_shape(self):
        """A clean V=2 image yields a 25×25 boolean matrix."""
        img = make_qr_image(content="v2", version=2, box_size=10, border=4)
        bits = detect_sample(img)
        assert bits.shape == (25, 25)

    def test_distorted_image(self):
        """A distorted image can still be sampled."""
        img = _generate_distorted_image(version=1, content="sample", seed=3)
        bits = detect_sample(img)
        assert bits.shape == (21, 21)
        assert bits.dtype == bool

    def test_colour_input(self):
        """BGR colour images are accepted."""
        gray = make_qr_image(content="rgb", version=1, box_size=10, border=4)
        bgr = np.stack([gray] * 3, axis=-1)
        bits = detect_sample(bgr)
        assert bits.shape == (21, 21)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for failure modes."""

    def test_blank_image_raises(self):
        """A completely blank image should raise ValueError."""
        img = np.full((200, 200), 255, dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_corners(img)

    def test_random_noise_raises(self):
        """Random noise should raise ValueError."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_homography(img)


# ---------------------------------------------------------------------------
# Consistency across API functions
# ---------------------------------------------------------------------------


class TestConsistency:
    """The three APIs should agree on version and geometry."""

    def test_same_version_across_methods(self):
        """detect_corners, detect_homography, and detect_sample agree on version."""
        img = _generate_distorted_image(version=2, content="agree", seed=5)

        _corners, v1 = detect_corners(img)
        _H, v2 = detect_homography(img)
        bits = detect_sample(img)
        v3 = (bits.shape[0] - 17) // 4

        assert v1 == v2 == v3 == 2

    def test_corners_match_homography_projection(self):
        """Corners from detect_corners should match H projected corners."""
        img = make_qr_image(content="match", version=1, box_size=10, border=4)
        corners, version = detect_corners(img)
        H, _version = detect_homography(img)

        N = 4 * version + 17
        grid_corners = np.array(
            [
                [0.0, 0.0, 1.0],
                [N, 0.0, 1.0],
                [N, N, 1.0],
                [0.0, N, 1.0],
            ]
        )
        projected = (H @ grid_corners.T).T
        projected = projected[:, :2] / projected[:, 2:]

        np.testing.assert_allclose(corners, projected, atol=1e-6)
