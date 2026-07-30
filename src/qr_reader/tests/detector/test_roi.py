"""Tests for detector/roi.py — cluster_to_bbox and cutout."""

import numpy as np
import pytest

from qr_reader.detector.clustering import CandidateCluster
from qr_reader.detector.roi import cluster_to_bbox, cutout

# ---------------------------------------------------------------------------
# cluster_to_bbox
# ---------------------------------------------------------------------------


def _make_cluster(
    row=100.0,
    left_outer=50,
    left_inner=55,
    left_center=60,
    right_center=70,
    right_inner=75,
    right_outer=80,
    num_candidates=3,
):
    """Helper to build a CandidateCluster with explicit column boundaries."""
    cols = np.array(
        [left_outer, left_inner, left_center, right_center, right_inner, right_outer],
        dtype=np.float64,
    )
    height = float(right_center - left_center)
    return CandidateCluster(
        row=row, cols=cols, height=height, num_candidates=num_candidates
    )


def test_bbox_is_centered_on_cluster():
    """The bbox center should match the cluster center (row, mid of cols[2:3])."""
    c = _make_cluster(row=100.0, left_center=60, right_center=70)
    r0, r1, c0, c1 = cluster_to_bbox(c, scale=1.0)
    center_r = (r0 + r1) / 2
    center_c = (c0 + c1) / 2
    assert center_r == pytest.approx(100.0)
    assert center_c == pytest.approx(65.0)


def test_bbox_squareness():
    """Bbox is square — half-width equals half-height — when half_extent uses max(width, height)."""
    c = _make_cluster(left_outer=10, right_outer=110, left_center=50, right_center=70)
    # width = 100, height = 20 → half_extent = 50
    r0, r1, c0, c1 = cluster_to_bbox(c, scale=1.0)
    half_h = (r1 - r0) / 2
    half_w = (c1 - c0) / 2
    assert half_h == pytest.approx(50.0)
    assert half_w == pytest.approx(50.0)


def test_bbox_scale():
    """Scale linearly expands the bbox."""
    c = _make_cluster()
    r0_1, r1_1, c0_1, c1_1 = cluster_to_bbox(c, scale=1.0)
    r0_2, r1_2, c0_2, c1_2 = cluster_to_bbox(c, scale=2.0)
    h_1 = r1_1 - r0_1
    h_2 = r1_2 - r0_2
    assert h_2 == pytest.approx(2 * h_1)
    w_1 = c1_1 - c0_1
    w_2 = c1_2 - c0_2
    assert w_2 == pytest.approx(2 * w_1)


def test_bbox_scale_default():
    """Default scale is 1.5."""
    c = _make_cluster()
    r0, r1, c0, c1 = cluster_to_bbox(c)
    r0_1, r1_1, c0_1, c1_1 = cluster_to_bbox(c, scale=1.0)
    h = r1 - r0
    h_1 = r1_1 - r0_1
    assert h == pytest.approx(1.5 * h_1)


def test_bbox_integer_bounds():
    """cluster_to_bbox returns integer bounds."""
    r0, r1, c0, c1 = cluster_to_bbox(_make_cluster())
    assert all(isinstance(v, int) for v in (r0, r1, c0, c1))


def test_bbox_tall_cluster():
    """When height > width, half_extent is driven by height."""
    c = _make_cluster(left_outer=55, right_outer=65, left_center=58, right_center=62)
    # width = 10, height = 4 → half_extent = 5
    r0, r1, c0, c1 = cluster_to_bbox(c, scale=1.0)
    assert (r1 - r0) == pytest.approx(10.0)
    assert (c1 - c0) == pytest.approx(10.0)


def test_bbox_wide_cluster():
    """When width > height, half_extent is driven by width."""
    c = _make_cluster(left_outer=0, right_outer=100, left_center=45, right_center=55)
    # width = 100, height = 10 → half_extent = 50
    r0, r1, c0, c1 = cluster_to_bbox(c, scale=1.0)
    assert (r1 - r0) == pytest.approx(100.0)
    assert (c1 - c0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# cutout
# ---------------------------------------------------------------------------


def test_cutout_shape():
    """cutout returns the region defined by the bbox."""
    img = np.arange(200, dtype=np.uint8).reshape(10, 20)
    bbox = (2, 5, 3, 8)
    sub = cutout(img, bbox)
    assert sub.shape == (3, 5)
    np.testing.assert_array_equal(sub, img[2:5, 3:8])


def test_cutout_clamps_negative_bounds():
    """Negative bbox coords are clamped to 0."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    bbox = (-5, 5, -3, 8)
    sub = cutout(img, bbox)
    assert sub.shape == (5, 8)
    np.testing.assert_array_equal(sub, img[0:5, 0:8])


def test_cutout_clamps_over_bounds():
    """Bbox extending past the image is clamped."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    bbox = (5, 20, 3, 25)
    sub = cutout(img, bbox)
    assert sub.shape == (5, 7)
    np.testing.assert_array_equal(sub, img[5:10, 3:10])


def test_cutout_entirely_outside():
    """Bbox entirely outside the image returns empty slice."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    bbox = (50, 60, 70, 80)
    sub = cutout(img, bbox)
    assert sub.size == 0
    assert sub.shape[0] == 0 or sub.shape[1] == 0


def test_cutout_returns_view():
    """cutout should return a view, not a copy (unless clamping modified bounds)."""
    img = np.arange(200, dtype=np.uint8).reshape(10, 20)
    bbox = (2, 5, 3, 8)
    sub = cutout(img, bbox)
    sub[0, 0] = 255
    assert img[2, 3] == 255  # mutation reflected in original


def test_cutout_clamped_not_view():
    """When clamping occurs, cutout preserves original data but can't be a view of exact same region if bbox was invalid."""
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    original_val = img[0, 0]
    bbox = (-5, 5, -3, 8)
    sub = cutout(img, bbox)
    # sub starts at (0,0) in original, so it IS a view
    sub[0, 0] = 255
    # But original[0,0] now changed because clamping produced a valid view into the same region
    assert img[0, 0] != original_val


# ---------------------------------------------------------------------------
# Integration: cluster_to_bbox → cutout
# ---------------------------------------------------------------------------


def test_round_trip():
    """cluster_to_bbox + cutout work together end-to-end."""
    img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)
    c = _make_cluster(
        row=150.0,
        left_outer=100,
        left_inner=110,
        left_center=120,
        right_center=140,
        right_inner=150,
        right_outer=160,
    )
    bbox = cluster_to_bbox(c, scale=1.2)
    sub = cutout(img, bbox)
    assert sub.ndim == 2
    assert sub.size > 0
    # The cutout should contain the cluster center
    center_r_in_sub = 150.0 - bbox[0]
    center_c_in_sub = 130.0 - bbox[2]
    assert 0 <= center_r_in_sub < sub.shape[0]
    assert 0 <= center_c_in_sub < sub.shape[1]
