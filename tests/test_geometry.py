import numpy as np
import pytest
from qr_reader.geometry import (
    angular_distance,
    point_line_distance,
    max_offset,
    segments_intersect,
    polygon_area,
)

def test_angular_distance():
    p1, p2 = [0, 0], [1, 0]
    q1, q2 = [0, 0], [0, 1]
    # Orthogonal segments
    assert np.isclose(angular_distance(p1, p2, q1, q2), np.pi / 2)

    # Parallel segments
    q1, q2 = [0, 1], [1, 1]
    assert np.isclose(angular_distance(p1, p2, q1, q2), 0.0)

    # Anti-parallel segments
    q1, q2 = [1, 1], [0, 1]
    assert np.isclose(angular_distance(p1, p2, q1, q2), 0.0)

def test_point_line_distance():
    line_p1, line_p2 = [0, 0], [1, 0]
    pt = [0.5, 1.0]
    assert np.isclose(point_line_distance(pt, line_p1, line_p2), 1.0)

    pt = [2.0, 0.0]
    assert np.isclose(point_line_distance(pt, line_p1, line_p2), 0.0)

def test_max_offset():
    p1, p2 = [0, 0], [2, 0]
    q1, q2 = [0, 1], [2, 1]
    # distance between midpoints is 1
    # distance from q to P is 1, p to Q is 1. max is 1.
    # ratio is 1 / 1 = 1.0
    assert np.isclose(max_offset(p1, p2, q1, q2), 1.0)

    q1, q2 = [3, 1], [5, 1]
    # mid_P = [1, 0], mid_Q = [4, 1]
    # dist between midpoints = sqrt(3^2 + 1^2) = sqrt(10)
    # distance from any point to the other line is 1.
    # ratio = 1 / sqrt(10)
    assert np.isclose(max_offset(p1, p2, q1, q2), 1.0 / np.sqrt(10))

def test_segments_intersect():
    p1, p2 = [0, 0], [1, 1]
    q1, q2 = [0, 1], [1, 0]
    assert segments_intersect(p1, p2, q1, q2) is True

    q1, q2 = [0, 1], [0, 2]
    assert segments_intersect(p1, p2, q1, q2) is False

    # Colinear overlapping
    q1, q2 = [0.5, 0.5], [1.5, 1.5]
    assert segments_intersect(p1, p2, q1, q2) is True

    # Colinear non-overlapping
    q1, q2 = [2, 2], [3, 3]
    assert segments_intersect(p1, p2, q1, q2) is False

def test_polygon_area():
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert np.isclose(polygon_area(corners), 1.0)

    corners = [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert np.isclose(polygon_area(corners), 4.0)

    # A rhombus
    corners = [[1, 0], [2, 1], [1, 2], [0, 1]]
    # diagonals are from [1,0] to [1,2] (length 2) and [2,1] to [0,1] (length 2)
    # area = 0.5 * 2 * 2 = 2.0
    assert np.isclose(polygon_area(corners), 2.0)
