import numpy as np

def segment_vector(p1, p2):
    return np.array(p2) - np.array(p1)

def angular_distance(p1, p2, q1, q2):
    """Computes the acute angle (in radians) between two line segments P(p1, p2) and Q(q1, q2)."""
    v1 = segment_vector(p1, p2)
    v2 = segment_vector(q1, q2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    # We want the acute angle, so if angle > pi/2, we take pi - angle
    if angle > np.pi / 2:
        angle = np.pi - angle
    return angle

def point_line_distance(pt, line_p1, line_p2):
    """Computes the orthogonal distance from point `pt` to the infinite line defined by `line_p1` and `line_p2`."""
    pt = np.array(pt)
    p1 = np.array(line_p1)
    p2 = np.array(line_p2)

    # Distance from point to line: ||(p2 - p1) x (p1 - pt)|| / ||p2 - p1||
    # In 2D, cross product scalar is |(p2_x - p1_x)*(p1_y - pt_y) - (p1_x - pt_x)*(p2_y - p1_y)|
    num = np.abs((p2[0] - p1[0]) * (p1[1] - pt[1]) - (p1[0] - pt[0]) * (p2[1] - p1[1]))
    den = np.linalg.norm(p2 - p1)
    if den == 0:
        return np.linalg.norm(pt - p1)
    return num / den

def max_offset(p1, p2, q1, q2):
    """
    Computes the max offset between two line segments as:
    max(d(q1,P), d(q2,P), d(p1,Q), d(p2,Q)) / L,
    where L is the distance between their midpoints.
    P is the infinite line passing through p1 and p2.
    Q is the infinite line passing through q1 and q2.
    """
    p1, p2, q1, q2 = np.array(p1), np.array(p2), np.array(q1), np.array(q2)

    d_q1_P = point_line_distance(q1, p1, p2)
    d_q2_P = point_line_distance(q2, p1, p2)
    d_p1_Q = point_line_distance(p1, q1, q2)
    d_p2_Q = point_line_distance(p2, q1, q2)

    max_d = max(d_q1_P, d_q2_P, d_p1_Q, d_p2_Q)

    mid_P = (p1 + p2) / 2
    mid_Q = (q1 + q2) / 2
    L = np.linalg.norm(mid_Q - mid_P)

    if L == 0:
        return float('inf')  # Prevent division by zero

    return max_d / L

def segments_intersect(p1, p2, q1, q2):
    """
    Checks if line segment p1p2 intersects with line segment q1q2.
    Uses the cross product orientation method.
    """
    def orientation(a, b, c):
        # > 0: counterclockwise, < 0: clockwise, == 0: colinear
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

    p1, p2, q1, q2 = np.array(p1), np.array(p2), np.array(q1), np.array(q2)

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Colinear cases
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True

    return False

def polygon_area(corners):
    """
    Computes the area of a polygon given its corners (in order).
    Works for 4 corners (quadrilateral) like finder patterns.
    """
    corners = np.array(corners)
    if len(corners) == 4:
        # Use the formula from dev3.py for 4 corners (diagonals cross product)
        # 0.5 * |d1 x d2|
        diag1 = corners[0] - corners[2]
        diag2 = corners[1] - corners[3]
        return 0.5 * np.abs(np.linalg.det(np.vstack([diag1, diag2])))
    else:
        # General Shoelace formula
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
