from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from qr_reader.detector.geometry import (
    angular_distance,
    local_offset,
    polygon_area,
    segments_intersect,
)


@dataclass
class FinderPattern:
    cluster_idx: int
    outer_corners: np.ndarray  # shape (4, 2), the outer 7×7 square in (row, col)
    inner_corners: np.ndarray | None = (
        None  # shape (4, 2), the inner white-ring square (offset 1) or None
    )
    # Segment definitions: 0-1, 1-2, 2-3, 3-0

    def segments(self):
        """Returns 4 segments of the outer corners."""
        c = self.outer_corners
        return [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]


# Deprecated: replaced by find_valid_triplets in the main pipeline
def extract_finder_patterns(
    all_corners: List[Tuple[int, np.ndarray]],
) -> List[FinderPattern]:
    """
    Groups corners by cluster_idx and selects the one with the maximum area
    as the outer_corners for the FinderPattern.
    """
    cluster_groups = {}
    for ci, corners in all_corners:
        if ci not in cluster_groups:
            cluster_groups[ci] = []
        cluster_groups[ci].append(corners)

    fps = []
    for ci, corners_list in cluster_groups.items():
        # Sort candidate quads by area descending
        areas = [(polygon_area(c), c) for c in corners_list]
        areas.sort(key=lambda x: x[0], reverse=True)

        # TODO: a check that the inner pattern is contained in the outer pattern is nice as method to reject false positives

        if len(areas) == 0:
            continue

        outer_corners = np.array(areas[0][1])
        inner_corners = np.array(areas[1][1]) if len(areas) >= 2 else None

        fps.append(
            FinderPattern(
                cluster_idx=ci,
                outer_corners=outer_corners,
                inner_corners=inner_corners,
            )
        )

    return fps


@dataclass
class Association:
    fp1_idx: int
    fp2_idx: int
    colinear_segments_1: List[int]  # Indices of segments in fp1
    colinear_segments_2: List[int]  # Indices of segments in fp2


def check_association(
    fp1: FinderPattern, fp2: FinderPattern, angle_tol=0.1, offset_tol=0.30
) -> Optional[Association]:
    """
    Checks if fp1 and fp2 are associated (aligned).
    Returns an Association if they are, else None.
    """
    segs1 = fp1.segments()
    segs2 = fp2.segments()

    # Check if the finder patterns intersect (pathological case)
    # We do a simple AABB intersection or just check if any segments intersect
    for s1 in segs1:
        for s2 in segs2:
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                return None

    candidates = []
    axes = ((0, 2), (1, 3))

    for axis1 in axes:
        for axis2 in axes:
            # Try both one-to-one pairings between opposite sides. Corner ordering
            # may differ between finder patterns, so matching segment indices is
            # not guaranteed to be the best valid pairing.
            pairings = (
                ((axis1[0], axis2[0]), (axis1[1], axis2[1])),
                ((axis1[0], axis2[1]), (axis1[1], axis2[0])),
            )

            for pairing in pairings:
                angles = []
                offsets = []
                scores = []
                pairing_ok = True

                for i, j in pairing:
                    s1 = segs1[i]
                    s2 = segs2[j]
                    ang_dist = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
                    off = float(local_offset(s1[0], s1[1], s2[0], s2[1]))

                    if ang_dist >= angle_tol or off >= offset_tol:
                        pairing_ok = False
                        break

                    angles.append(ang_dist)
                    offsets.append(off)
                    scores.append(ang_dist + off)

                if pairing_ok:
                    candidates.append(
                        (
                            max(offsets),
                            sum(scores),
                            Association(
                                fp1_idx=fp1.cluster_idx,
                                fp2_idx=fp2.cluster_idx,
                                colinear_segments_1=[p[0] for p in pairing],
                                colinear_segments_2=[p[1] for p in pairing],
                            ),
                        )
                    )

    if not candidates:
        return None

    return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


# Deprecated: replaced by find_valid_triplets in the main pipeline
def find_all_associations(
    fps: List[FinderPattern], angle_tol=0.1, offset_tol=0.30
) -> List[Association]:
    associations = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            assoc = check_association(fps[i], fps[j], angle_tol, offset_tol)
            if assoc is not None:
                associations.append(assoc)
    return associations


@dataclass
class Triplet:
    top_left_idx: int
    top_right_idx: int
    bottom_left_idx: int


# Deprecated: replaced by find_valid_triplets in the main pipeline
def find_triplets(
    fps: List[FinderPattern], associations: List[Association]
) -> List[Triplet]:
    """
    Finds triplets of finder patterns that form an L-shape, identifying the top-left corner.
    Returns a list of Triplets.
    """
    # Build an adjacency list
    from collections import defaultdict

    adj = defaultdict(list)
    for a in associations:
        adj[a.fp1_idx].append((a.fp2_idx, a))
        adj[a.fp2_idx].append((a.fp1_idx, a))

    triplets = []

    # Iterate through all possible B (center of L-shape)
    for b_idx in adj:
        neighbors = adj[b_idx]
        if len(neighbors) < 2:
            continue

        # Check all pairs of neighbors for B
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a_idx, assoc_ab = neighbors[i]
                c_idx, assoc_bc = neighbors[j]

                # A and C should not be associated directly
                ac_associated = False
                for a in associations:
                    if (a.fp1_idx == a_idx and a.fp2_idx == c_idx) or (
                        a.fp1_idx == c_idx and a.fp2_idx == a_idx
                    ):
                        ac_associated = True
                        break

                if ac_associated:
                    continue

                # The segments used in B for AB and BC should be different.
                # For AB association, what segments of B are colinear?
                if assoc_ab.fp1_idx == b_idx:
                    b_segs_ab = set(assoc_ab.colinear_segments_1)
                else:
                    b_segs_ab = set(assoc_ab.colinear_segments_2)

                if assoc_bc.fp1_idx == b_idx:
                    b_segs_bc = set(assoc_bc.colinear_segments_1)
                else:
                    b_segs_bc = set(assoc_bc.colinear_segments_2)

                if len(b_segs_ab.intersection(b_segs_bc)) > 0:
                    # They share a colinear segment axis, so they are in a straight line, not an L-shape.
                    continue

                # Determine which is top-right and which is bottom-left.
                # We can do this using the vectors from B to A and B to C.
                # Cross product (BA x BC) tells us the orientation.
                # Since image coordinates have y pointing down:
                # Top-Right (x positive), Bottom-Left (y positive)
                # Let's get centroids
                def get_centroid(fp_idx):
                    for fp in fps:
                        if fp.cluster_idx == fp_idx:
                            return fp.outer_corners.mean(axis=0)
                    return np.zeros(2)

                b_center = get_centroid(b_idx)
                a_center = get_centroid(a_idx)
                c_center = get_centroid(c_idx)

                vec_ba = a_center - b_center
                vec_bc = c_center - b_center

                v_ba_xy = vec_ba[::-1]
                v_bc_xy = vec_bc[::-1]
                cross = float(v_ba_xy[0] * v_bc_xy[1] - v_ba_xy[1] * v_bc_xy[0])

                if cross > 0:
                    # BA is to the right of BC (counter-clockwise from BC to BA)
                    # Because y points down, x points right.
                    # Top-right is +x, bottom-left is +y.
                    # Top-right x Bottom-left = (+x) x (+y) = 1*0 - 0*1 = 0
                    # Wait: vec_TR = (1, 0), vec_BL = (0, 1)
                    # TR x BL = 1*1 - 0*0 = 1 > 0
                    # So if cross > 0, BA is TR and BC is BL
                    top_right = a_idx
                    bottom_left = c_idx
                else:
                    top_right = c_idx
                    bottom_left = a_idx

                triplets.append(
                    Triplet(
                        top_left_idx=b_idx,
                        top_right_idx=top_right,
                        bottom_left_idx=bottom_left,
                    )
                )

    return triplets


def find_valid_triplets(
    fps: list[FinderPattern],
    score_map: dict[int, float],
    module_size_tol: float = 0.3,
    min_module_size: float = 2.0,
    dist_min: float = 2.0,
    dist_max: float = 200.0,
) -> list[Triplet]:
    """Find triplets using centre geometry, axis alignment, and module-size compatibility.

    Replaces the ``find_all_associations → find_triplets`` pipeline when fine-grained
    per-finder geometry is available from the finder corners.  The algorithm:

    1. **Pair connectivity** — Two finders are connected if their module sizes are
       compatible (``|mi - mj| / max(mi, mj) < module_size_tol``), their centres are
       within ``[dist_min * m_avg, dist_max * m_avg]``, and the inter-centre vector
       is approximately aligned with one of the finder's corner-edge directions.

    2. **Triplet discovery** — A finder with >= 2 neighbours forms a candidate triplet
       when its two neighbours are not directly connected and the angle at the centre
       is within 15° of a right angle.

    3. **Orientation resolution** — The top-left, top-right, and bottom-left roles are
       assigned via a cross-product check on the inter-centre vectors.

    Args:
        fps: Finder patterns (already deduplicated).
        score_map: Fit quality scores keyed by ``cluster_idx`` (unused here, kept
            for future compatibility).
        module_size_tol: Maximum relative module-pitch difference.
        dist_min: Minimum inter-centre distance as a multiple of average module pitch.
        dist_max: Maximum inter-centre distance as a multiple of average module pitch.
        min_module_size: Minimum module pitch (px) for a valid finder pattern.

    Returns:
        List of ``Triplet`` objects (TL, TR, BL roles resolved).
    """
    n = len(fps)
    if n < 3:
        return []

    idx_to_pos = {fp.cluster_idx: i for i, fp in enumerate(fps)}
    centers_rc = np.array([fp.outer_corners.mean(axis=0) for fp in fps])

    # Pre-compute per-finder m and axes from corners
    finder_m: dict[int, float] = {}
    finder_e1: dict[int, np.ndarray] = {}
    finder_e2: dict[int, np.ndarray] = {}
    for fp in fps:
        ci = fp.cluster_idx
        corners_rc = fp.outer_corners
        # Module pitch: mean side length / 7
        sides = []
        for k in range(4):
            v = corners_rc[(k + 1) % 4] - corners_rc[k]
            sides.append(float(np.linalg.norm(v)))
        finder_m[ci] = float(np.mean(sides)) / 7.0

        # Axis directions from corners (in (col, row) = (x, y) order)
        corners_xy = corners_rc[:, ::-1]
        e1_xy = corners_xy[1] - corners_xy[0]
        e2_xy = corners_xy[3] - corners_xy[0]
        n1 = float(np.linalg.norm(e1_xy))
        n2 = float(np.linalg.norm(e2_xy))
        if n1 > 1e-9:
            e1_xy = e1_xy / n1
        if n2 > 1e-9:
            e2_xy = e2_xy / n2
        finder_e1[ci] = e1_xy
        finder_e2[ci] = e2_xy

    adj: dict[int, list[int]] = {fp.cluster_idx: [] for fp in fps}

    for i in range(n):
        idx_i = fps[i].cluster_idx
        mi = finder_m[idx_i]
        ci_rc = centers_rc[i]
        e1_i = finder_e1[idx_i]
        e2_i = finder_e2[idx_i]

        for j in range(i + 1, n):
            idx_j = fps[j].cluster_idx
            mj = finder_m[idx_j]

            if mi < min_module_size or mj < min_module_size:
                continue

            if abs(mi - mj) / max(mi, mj) > module_size_tol:
                continue

            cj_rc = centers_rc[j]
            delta_rc = cj_rc - ci_rc
            dist = float(np.linalg.norm(delta_rc))
            m_avg = (mi + mj) / 2.0

            if dist < dist_min * m_avg or dist > dist_max * m_avg:
                continue

            if dist < 1e-9:
                continue
            delta_xy = delta_rc[::-1]
            delta_unit = delta_xy / dist

            dot1 = abs(float(np.dot(delta_unit, e1_i)))
            dot2 = abs(float(np.dot(delta_unit, e2_i)))
            parallel_score = max(dot1, dot2)
            perp_score = min(dot1, dot2)

            if parallel_score < 0.9 or perp_score > 0.25 * parallel_score:
                continue

            adj[idx_i].append(idx_j)
            adj[idx_j].append(idx_i)

    triplets: list[Triplet] = []
    for b_idx in adj:
        neighbors = adj[b_idx]
        if len(neighbors) < 2:
            continue
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                a_idx = neighbors[ni]
                c_idx = neighbors[nj]

                if a_idx in adj.get(c_idx, []) or c_idx in adj.get(a_idx, []):
                    continue

                ca = centers_rc[idx_to_pos[a_idx]]
                cb = centers_rc[idx_to_pos[b_idx]]
                cc = centers_rc[idx_to_pos[c_idx]]

                vec_ba = ca - cb
                vec_bc = cc - cb

                dot = float(np.dot(vec_ba, vec_bc))
                norm = float(np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc))
                if norm < 1e-9:
                    continue
                cos_angle = np.clip(dot / norm, -1.0, 1.0)
                angle = float(np.arccos(cos_angle))

                if abs(angle - np.pi / 2) > np.deg2rad(15):
                    continue

                ma = finder_m[a_idx]
                mb = finder_m[b_idx]
                mc = finder_m[c_idx]
                m_max = max(ma, mb, mc)
                m_min = min(ma, mb, mc)
                if m_max < 1e-9 or (m_max - m_min) / m_max > module_size_tol:
                    continue

                v_ba_xy = vec_ba[::-1]
                v_bc_xy = vec_bc[::-1]
                cross = float(v_ba_xy[0] * v_bc_xy[1] - v_ba_xy[1] * v_bc_xy[0])
                if cross > 0:
                    top_right = a_idx
                    bottom_left = c_idx
                else:
                    top_right = c_idx
                    bottom_left = a_idx

                triplets.append(
                    Triplet(
                        top_left_idx=b_idx,
                        top_right_idx=top_right,
                        bottom_left_idx=bottom_left,
                    )
                )

    return triplets
