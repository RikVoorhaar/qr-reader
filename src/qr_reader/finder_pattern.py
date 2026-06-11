from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from qr_reader.geometry import (
    angular_distance,
    max_offset,
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
    fp1: FinderPattern, fp2: FinderPattern, angle_tol=0.1, offset_tol=0.15
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

    colinear_pairs = []

    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            ang_dist = angular_distance(s1[0], s1[1], s2[0], s2[1])
            # Lines could be parallel or anti-parallel.
            # angular_distance returns acute angle, so it's between 0 and pi/2.
            if ang_dist < angle_tol:
                # They are roughly parallel. Check offset.
                off = max_offset(s1[0], s1[1], s2[0], s2[1])
                if off < offset_tol:
                    colinear_pairs.append((i, j))

    # We expect exactly 2 pairs of colinear segments if they are properly aligned
    if len(colinear_pairs) == 2:
        return Association(
            fp1_idx=fp1.cluster_idx,
            fp2_idx=fp2.cluster_idx,
            colinear_segments_1=[p[0] for p in colinear_pairs],
            colinear_segments_2=[p[1] for p in colinear_pairs],
        )
    return None


def find_all_associations(
    fps: List[FinderPattern], angle_tol=0.1, offset_tol=0.15
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

                # Cross product in 2D
                cross = vec_ba[0] * vec_bc[1] - vec_ba[1] * vec_bc[0]

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
