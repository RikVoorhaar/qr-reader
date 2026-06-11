import numpy as np
from qr_reader.finder_pattern import (
    FinderPattern,
    extract_finder_patterns,
    check_association,
    find_all_associations,
    find_triplets,
)

def test_extract_finder_patterns():
    # Construct a mock of all_corners:
    # 2 clusters, each with two sets of corners (an inner and an outer)
    # The larger area should be selected.
    c1_small = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) # area 1
    c1_large = np.array([[0, 0], [2, 0], [2, 2], [0, 2]]) # area 4

    c2_small = np.array([[10, 0], [11, 0], [11, 1], [10, 1]]) # area 1
    c2_large = np.array([[10, 0], [12, 0], [12, 2], [10, 2]]) # area 4

    all_corners = [
        (0, c1_small),
        (0, c1_large),
        (1, c2_large),
        (1, c2_small),
    ]

    fps = extract_finder_patterns(all_corners)
    assert len(fps) == 2

    fp0 = next(fp for fp in fps if fp.cluster_idx == 0)
    fp1 = next(fp for fp in fps if fp.cluster_idx == 1)

    assert np.allclose(fp0.outer_corners, c1_large)
    assert np.allclose(fp1.outer_corners, c2_large)

def test_check_association():
    # Two aligned finder patterns (horizontally)
    fp1 = FinderPattern(
        cluster_idx=0,
        outer_corners=np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    )
    fp2 = FinderPattern(
        cluster_idx=1,
        outer_corners=np.array([[10, 0], [12, 0], [12, 2], [10, 2]])
    )

    assoc = check_association(fp1, fp2)
    assert assoc is not None
    assert assoc.fp1_idx == 0
    assert assoc.fp2_idx == 1
    # For horizontal alignment, top and bottom segments should be colinear.
    # Segments: 0-1 (top), 1-2 (right), 2-3 (bottom), 3-0 (left)
    # The top segment is [0, 2] -> [12, 0] ??? No, wait.
    # [0, 0] to [2, 0] is the top segment. That's index 0.
    # [2, 2] to [0, 2] is the bottom segment. That's index 2.
    assert set(assoc.colinear_segments_1) == {0, 2}
    assert set(assoc.colinear_segments_2) == {0, 2}

    # Pathological case: intersecting
    fp3 = FinderPattern(
        cluster_idx=2,
        outer_corners=np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
    )
    assoc_int = check_association(fp1, fp3)
    assert assoc_int is None

    # Non-aligned case
    fp4 = FinderPattern(
        cluster_idx=3,
        outer_corners=np.array([[10, 5], [12, 5], [12, 7], [10, 7]])
    )
    assoc_unaligned = check_association(fp1, fp4)
    assert assoc_unaligned is None

def test_find_triplets():
    # Top-Left at (0, 0)
    fp_tl = FinderPattern(cluster_idx=1, outer_corners=np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
    # Top-Right at (10, 0)
    fp_tr = FinderPattern(cluster_idx=0, outer_corners=np.array([[10, 0], [12, 0], [12, 2], [10, 2]]))
    # Bottom-Left at (0, 10)
    fp_bl = FinderPattern(cluster_idx=2, outer_corners=np.array([[0, 10], [2, 10], [2, 12], [0, 12]]))

    fps = [fp_tr, fp_tl, fp_bl]
    associations = find_all_associations(fps)
    assert len(associations) == 2

    triplets = find_triplets(fps, associations)
    assert len(triplets) == 1
    t = triplets[0]

    assert t.top_left_idx == 1
    assert t.top_right_idx == 0
    assert t.bottom_left_idx == 2
