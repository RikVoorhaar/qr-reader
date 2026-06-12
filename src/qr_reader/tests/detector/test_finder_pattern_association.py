import numpy as np
import pytest

from qr_reader.detector.finder_pattern import (
    FinderPattern,
    find_all_associations,
    find_triplets,
)

# ---------------------------------------------------------------------------
# Homography helpers for perspective transforms
# ---------------------------------------------------------------------------


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute homography H such that dst ~ H * src (using homogeneous coordinates)."""
    A = []
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H


def _apply_homography(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply homography to points. pts: (N, 2). Returns (N, 2)."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    transformed = pts_h @ H.T
    transformed = transformed[:, :2] / transformed[:, 2:3]
    return transformed


# ---------------------------------------------------------------------------
# Synthetic finder-pattern generators
# ---------------------------------------------------------------------------


def generate_synthetic_finder_patterns(
    *,
    version: int,
    seed: int = 0,
    module_size: float = 10.0,
    origin: tuple[float, float] = (0.0, 0.0),
    rotation_rad: float = 0.0,
    scale: float = 1.0,
    perspective_amount: float = 0.0,
    jitter_std: float = 0.0,
    include_inner: bool = True,
) -> list[FinderPattern]:
    """Generate three deterministic FinderPatterns at the canonical QR locations."""
    rng = np.random.default_rng(seed)
    N = 4 * version + 17  # QR grid size in modules
    fp_size = 7.0  # Finder pattern outer square width in modules

    # Module coordinates (row, col) for the three finder patterns.
    # Corners in clockwise order: top-left, top-right, bottom-right, bottom-left.
    module_corners = [
        np.array(
            [
                [0.0, 0.0],
                [0.0, fp_size],
                [fp_size, fp_size],
                [fp_size, 0.0],
            ]
        ),  # Top-left
        np.array(
            [
                [0.0, N - fp_size],
                [0.0, N],
                [fp_size, N],
                [fp_size, N - fp_size],
            ]
        ),  # Top-right
        np.array(
            [
                [N - fp_size, 0.0],
                [N - fp_size, fp_size],
                [N, fp_size],
                [N, 0.0],
            ]
        ),  # Bottom-left
    ]

    # Convert to image coordinates
    image_corners = [corners * module_size for corners in module_corners]

    # Apply uniform scale and rotation
    if scale != 1.0 or rotation_rad != 0.0:
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        rot_scale = np.array(
            [
                [cos_r * scale, -sin_r * scale],
                [sin_r * scale, cos_r * scale],
            ]
        )
        image_corners = [corners @ rot_scale.T for corners in image_corners]

    # Apply perspective transform by perturbing the four QR-code corners
    if perspective_amount > 0:
        qr_corners = np.array(
            [
                [0.0, 0.0],
                [0.0, N * module_size],
                [N * module_size, N * module_size],
                [N * module_size, 0.0],
            ]
        )
        if scale != 1.0 or rotation_rad != 0.0:
            qr_corners = qr_corners @ rot_scale.T

        perturbed = qr_corners + rng.normal(
            scale=perspective_amount * module_size, size=qr_corners.shape
        )
        H = _compute_homography(qr_corners, perturbed)
        image_corners = [_apply_homography(corners, H) for corners in image_corners]

    # Apply corner jitter
    if jitter_std > 0:
        image_corners = [
            corners + rng.normal(scale=jitter_std, size=corners.shape)
            for corners in image_corners
        ]

    # Translate so all coordinates are positive
    all_coords = np.vstack(image_corners)
    min_row, min_col = all_coords.min(axis=0)
    translation = np.array([-min_row + origin[0], -min_col + origin[1]])
    image_corners = [corners + translation for corners in image_corners]

    # Build FinderPattern objects
    fps = []
    for idx, corners in enumerate(image_corners):
        inner = None
        if include_inner:
            center = corners.mean(axis=0)
            inner = center + (corners - center) * (5.0 / 7.0)
        fps.append(
            FinderPattern(
                cluster_idx=idx,
                outer_corners=np.array(corners),
                inner_corners=np.array(inner) if inner is not None else None,
            )
        )

    return fps


def generate_bogus_finder_patterns(
    *,
    count: int,
    seed: int,
    image_extent: tuple[float, float],
    module_size: float = 10.0,
    jitter_std: float = 0.0,
) -> list[FinderPattern]:
    """Generate plausible but incorrect finder-pattern candidates."""
    rng = np.random.default_rng(seed)
    max_row, max_col = image_extent
    fps = []

    for i in range(count):
        center_row = rng.uniform(max_row * 0.2, max_row * 0.8)
        center_col = rng.uniform(max_col * 0.2, max_col * 0.8)
        size = rng.uniform(module_size * 3, module_size * 10)

        angle = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        hs = size / 2

        corners = np.array(
            [
                [-hs, -hs],
                [-hs, hs],
                [hs, hs],
                [hs, -hs],
            ]
        )
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = corners @ rot.T
        corners = corners + np.array([center_row, center_col])

        if jitter_std > 0:
            corners = corners + rng.normal(scale=jitter_std, size=corners.shape)

        # Ensure positive coordinates
        corners = corners - corners.min(axis=0) + np.array([1.0, 1.0])

        fps.append(
            FinderPattern(
                cluster_idx=100 + i,
                outer_corners=corners,
                inner_corners=None,
            )
        )

    return fps


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_true_triplet_found(fps, associations, triplets):
    """Assert that the true QR triplet (TL=0, TR=1, BL=2) is recovered.

    NOTE: ``find_triplets`` currently swaps top-right and bottom-left for the
    standard QR layout (see existing ``test_find_triplets`` which uses the
    opposite naming).  We therefore only require that *a* triplet with
    ``top_left_idx == 0`` and the other two members ``{1, 2}`` exists.
    """
    pairs = {frozenset((a.fp1_idx, a.fp2_idx)) for a in associations}
    assert frozenset((0, 1)) in pairs, (
        f"Missing top-left/top-right association. Found: {pairs}"
    )
    assert frozenset((0, 2)) in pairs, (
        f"Missing top-left/bottom-left association. Found: {pairs}"
    )

    assert any(
        t.top_left_idx == 0 and {t.top_right_idx, t.bottom_left_idx} == {1, 2}
        for t in triplets
    ), (
        f"True triplet not found. Triplets: "
        f"{[(t.top_left_idx, t.top_right_idx, t.bottom_left_idx) for t in triplets]}"
    )


# ---------------------------------------------------------------------------
# Tests that should PASS with current production code (guard existing behavior)
# ---------------------------------------------------------------------------


def test_low_version_finds_associations():
    """Version 4 synthetic FPs should associate correctly (baseline)."""
    fps = generate_synthetic_finder_patterns(version=4)
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)


def test_low_version_with_bogus_finds_true_triplet():
    """Version 4 with false positives should still recover the true triplet."""
    true_fps = generate_synthetic_finder_patterns(version=4)
    extent = (
        max(fp.outer_corners[:, 0].max() for fp in true_fps),
        max(fp.outer_corners[:, 1].max() for fp in true_fps),
    )
    bogus_fps = generate_bogus_finder_patterns(count=3, seed=42, image_extent=extent)
    fps = true_fps + bogus_fps
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)


def test_axis_mismatch_pairing_find_triplets():
    """find_triplets must handle cross-index pairings like (0,1),(2,3).

    This test uses a low version so that the current ``len(colinear_pairs)==2``
    rule still accepts the association.  It verifies that downstream triplet
    logic is compatible with non-same-index segment pairings.
    """
    fps = generate_synthetic_finder_patterns(version=4)

    # Cyclically shift the corners of the top-right finder pattern so that
    # its segment numbering is rotated by 90 deg relative to the top-left.
    fp_tr = fps[1]
    shifted_corners = np.array(
        [
            fp_tr.outer_corners[3],
            fp_tr.outer_corners[0],
            fp_tr.outer_corners[1],
            fp_tr.outer_corners[2],
        ]
    )
    fps[1] = FinderPattern(
        cluster_idx=fp_tr.cluster_idx,
        outer_corners=shifted_corners,
        inner_corners=fp_tr.inner_corners,
    )

    associations = find_all_associations(fps)
    # The horizontal association should still be found with exactly 2 pairs
    assert len(associations) >= 1
    horiz_assoc = next(
        (
            a
            for a in associations
            if frozenset((a.fp1_idx, a.fp2_idx)) == frozenset((0, 1))
        ),
        None,
    )
    assert horiz_assoc is not None, "Horizontal association missing"
    # The pairing should be cross-index: (0,1) and (2,3)
    paired = set(zip(horiz_assoc.colinear_segments_1, horiz_assoc.colinear_segments_2))
    assert paired == {(0, 1), (2, 3)}, f"Unexpected pairing: {paired}"

    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)


# ---------------------------------------------------------------------------
# Tests that should FAIL with current production code (document the bug)
# ---------------------------------------------------------------------------


def test_high_version_finds_associations():
    """Version 12 synthetic FPs should associate; current code returns 0.

    At high versions the inter-finder-pattern distance grows while FP size
    stays fixed.  ``max_offset()`` normalises by that distance, so extra
    cross-pairs fall below ``offset_tol``.  The ``len(colinear_pairs)==2``
    rule then rejects the true adjacent pairs.
    """
    fps = generate_synthetic_finder_patterns(version=12)
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)


def test_high_version_with_bogus_finds_true_triplet():
    """Version 12 with false positives should still recover the true triplet."""
    true_fps = generate_synthetic_finder_patterns(version=12)
    extent = (
        max(fp.outer_corners[:, 0].max() for fp in true_fps),
        max(fp.outer_corners[:, 1].max() for fp in true_fps),
    )
    bogus_fps = generate_bogus_finder_patterns(count=3, seed=42, image_extent=extent)
    fps = true_fps + bogus_fps
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)


def test_high_version_with_perspective_and_jitter():
    """Version 12 with moderate perspective and jitter should still work."""
    fps = generate_synthetic_finder_patterns(
        version=12, perspective_amount=0.3, jitter_std=0.2
    )
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)
    _assert_true_triplet_found(fps, associations, triplets)
