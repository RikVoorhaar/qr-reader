"""High-level QR detector API.

Provides three convenience functions that work on a single-QR image:

* ``detect_corners``  →  image corners + version
* ``detect_homography`` → homography matrix + version
* ``detect_sample``   →  sampled bit matrix

All functions assume the input contains **one** QR code.
"""

from __future__ import annotations

import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.corner import angular_nms_top_radial_indices
from qr_reader.detector.finder_pattern import (
    Triplet,
    extract_finder_patterns,
    find_all_associations,
    find_triplets,
)
from qr_reader.detector.homography import (
    compute_qr_corners,
    ransac_homography,
    refine_homography_lm,
)
from qr_reader.detector.landmarks import (
    build_named_landmarks,
    canonical_grid_landmarks,
)
from qr_reader.detector.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.detector.sample import sample_qr_bits
from qr_reader.detector.version import (
    build_constraints,
    estimate_version,
    filter_constraints,
)
from qr_reader.qr_gen import binarize_image


def _run_detection(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Shared pipeline: image → homography + version.

    Returns ``(H, V)`` where *H* maps QR-grid (x, y) → image (x, y) and
    *V* is the estimated QR version (1..40).

    Raises ``ValueError`` if no QR code is detected.
    """
    img_gray = np.asarray(image)
    if img_gray.ndim == 3:
        import cv2

        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    # 1. Binarize
    img_binary = binarize_image(img_gray)

    # 2. Find alignment-pattern candidates
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        raise ValueError("No alignment patterns found")

    # 3. Cluster candidates
    clusters = cluster_candidates(rows_valid, cols_valid_all)

    # 4. Per-cluster corner finding
    angular_distance_nms = 10 * 2 * np.pi / 360  # 10 degrees
    all_corners: list[tuple[int, np.ndarray]] = []

    for ci, cluster in enumerate(clusters):
        seed_row = int(cluster.row)
        seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2)

        region_mask = region_fill_wave_front(np.asarray(img_binary), seed_row, seed_col)
        boundary = region_boundary_8(region_mask)
        components = boundary_connected_components_ndimage(np.asarray(boundary))

        for comp in components:
            comp_arr = np.asarray(comp, dtype=np.float64)
            if comp_arr.shape[0] < 4:
                continue
            centroid_i = comp_arr.mean(axis=0)
            rd = np.linalg.norm(comp_arr - centroid_i, axis=1)
            ang = np.arctan2(
                comp_arr[:, 1] - centroid_i[1], comp_arr[:, 0] - centroid_i[0]
            )
            try:
                idx = angular_nms_top_radial_indices(
                    rd, ang, angular_nms_rad=angular_distance_nms, k=4
                )
            except ValueError:
                continue
            all_corners.append((ci, comp_arr[idx]))

    if not all_corners:
        raise ValueError("No corners detected")

    # 5. Extract finder patterns
    fps = extract_finder_patterns(all_corners)
    associations = find_all_associations(fps)
    triplets = find_triplets(fps, associations)

    if not triplets:
        raise ValueError("No finder-pattern triplet found")

    raw_triplet = triplets[0]
    # The lower-level triplet finder reasons in xy-style coordinates, while the
    # landmark code represents image points as (row, col). Convert its labels
    # back to row/col semantics before building landmarks.
    triplet = Triplet(
        top_left_idx=raw_triplet.top_left_idx,
        top_right_idx=raw_triplet.bottom_left_idx,
        bottom_left_idx=raw_triplet.top_right_idx,
    )
    landmarks = build_named_landmarks(triplet, fps)

    # 6. Version estimation
    constraints = build_constraints(landmarks)
    usable = filter_constraints(constraints, k=4, min_span=1.0)
    V_best, _scores = estimate_version(usable)
    N_best = 4 * V_best + 17

    # 7. Homography estimation
    grid_lm = canonical_grid_landmarks(N_best)
    image_lm = build_named_landmarks(triplet, fps)

    def rc_to_xy(pts: np.ndarray) -> np.ndarray:
        return pts[:, ::-1]

    src_xy = []
    dst_xy = []
    for attr in ("A", "B", "C", "D", "E", "F"):
        g = getattr(grid_lm, attr)
        i = getattr(image_lm, attr)
        if g is not None and i is not None:
            src_xy.append(rc_to_xy(g))
            dst_xy.append(rc_to_xy(i))
    src_xy = np.vstack(src_xy)
    dst_xy = np.vstack(dst_xy)

    H_ransac, _inliers = ransac_homography(src_xy, dst_xy, threshold=3.0, iters=2000)
    H_refined = refine_homography_lm(H_ransac, src_xy, dst_xy, loss="linear")

    return H_refined, V_best


def detect_corners(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect the four image corners of a single QR code.

    Args:
        image: Grayscale or colour image containing one QR code.

    Returns:
        ``(corners, version)`` where *corners* is a ``(4, 2)`` float64 array
        in ``[TL, TR, BR, BL]`` order (x, y image coordinates) and *version*
        is the estimated QR version (1..40).

    Raises:
        ValueError: if detection fails.
    """
    H, version = _run_detection(image)
    N = 4 * version + 17
    corners = compute_qr_corners(H, N)
    return corners, version


def detect_homography(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Estimate the homography mapping QR-grid coordinates to image coordinates.

    Args:
        image: Grayscale or colour image containing one QR code.

    Returns:
        ``(H, version)`` where *H* is a ``(3, 3)`` float64 homography that
        maps ``(x_grid, y_grid)`` → ``(x_image, y_image)``, and *version* is
        the estimated QR version.

    Raises:
        ValueError: if detection fails.
    """
    return _run_detection(image)


def detect_sample(image: np.ndarray) -> np.ndarray:
    """Sample a QR code image into its module bit matrix.

    Args:
        image: Grayscale or colour image containing one QR code.

    Returns:
        ``(N, N)`` boolean array where ``True`` = dark module and ``N`` depends
        on the detected version.  The matrix is transposed so that indexing
        ``bits[col, row]`` matches the QR standard module layout.

    Raises:
        ValueError: if detection fails.
    """
    H, version = _run_detection(image)
    N = 4 * version + 17

    img_gray = np.asarray(image)
    if img_gray.ndim == 3:
        import cv2

        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    return sample_qr_bits(img_gray, H, N)
