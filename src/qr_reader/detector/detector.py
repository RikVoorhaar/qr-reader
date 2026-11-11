"""High-level QR detector API.

Provides three convenience functions that work on a single-QR image:

* ``detect_corners``  →  image corners + version
* ``detect_homography`` → homography matrix + version
* ``detect_sample``   →  sampled bit matrix

All functions assume the input contains **one** QR code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edge_fitting import MAX_GAP, DISTANCE_THRESHOLD
from qr_reader.detector.ray_fit import RayFitResult, fit_finder_ray
from qr_reader.detector.finder_pattern import (
    FinderPattern,
    Triplet,
    find_valid_triplets,
)
from qr_reader.detector.homography import (
    compute_qr_corners,
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)
from qr_reader.detector.landmarks import NamedLandmarks, build_named_landmarks
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.detector.sample import sample_qr_bits
from qr_reader.detector.version import (
    build_constraints,
    estimate_version,
    filter_constraints,
)
from qr_reader.qr_gen import binarize_image


def _score_timing_pattern(img_gray: np.ndarray, H: np.ndarray, N: int) -> float:
    """Score row 6 and column 6 for timing-pattern alternation.

    The timing pattern runs from module 8 to N−9 on row 6 and column 6,
    alternating dark/light each module.  Returns a score in [0, 1] where
    1 = perfect alternation.
    """
    if N <= 17:
        return 0.0

    mid_start, mid_end = 8, N - 9
    if mid_end <= mid_start:
        return 0.0

    nc = mid_end - mid_start + 1

    xs = np.arange(mid_start, mid_end + 1, dtype=np.float64)
    ys6 = np.full(nc, 6.0, dtype=np.float64)

    grid_row6 = np.column_stack([xs, ys6])
    img_row6 = project_points(H, grid_row6)
    coords_r = np.stack([img_row6[:, 1], img_row6[:, 0]])
    row6_vals = map_coordinates(
        img_gray.astype(np.float64), coords_r, order=1, mode="nearest"
    )

    grid_col6 = np.column_stack([ys6, xs])
    img_col6 = project_points(H, grid_col6)
    coords_c = np.stack([img_col6[:, 1], img_col6[:, 0]])
    col6_vals = map_coordinates(
        img_gray.astype(np.float64), coords_c, order=1, mode="nearest"
    )

    all_vals = np.concatenate([row6_vals, col6_vals])
    thr = float(np.mean(all_vals))

    row6_bits = row6_vals >= thr
    col6_bits = col6_vals >= thr

    row6_score = int(np.sum(row6_bits[1:] != row6_bits[:-1]))
    col6_score = int(np.sum(col6_bits[1:] != col6_bits[:-1]))

    max_possible = 2 * (nc - 1)
    if max_possible < 1:
        return 0.0

    return (row6_score + col6_score) / max_possible


@dataclass
class DetectionTimings:
    """Wall times for each stage of a detection run, in seconds."""

    binarize_s: float = 0.0
    alignment_s: float = 0.0
    clustering_s: float = 0.0
    refinement_s: float = 0.0
    dedup_s: float = 0.0
    triplet_s: float = 0.0
    homography_s: float = 0.0
    total_s: float = 0.0


@dataclass
class DetectionResult:
    """Outcome and metadata of a detection run.

    ``_run_detection`` always returns a ``DetectionResult`` — even on failure,
    so callers can inspect how far the pipeline got (stage counts) and why it
    stopped (``error``).  On success, ``H`` maps QR-grid (x, y) → image (x, y)
    and ``version`` is the estimated QR version (1..40).
    """

    success: bool
    H: np.ndarray | None = None
    version: int | None = None
    error: Exception | None = None
    n_candidates: int = 0
    n_finder_patterns: int = 0
    n_triplets: int = 0
    timings: DetectionTimings = field(default_factory=DetectionTimings)


@dataclass
class DetectionContext:
    """Preprocessing output shared by all detection candidates.

    ``_prepare_detection`` runs binarization, alignment scanning, finder
    fitting, deduplication, and triplet discovery exactly once.  The resulting
    context can then be ranked and fitted repeatedly without repeating the
    expensive preprocessing.
    """

    img_gray: np.ndarray
    fps: list[FinderPattern]
    score_map: dict[int, float]
    global_corners_xy: dict[int, np.ndarray]
    triplets: list[Triplet]
    timings: DetectionTimings
    n_candidates: int = 0
    n_finder_patterns: int = 0


@dataclass
class TripletCandidate:
    """A triplet ranked by its best cross-ratio version score.

    ``version_candidates`` holds all finite-score versions in ascending score
    order; ``best_version`` is the first entry.  ``cross_ratio_score`` is the
    median ``|log(r_measured / r_expected)|`` of the winning version (lower is
    better).
    """

    triplet: Triplet
    landmarks: NamedLandmarks
    version_candidates: list[int]
    version_scores: np.ndarray
    best_version: int
    cross_ratio_score: float


def _prepare_detection(image: np.ndarray, result: DetectionResult) -> DetectionContext:
    """Run the finder/triplet stage once and return a reusable context.

    ``result`` receives the stage counts and per-stage timings as they become
    available, so callers still see partial progress when this raises.
    """
    timings = result.timings

    img_gray = np.asarray(image)
    if img_gray.ndim == 3:
        import cv2

        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    t0 = perf_counter()
    img_binary = binarize_image(img_gray)
    timings.binarize_s = perf_counter() - t0

    max_error = np.log(1.3)
    t0 = perf_counter()
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    timings.alignment_s = perf_counter() - t0
    if len(rows_valid) == 0:
        raise ValueError("No alignment patterns found")

    t0 = perf_counter()
    clusters = cluster_candidates(rows_valid, cols_valid_all)
    timings.clustering_s = perf_counter() - t0
    result.n_candidates = len(clusters)

    # Per-cluster finder fitting
    t0 = perf_counter()
    fps: list[FinderPattern] = []
    score_map: dict[int, float] = {}
    global_corners_xy: dict[int, np.ndarray] = {}
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(img_gray, bbox)
        if roi.size == 0:
            continue

        r0 = max(0, int(bbox[0]))
        c0 = max(0, int(bbox[2]))

        c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
        c_row = float(cluster.row) - r0
        center_xy = np.array([c_col, c_row], dtype=np.float64)
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

        fr = fit_finder_ray(
            roi, center_xy, m_est,
            max_gap=MAX_GAP,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        if not fr.valid:
            continue

        corners_xy_global = fr.corners + np.array([c0, r0], dtype=np.float64)
        # Sanity: remove degenerate (flat) findings
        wh = np.ptp(corners_xy_global, axis=0)
        if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est:
            continue

        corners_rc = corners_xy_global[:, ::-1]
        fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
        score_map[ci] = fr.score
        global_corners_xy[ci] = corners_xy_global
    timings.refinement_s = perf_counter() - t0

    if not fps:
        raise ValueError("No finder patterns fitted")

    # Deduplicate overlapping finder candidates (nearby clusters from the
    # same physical finder).  Within the smaller segment length, keep only the
    # higher-scoring candidate.
    t0 = perf_counter()
    keep_mask = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep_mask[i]:
            continue
        ci = fps[i].outer_corners.mean(axis=0)
        seg_i = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
        for j in range(i + 1, len(fps)):
            if not keep_mask[j]:
                continue
            cj = fps[j].outer_corners.mean(axis=0)
            seg_j = float(np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1]))
            if float(np.linalg.norm(ci - cj)) < 1.0 * min(seg_i, seg_j):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    fps = [fp for fp, keep in zip(fps, keep_mask) if keep]
    timings.dedup_s = perf_counter() - t0
    result.n_finder_patterns = len(fps)

    if not fps:
        raise ValueError("No finder patterns after deduplication")

    # Find triplets via centre geometry + axis alignment
    t0 = perf_counter()
    triplets = find_valid_triplets(fps, score_map)
    timings.triplet_s = perf_counter() - t0
    result.n_triplets = len(triplets)
    if not triplets:
        raise ValueError("No finder-pattern triplet found")

    return DetectionContext(
        img_gray=img_gray,
        fps=fps,
        score_map=score_map,
        global_corners_xy=global_corners_xy,
        triplets=triplets,
        timings=timings,
        n_candidates=result.n_candidates,
        n_finder_patterns=result.n_finder_patterns,
    )


def _rank_triplets(context: DetectionContext) -> list[TripletCandidate]:
    """Rank every triplet by its best cross-ratio version score.

    For each triplet the landmarks are built from the fitted finder corners,
    cross-ratio constraints are measured (outer rings only when inner corners
    are unavailable), filtered, and scored against every version.  Empty
    constraint sets and all-non-finite score vectors raise ``ValueError``:
    they indicate a bug in landmark construction, not a recoverable runtime
    condition.
    """
    ranked: list[TripletCandidate] = []
    for triplet in context.triplets:
        landmarks = build_named_landmarks(triplet, context.fps)
        constraints = build_constraints(landmarks)
        if not constraints:
            raise ValueError(
                "Cross-ratio measurement produced no constraints for triplet "
                f"(TL={triplet.top_left_idx}, TR={triplet.top_right_idx}, "
                f"BL={triplet.bottom_left_idx})"
            )

        filtered = filter_constraints(constraints)
        if not filtered:
            raise ValueError(
                "Cross-ratio constraints were all discarded by span/colinearity "
                f"filtering for triplet (TL={triplet.top_left_idx}, "
                f"TR={triplet.top_right_idx}, BL={triplet.bottom_left_idx})"
            )

        _best_version, scores = estimate_version(filtered)
        finite = np.isfinite(scores)
        if not np.any(finite):
            raise ValueError(
                "All cross-ratio version scores are non-finite for triplet "
                f"(TL={triplet.top_left_idx}, TR={triplet.top_right_idx}, "
                f"BL={triplet.bottom_left_idx})"
            )

        order = np.argsort(np.where(finite, scores, np.inf), kind="stable")
        version_candidates = [int(i) + 1 for i in order if finite[i]]
        best_version = version_candidates[0]
        best_score = float(scores[best_version - 1])
        ranked.append(
            TripletCandidate(
                triplet=triplet,
                landmarks=landmarks,
                version_candidates=version_candidates,
                version_scores=scores,
                best_version=best_version,
                cross_ratio_score=best_score,
            )
        )

    ranked.sort(key=lambda candidate: candidate.cross_ratio_score)
    return ranked


def _triplet_axes(
    context: DetectionContext, triplet: Triplet
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (right, down) unit vectors in image (x, y) for a triplet."""
    tl = context.global_corners_xy[triplet.top_left_idx].mean(axis=0)
    tr = context.global_corners_xy[triplet.top_right_idx].mean(axis=0)
    bl = context.global_corners_xy[triplet.bottom_left_idx].mean(axis=0)

    right = tr - tl
    right = right / (float(np.linalg.norm(right)) + 1e-12)
    down = bl - tl
    down = down / (float(np.linalg.norm(down)) + 1e-12)
    return right, down


def _canonicalize_corners(
    corners_xy: np.ndarray, right: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Order four finder corners as [TL, TR, BR, BL] using a triplet basis."""
    centre_xy = corners_xy.mean(axis=0)
    uv = corners_xy - centre_xy
    u_proj = uv @ right
    v_proj = uv @ down
    idx_tl = int(np.argmin(u_proj + v_proj))
    idx_tr = int(np.argmax(u_proj - v_proj))
    idx_br = int(np.argmax(u_proj + v_proj))
    idx_bl = int(np.argmin(u_proj - v_proj))
    return corners_xy[np.array([idx_tl, idx_tr, idx_br, idx_bl])]


def _fit_candidate_homography(
    context: DetectionContext, triplet: Triplet, N: int
) -> tuple[np.ndarray, float]:
    """Fit the 12-point DLT + LM homography for one (triplet, version).

    Returns ``(H, reprojection_error_px)``.  ``H`` maps QR-grid (x, y) to image
    (x, y).
    """
    right, down = _triplet_axes(context, triplet)
    tl_c = _canonicalize_corners(
        context.global_corners_xy[triplet.top_left_idx], right, down
    )
    tr_c = _canonicalize_corners(
        context.global_corners_xy[triplet.top_right_idx], right, down
    )
    bl_c = _canonicalize_corners(
        context.global_corners_xy[triplet.bottom_left_idx], right, down
    )

    grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64)
    src_xy: list[list[float]] = []
    dst_xy: list[list[float]] = []
    for corners, origin in [
        (tl_c, (0, 0)),
        (tr_c, (N - 7, 0)),
        (bl_c, (0, N - 7)),
    ]:
        for i in range(4):
            src_xy.append(
                [origin[0] + grid_offsets[i, 0], origin[1] + grid_offsets[i, 1]]
            )
            dst_xy.append(corners[i].tolist())

    src_arr = np.array(src_xy, dtype=np.float64)
    dst_arr = np.array(dst_xy, dtype=np.float64)

    H = estimate_homography_dlt(src_arr, dst_arr)
    try:
        H = refine_homography_lm(H, src_arr, dst_arr, loss="linear")
    except Exception:
        pass

    proj = project_points(H, src_arr)
    err = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
    return H, err


def _run_detection(image: np.ndarray) -> DetectionResult:
    """Shared pipeline: image → homography + version.

    Preprocessing and triplet discovery run once; triplets are ranked by their
    best cross-ratio version score and the top-ranked (triplet, version) is
    fitted.  Single-candidate wrappers (``detect_corners``,
    ``detect_homography``, ``detect_sample``) use this result.

    Returns a :class:`DetectionResult` carrying the estimated homography,
    version, per-stage candidate counts, and per-stage wall times.  On failure
    the result has ``success=False``, ``error`` set to the offending exception,
    and the partial counts/timings recorded as far as the pipeline got.
    """
    result = DetectionResult(success=False)
    t_total = perf_counter()
    try:
        context = _prepare_detection(image, result)
        t0 = perf_counter()
        ranked = _rank_triplets(context)
        candidate = ranked[0]
        N = 4 * candidate.best_version + 17
        H, _err = _fit_candidate_homography(context, candidate.triplet, N)
        result.timings.homography_s = perf_counter() - t0

        result.success = True
        result.H = H
        result.version = candidate.best_version
    except Exception as exc:
        result.success = False
        result.error = exc
    finally:
        result.timings.total_s = perf_counter() - t_total

    return result


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
    result = _run_detection(image)
    if not result.success:
        raise result.error or ValueError("Detection failed")
    assert result.H is not None and result.version is not None
    N = 4 * result.version + 17
    corners = compute_qr_corners(result.H, N)
    return corners, result.version


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
    result = _run_detection(image)
    if not result.success:
        raise result.error or ValueError("Detection failed")
    assert result.H is not None and result.version is not None
    return result.H, result.version


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
    result = _run_detection(image)
    if not result.success:
        raise result.error or ValueError("Detection failed")
    assert result.H is not None and result.version is not None
    N = 4 * result.version + 17

    img_gray = np.asarray(image)
    if img_gray.ndim == 3:
        import cv2

        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    return sample_qr_bits(img_gray, result.H, N)
