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
    find_valid_triplets,
)
from qr_reader.detector.homography import (
    compute_qr_corners,
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.detector.sample import sample_qr_bits
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


def _run_detection(image: np.ndarray) -> DetectionResult:
    """Shared pipeline: image → homography + version.

    Returns a :class:`DetectionResult` carrying the estimated homography,
    version, per-stage candidate counts, and per-stage wall times.  On failure
    the result has ``success=False``, ``error`` set to the offending exception,
    and the partial counts/timings recorded as far as the pipeline got.
    """
    result = DetectionResult(success=False)
    timings = result.timings
    t_total = perf_counter()
    try:
        img_gray = np.asarray(image)
        h_img, w_img = img_gray.shape[:2]
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

        # 4. Per-cluster finder fitting
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
        # same physical finder).  Within 6× the smaller segment length, keep
        # only the higher-scoring candidate.
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

        # 5. Find triplets via centre geometry + axis alignment
        t0 = perf_counter()
        raw_triplets = find_valid_triplets(fps, score_map)
        timings.triplet_s = perf_counter() - t0
        result.n_triplets = len(raw_triplets)
        if not raw_triplets:
            raise ValueError("No finder-pattern triplet found")

        raw = raw_triplets[0]
        tl_idx = raw.top_left_idx
        tr_idx = raw.top_right_idx
        bl_idx = raw.bottom_left_idx

        fp_map = {fp.cluster_idx: fp for fp in fps}
        rows = {idx: float(fp_map[idx].outer_corners.mean(axis=0)[0]) for idx in [tl_idx, tr_idx, bl_idx]}
        cols = {idx: float(fp_map[idx].outer_corners.mean(axis=0)[1]) for idx in [tl_idx, tr_idx, bl_idx]}

        # 6. Version estimation via homography + timing-pattern scoring.
        #    Search all legal N values (N = 17 + 4*V, V ∈ [1, 40]).
        center_tl_xy = np.array([cols[tl_idx], rows[tl_idx]], dtype=np.float64)
        c_tr = np.array([cols[tr_idx], rows[tr_idx]], dtype=np.float64)
        c_bl = np.array([cols[bl_idx], rows[bl_idx]], dtype=np.float64)

        global_u = c_tr - center_tl_xy
        global_u = global_u / (float(np.linalg.norm(global_u)) + 1e-12)
        global_v = c_bl - center_tl_xy
        global_v = global_v / (float(np.linalg.norm(global_v)) + 1e-12)

        def _canonicalize_corners(corners_xy: np.ndarray) -> np.ndarray:
            centre_xy = corners_xy.mean(axis=0)
            uv = corners_xy - centre_xy
            u_proj = uv @ global_u
            v_proj = uv @ global_v
            idx_tl = int(np.argmin(u_proj + v_proj))
            idx_tr = int(np.argmax(u_proj - v_proj))
            idx_br = int(np.argmax(u_proj + v_proj))
            idx_bl = int(np.argmin(u_proj - v_proj))
            return corners_xy[np.array([idx_tl, idx_tr, idx_br, idx_bl])]

        # 7. Global homography from 12 refined finder corners: DLT + LM.
        grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64)
        tl_c = _canonicalize_corners(global_corners_xy[tl_idx])
        tr_c = _canonicalize_corners(global_corners_xy[tr_idx])
        bl_c = _canonicalize_corners(global_corners_xy[bl_idx])

        t0 = perf_counter()
        best_err = np.inf
        best_H: np.ndarray | None = None
        best_N = 0

        for N_cand in range(21, 178, 4):
            src_xy: list[list[float]] = []
            dst_xy: list[list[float]] = []
            for corners, origin in [
                (tl_c, (0, 0)),
                (tr_c, (N_cand - 7, 0)),
                (bl_c, (0, N_cand - 7)),
            ]:
                for i in range(4):
                    src_xy.append([origin[0] + grid_offsets[i, 0], origin[1] + grid_offsets[i, 1]])
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
            timing = _score_timing_pattern(img_gray, H, N_cand)
            combined_err = err - timing  # combine reprojection error + timing score
            if combined_err < best_err:
                best_err = combined_err
                best_H = H
                best_N = N_cand
        timings.homography_s = perf_counter() - t0

        if best_H is None:
            raise ValueError("Homography estimation failed")

        V_best = (best_N - 17) // 4
        if V_best < 1 or V_best > 40:
            raise ValueError(f"Implausible version {V_best}")

        result.success = True
        result.H = best_H
        result.version = V_best
    except Exception as exc:
        result.success = False
        result.error = exc
    finally:
        timings.total_s = perf_counter() - t_total

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
