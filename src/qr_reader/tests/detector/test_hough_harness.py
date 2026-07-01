"""Hough + Refine Test Harness.

Phases I, II, and IV from plan-006.

Phase I:   _compute_finder_edges — ground-truth edge geometry helper.
Phase II:  TestFixtureReal — pipeline reproduction tests against synth data.
Phase IV:  _describe_support — diagnostic instrumentation for failing tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.hough import LineSegment, hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

# ===========================================================================
# Phase I — Ground-truth edge geometry
# ===========================================================================

# --- Clipping helpers -------------------------------------------------------

_INSIDE = 0  # 0000
_LEFT = 1  # 0001
_RIGHT = 2  # 0010
_BOTTOM = 4  # 0100
_TOP = 8  # 1000


def _compute_outcode(
    x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float
) -> int:
    code = _INSIDE
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _TOP
    elif y > ymax:
        code |= _BOTTOM
    return code


def _clip_segment(
    p0: np.ndarray, p1: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> np.ndarray | None:
    """Cohen-Sutherland line clipping. Returns clipped (2, 2) or None."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
    outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)

    while True:
        if (outcode0 | outcode1) == 0:
            # Both inside
            return np.array([[x0, y0], [x1, y1]], dtype=np.float64)
        if (outcode0 & outcode1) != 0:
            # Both outside same region
            return None

        # Pick the point that's outside
        oc = outcode0 if outcode0 != 0 else outcode1
        x, y = 0.0, 0.0

        if oc & _TOP:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymin
        elif oc & _BOTTOM:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymax
        elif oc & _RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmax
        elif oc & _LEFT:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmin

        if oc == outcode0:
            x0, y0 = x, y
            outcode0 = _compute_outcode(x0, y0, xmin, xmax, ymin, ymax)
        else:
            x1, y1 = x, y
            outcode1 = _compute_outcode(x1, y1, xmin, xmax, ymin, ymax)


# --- Main helper ------------------------------------------------------------


def _compute_finder_edges(
    metadata: dict,
    roi_offset: tuple[int, int] | None = None,
    roi_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Compute 36 GT finder-pattern edges via module-grid homography.

    12 per finder (TL, TR, BL): 4 sides × 3 module boundaries (k=0,1,2 and k=5,6,7).
    Inner segments clipped: k_vis = min(k, 7-k) — visible feature span only.

    Each returned dict has keys:
        label        — "TL_top0", "TL_left0", …
        normal       — unit normal (2,)
        rho          — ≥ 0
        segment      — ``(2, 2)`` endpoints or None if not intersecting the ROI
    """
    corners = metadata["corners_qr"]
    N = metadata["N"]

    src_xy = np.array(
        [[0.0, 0.0], [float(N), 0.0], [float(N), float(N)], [0.0, float(N)]],
        dtype=np.float64,
    )
    dst_xy = np.array(
        [
            [float(corners["TL"][0]), float(corners["TL"][1])],
            [float(corners["TR"][0]), float(corners["TR"][1])],
            [float(corners["BR"][0]), float(corners["BR"][1])],
            [float(corners["BL"][0]), float(corners["BL"][1])],
        ],
        dtype=np.float64,
    )
    H = estimate_homography_dlt(src_xy, dst_xy)

    def _grid_to_image(row: float, col: float) -> np.ndarray:
        pt = np.array([[col, row]], dtype=np.float64)
        return project_points(H, pt)[0]

    finder_positions: dict[str, tuple[int, int]] = {
        "TL": (0, 0),
        "TR": (0, N - 7),
        "BL": (N - 7, 0),
    }

    TOP = [0, 1, 2]
    BOTTOM = [5, 6, 7]
    LEFT = [0, 1, 2]
    RIGHT = [5, 6, 7]

    results: list[dict] = []

    for finder_name, (r0, c0) in finder_positions.items():

        for side, offsets in [("top", TOP), ("bot", BOTTOM)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k), float(c0 + k_vis))
                b = _grid_to_image(float(r0 + k), float(c0 + 7 - k_vis))
                _add_gt_edge(results, finder_name, side, k, a, b, roi_offset, roi_shape)

        for side, offsets in [("left", LEFT), ("right", RIGHT)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k_vis), float(c0 + k))
                b = _grid_to_image(float(r0 + 7 - k_vis), float(c0 + k))
                _add_gt_edge(results, finder_name, side, k, a, b, roi_offset, roi_shape)

    return results


def _add_gt_edge(
    results: list[dict],
    finder_name: str,
    side: str,
    k: int,
    a: np.ndarray,
    b: np.ndarray,
    roi_offset: tuple[int, int] | None,
    roi_shape: tuple[int, int] | None,
) -> None:
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        normal = np.array([1.0, 0.0], dtype=np.float64)
        rho = 0.0
    else:
        direction = d / length
        normal = np.array([direction[1], -direction[0]], dtype=np.float64)
        rho = float(normal @ a)
        if rho < 0:
            normal = -normal
            rho = -rho

    label = f"{finder_name}_{side}{k}"

    if roi_offset is not None and roi_shape is not None:
        row0, col0 = int(roi_offset[0]), int(roi_offset[1])
        H_img, W_img = int(roi_shape[0]), int(roi_shape[1])
        offset_xy = np.array([col0, row0], dtype=np.float64)
        a_local = a - offset_xy
        b_local = b - offset_xy
        rho_local = float(rho - normal @ offset_xy)
        if rho_local < 0:
            rho_local = -rho_local
            normal_local = -normal
        else:
            normal_local = normal.copy()
        clipped = _clip_segment(
            a_local, b_local, 0.0, float(W_img - 1), 0.0, float(H_img - 1)
        )
        segment = None if clipped is None else clipped
    else:
        normal_local = normal.copy()
        rho_local = rho
        segment = np.array([a.copy(), b.copy()], dtype=np.float64)

    results.append(
        {
            "label": label,
            "normal": normal_local,
            "rho": rho_local,
            "segment": segment,
        }
    )


# ===========================================================================
# Phase I — Tests
# ===========================================================================


class TestFinderEdges:
    """Unit tests for _compute_finder_edges."""

    def test_clean_axis_aligned_qr(self):
        """A clean axis-aligned QR (version=1, no transform, border=4, ppm=10).

        TL top edge normal is (0, 1), rho = 40.
        Segment span in the ROI is within 1 px of expected.
        """
        corners = {
            "TL": [40.0, 40.0],
            "TR": [250.0, 40.0],
            "BR": [250.0, 250.0],
            "BL": [40.0, 250.0],
        }
        metadata = {"corners_qr": corners, "N": 21}
        roi_offset = (0, 0)
        roi_shape = (300, 300)

        edges = _compute_finder_edges(metadata, roi_offset, roi_shape)

        # Find TL_top0
        tl_top = next(e for e in edges if e["label"] == "TL_top0")

        # Normal: (0, 1)
        np.testing.assert_allclose(tl_top["normal"], [0.0, 1.0], atol=0.01)
        # rho: y = 40
        assert abs(tl_top["rho"] - 40.0) < 1.0

        # Segment: from (40, 40) to (110, 40) in (x, y)
        seg = tl_top["segment"]
        assert seg is not None
        np.testing.assert_allclose(seg[0], [40.0, 40.0], atol=1.0)
        np.testing.assert_allclose(seg[1], [110.0, 40.0], atol=1.0)

    def test_all_edges_present(self):
        """All 36 expected edges are returned (12 per finder: 4 sides × 3 boundaries)."""
        corners = {
            "TL": [10.0, 10.0],
            "TR": [100.0, 20.0],
            "BR": [90.0, 110.0],
            "BL": [5.0, 100.0],
        }
        metadata = {"corners_qr": corners, "N": 35}

        edges = _compute_finder_edges(metadata)
        labels = {e["label"] for e in edges}
        assert len(labels) == 36

        suffixes = [f"{side}{k}" for side in ["top"] for k in [0, 1, 2]]
        suffixes += [f"{side}{k}" for side in ["bot"] for k in [5, 6, 7]]
        suffixes += [f"{side}{k}" for side in ["left"] for k in [0, 1, 2]]
        suffixes += [f"{side}{k}" for side in ["right"] for k in [5, 6, 7]]
        for finder in ["TL", "TR", "BL"]:
            for suffix in suffixes:
                assert f"{finder}_{suffix}" in labels

    def test_all_normals_unit_length(self):
        """Every returned normal has length ≈ 1."""
        corners = {
            "TL": [10.0, 10.0],
            "TR": [100.0, 20.0],
            "BR": [90.0, 110.0],
            "BL": [5.0, 100.0],
        }
        metadata = {"corners_qr": corners, "N": 35}

        for e in _compute_finder_edges(metadata):
            length = np.linalg.norm(e["normal"])
            assert abs(length - 1.0) < 0.001, f"{e['label']} normal length={length}"

    def test_all_rhos_non_negative(self):
        """Every returned rho is ≥ 0."""
        corners = {
            "TL": [10.0, 10.0],
            "TR": [100.0, 20.0],
            "BR": [90.0, 110.0],
            "BL": [5.0, 100.0],
        }
        metadata = {"corners_qr": corners, "N": 35}

        for e in _compute_finder_edges(metadata):
            assert e["rho"] >= -1e-12, f"{e['label']} rho={e['rho']}"

    def test_segment_clipped_when_outside_roi(self):
        """Segment outside the ROI returns None."""
        corners = {
            "TL": [100.0, 100.0],
            "TR": [200.0, 100.0],
            "BR": [200.0, 200.0],
            "BL": [100.0, 200.0],
        }
        metadata = {"corners_qr": corners, "N": 21}
        # ROI that misses the QR entirely
        roi_offset = (0, 0)
        roi_shape = (50, 50)

        edges = _compute_finder_edges(metadata, roi_offset, roi_shape)
        for e in edges:
            assert e["segment"] is None, f"{e['label']} should not intersect ROI"

    def test_roi_local_coordinates(self):
        """ROI-local coordinates are correctly translated."""
        corners = {
            "TL": [100.0, 100.0],
            "TR": [200.0, 100.0],
            "BR": [200.0, 200.0],
            "BL": [100.0, 200.0],
        }
        metadata = {"corners_qr": corners, "N": 21}
        roi_offset = (80, 80)  # row0=80, col0=80
        roi_shape = (60, 60)

        edges = _compute_finder_edges(metadata, roi_offset, roi_shape)
        tl_top = next(e for e in edges if e["label"] == "TL_top0")
        # TL is at (100, 100) in image → (20, 20) in ROI-local (x, y)
        assert tl_top["segment"] is not None
        np.testing.assert_allclose(tl_top["segment"][0], [20.0, 20.0], atol=2.0)


# ===========================================================================
# Phase II — Fixture-based reproduction tests
# ===========================================================================


# --- Pipeline up to clusters + ROIs -----------------------------------------


def _run_pipeline_to_rois(
    image: np.ndarray, *, blur_sigma: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], int]]:
    """Run the detection pipeline up to ROI extraction.

    Returns list of ``(roi_gray, nms, angle, bbox, cluster_idx)`` for each cluster.
    """
    if image.ndim == 3:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image)

    # Binarize
    img_binary = binarize_image(gray)

    # Find alignment patterns
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return []

    # Cluster
    clusters = cluster_candidates(rows_valid, cols_valid_all)

    results = []
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)

        if roi.size == 0:
            continue

        nms, angle = extract_thin_edges(roi, blur_sigma=blur_sigma)
        results.append((roi, nms, angle, bbox, ci))

    return results


# --- Soft-assert helper (no pytest_check dependency) ------------------------


def _collect_failures() -> tuple[list[str], callable]:
    """Return (failures, check) for manual soft-assert collection.

    Usage::

        failures, check = _collect_failures()
        check(condition, "message")
        if failures:
            pytest.fail("\\n".join(failures))
    """
    failures: list[str] = []

    def check(condition: bool, msg: str) -> None:
        if not condition:
            failures.append(msg)

    return failures, check


# --- Angle / rho matching ---------------------------------------------------


def _normal_angle_deg(normal: np.ndarray) -> float:
    """Angle of the unit normal vector in degrees, in [0, 180)."""
    rad = np.arctan2(normal[1], normal[0])
    if rad < 0:
        rad += np.pi
    return np.rad2deg(rad)


def _angular_distance_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Angular distance between two undirected normals, in degrees [0, 90]."""
    dot = np.clip(np.abs(np.dot(n1, n2)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


def _match_peak(
    gt_edge: dict,
    normals: np.ndarray,
    rhos: np.ndarray,
    angle_tol_deg: float = 5.0,
    rho_tol: float = 5.0,
) -> int:
    """Return index of best-matching Hough peak, or -1 if none."""
    best_i = -1
    best_dist = float("inf")
    for i in range(len(normals)):
        ang_dist = _angular_distance_deg(gt_edge["normal"], normals[i])
        rho_dist = abs(gt_edge["rho"] - rhos[i])
        if ang_dist <= angle_tol_deg and rho_dist <= rho_tol:
            score = ang_dist + rho_dist
            if score < best_dist:
                best_dist = score
                best_i = i
    return best_i


# --- Background fixture -----------------------------------------------------


def _make_background(H: int = 640, W: int = 640) -> np.ndarray:
    """Synthetic gradient background (RGB uint8)."""
    xx = np.linspace(0, 1, W, dtype=np.float32).reshape(1, -1)
    yy = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1)
    bg = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
    return np.stack([bg] * 3, axis=-1)  # (H, W, 3) RGB


# --- Test configs -----------------------------------------------------------


def _config_v12_default() -> AugmentationConfig:
    """Version 12, default difficulty (moderate perspective, noise)."""
    return AugmentationConfig(
        version=12,
        content="https://www.rikvoorhaar.com",
        error_correction="M",
        ppm_range=(5.0, 12.0),
        target_ppm_range=(4.0, 10.0),
        jitter_fraction=0.15,
        feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.2, 1.0),
        noise_sigma_range=(1.0, 5.0),
        jpeg_quality_range=(65, 95),
        global_seed=42,
    )


def _config_v12_clean() -> AugmentationConfig:
    """Version 12, clean — no noise, no blur, no perspective."""
    return AugmentationConfig(
        version=12,
        content="https://www.rikvoorhaar.com",
        error_correction="M",
        ppm_range=(10.0, 10.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(10.0, 10.0),
        feather_sigma_range=(0.5, 0.5),
        blur_sigma_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        jpeg_quality_range=(100, 100),
        global_seed=42,
    )


def _config_v5_default() -> AugmentationConfig:
    """Version 5, default difficulty."""
    return AugmentationConfig(
        version=5,
        content="QR Reader v1",
        error_correction="M",
        ppm_range=(5.0, 12.0),
        target_ppm_range=(4.0, 10.0),
        jitter_fraction=0.15,
        feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.2, 1.0),
        noise_sigma_range=(1.0, 5.0),
        jpeg_quality_range=(65, 95),
        global_seed=123,
    )


# --- Tests ------------------------------------------------------------------


class TestFixtureReal:
    """Pipeline reproduction tests against synth data."""

    # Reusable fixtures
    @pytest.fixture(scope="class")
    def background(self) -> np.ndarray:
        return _make_background(640, 640)

    @pytest.fixture(scope="class")
    def v12_default_data(self, background: np.ndarray):
        """Generate version-12 default-difficulty sample + metadata."""
        rng = np.random.default_rng(42)
        config = _config_v12_default()
        image, metadata = generate_sample(rng, config, background)
        return image, metadata

    @pytest.fixture(scope="class")
    def v12_clean_data(self, background: np.ndarray):
        """Generate version-12 clean sample + metadata."""
        rng = np.random.default_rng(42)
        config = _config_v12_clean()
        image, metadata = generate_sample(rng, config, background)
        return image, metadata

    @pytest.fixture(scope="class")
    def v5_default_data(self, background: np.ndarray):
        """Generate version-5 default-difficulty sample + metadata."""
        rng = np.random.default_rng(123)
        config = _config_v5_default()
        image, metadata = generate_sample(rng, config, background)
        return image, metadata

    # --- Assertion helpers --------------------------------------------------

    def _assert_peaks_exist(
        self,
        gt_edges: list[dict],
        normals: np.ndarray,
        rhos: np.ndarray,
        failures: list[str],
        cluster_idx: int,
    ) -> None:
        """Assertion 1: at least one Hough peak matches each GT edge (Failure D)."""
        for gt in gt_edges:
            if gt["segment"] is None:
                continue
            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                failures.append(
                    f"[C{cluster_idx}] {gt['label']}: no Hough peak within "
                    f"5° and 5 px of gt (normal=({gt['normal'][0]:.3f},{gt['normal'][1]:.3f}), "
                    f"rho={gt['rho']:.1f}) — Failure D (edge missing)"
                )

    def _assert_span_adequate(
        self,
        gt_edges: list[dict],
        normals: np.ndarray,
        rhos: np.ndarray,
        nms: np.ndarray,
        angle: np.ndarray,
        failures: list[str],
        cluster_idx: int,
        gap_tolerance: float = 2.0,
        distance_thresh: float = 1.5,
    ) -> None:
        """Assertion 2: refined segment span ≥ 80% of GT span (Failure A)."""
        for gt in gt_edges:
            if gt["segment"] is None:
                continue
            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                continue

            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                1.0,
                nms,
                angle,
                gap_tolerance=gap_tolerance,
                distance_thresh=distance_thresh,
            )

            if np.all(seg.endpoints == 0):
                failures.append(
                    f"[C{cluster_idx}] {gt['label']}: refined segment is degenerate — Failure A (span too short)\n"
                    + _describe_support(seg, nms, angle, distance_thresh)
                )
                continue

            # Compute GT span in ROI (project onto the line direction)
            gt_seg = gt["segment"]
            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
            gt_proj = gt_seg @ direction
            gt_span = abs(gt_proj[1] - gt_proj[0])

            ep_proj = seg.endpoints @ direction
            seg_span = abs(ep_proj[1] - ep_proj[0])

            if seg_span < 0.8 * gt_span:
                failures.append(
                    f"[C{cluster_idx}] {gt['label']}: span={seg_span:.1f} px "
                    f"< 80% of gt_span={gt_span:.1f} px — Failure A (span too short)\n"
                    + _describe_support(seg, nms, angle, distance_thresh)
                )

    def _assert_span_not_excessive(
        self,
        gt_edges: list[dict],
        normals: np.ndarray,
        rhos: np.ndarray,
        nms: np.ndarray,
        angle: np.ndarray,
        failures: list[str],
        cluster_idx: int,
        endpoint_tol: float = 5.0,
        gap_tolerance: float = 2.0,
        distance_thresh: float = 1.5,
    ) -> None:
        """Assertion 3: refined endpoints within 5 px of GT endpoints (Failure C)."""
        for gt in gt_edges:
            if gt["segment"] is None:
                continue
            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                continue

            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                1.0,
                nms,
                angle,
                gap_tolerance=gap_tolerance,
                distance_thresh=distance_thresh,
            )

            if np.all(seg.endpoints == 0):
                continue

            gt_seg = gt["segment"]
            # Check each GT endpoint is within tol of the segment endpoints
            for gt_ep in gt_seg:
                dists = np.linalg.norm(seg.endpoints - gt_ep, axis=1)
                if dists.min() > endpoint_tol:
                    failures.append(
                        f"[C{cluster_idx}] {gt['label']}: refined endpoints "
                        f"({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→"
                        f"({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f}) "
                        f"too far from gt — Failure C (span too long)\n"
                        + _describe_support(seg, nms, angle, distance_thresh)
                    )
                    break

    def _assert_no_phantom(
        self,
        gt_edges: list[dict],
        normals: np.ndarray,
        rhos: np.ndarray,
        nms: np.ndarray,
        angle: np.ndarray,
        failures: list[str],
        cluster_idx: int,
        strength_threshold: float = 400.0,
        angular_match_deg: float = 12.0,
        gap_tolerance: float = 2.0,
        distance_thresh: float = 1.5,
    ) -> None:
        """Assertion 4: unmatched peaks with strong support in blank regions (Failure B).

        Only flags a "phantom" when the peak:
        1. Doesn't match any GT finder-boundary edge, AND
        2. Its normal is far from all GT-edge normals (i.e., it isn't a parallel
           internal QR module edge at a different rho), AND
        3. Its mean support strength exceeds *strength_threshold*, AND
        4. Its segment is spatially far from all GT-edge segments
           (data-region edges near the finder boundary are skipped).
        """
        # Pre-compute GT normals for angular check
        gt_normals = np.array(
            [e["normal"] for e in gt_edges if e["segment"] is not None]
        )

        for i in range(len(normals)):
            # Check if this peak matches any GT edge
            matched = any(
                gt["segment"] is not None
                and _match_peak(gt, normals[[i]], rhos[[i]]) >= 0
                for gt in gt_edges
            )
            if matched:
                continue

            # Skip if normal is close to any GT normal — likely internal
            # QR module edge parallel to a finder boundary, not a phantom.
            if len(gt_normals) > 0:
                min_ang = min(
                    _angular_distance_deg(normals[i], gn) for gn in gt_normals
                )
                if min_ang < angular_match_deg:
                    continue

            seg = refine_line(
                normals[i],
                float(rhos[i]),
                1.0,
                nms,
                angle,
                gap_tolerance=gap_tolerance,
                distance_thresh=distance_thresh,
            )

            if np.all(seg.endpoints == 0):
                continue

            # Find support pixels for this segment
            ys, xs = np.nonzero(np.asarray(nms))
            strengths = nms[ys, xs]
            points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
            dists = np.abs(points @ seg.normal - seg.rho)
            mask = dists < distance_thresh

            if np.sum(mask) == 0:
                continue

            mean_strength = float(strengths[mask].mean())

            if mean_strength > strength_threshold:
                failures.append(
                    f"[C{cluster_idx}] phantom peak {i}: mean NMS strength={mean_strength:.1f} "
                    f"on {np.sum(mask)} support pixels — "
                    f"normal=({normals[i][0]:.3f},{normals[i][1]:.3f}), rho={rhos[i]:.1f} — "
                    f"Failure B (phantom in blank region)\n"
                    + _describe_support(seg, nms, angle, distance_thresh)
                )

    def _assert_non_degenerate(
        self,
        gt_edges: list[dict],
        normals: np.ndarray,
        rhos: np.ndarray,
        nms: np.ndarray,
        angle: np.ndarray,
        failures: list[str],
        cluster_idx: int,
        gap_tolerance: float = 2.0,
        distance_thresh: float = 1.5,
    ) -> None:
        """Assertion 5: at least one peak per GT edge produces non-zero endpoints (Failure D)."""
        for gt in gt_edges:
            if gt["segment"] is None:
                continue
            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                continue

            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                1.0,
                nms,
                angle,
                gap_tolerance=gap_tolerance,
                distance_thresh=distance_thresh,
            )

            if np.all(seg.endpoints == 0):
                failures.append(
                    f"[C{cluster_idx}] {gt['label']}: peak exists but refine_line "
                    f"returns degenerate — Failure D (edge missing / degeneracy)\n"
                    + _describe_support(seg, nms, angle, distance_thresh)
                )

    # --- Test cases ----------------------------------------------------------

    def test_fixture_version12_default(self, v12_default_data):
        """Version 12, default difficulty — the failing case."""
        image, metadata = v12_default_data
        all_failures: list[str] = []

        roi_results = _run_pipeline_to_rois(image)

        if len(roi_results) == 0:
            pytest.skip("No clusters found — pipeline failed before Hough stage")

        for roi, nms, angle, bbox, ci in roi_results:
            normals, rhos, scores = hough_vote_peaks(nms, angle)

            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),  # (row0, col0)
                roi_shape=roi.shape,
            )

            # Run all assertions as soft-asserts
            fail_list: list[str] = []
            self._assert_peaks_exist(gt_edges, normals, rhos, fail_list, ci)
            self._assert_span_adequate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_span_not_excessive(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_no_phantom(gt_edges, normals, rhos, nms, angle, fail_list, ci)
            self._assert_non_degenerate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )

            all_failures.extend(fail_list)

        if all_failures:
            header = f"Version-12 default: {len(all_failures)} assertion failure(s):"
            pytest.fail(header + "\n  " + "\n  ".join(all_failures))
        else:
            # If everything passes, that's fine too — report it
            pass

    def test_fixture_version12_clean(self, v12_clean_data):
        """Version 12, clean — baseline, should pass easily."""
        image, metadata = v12_clean_data
        all_failures: list[str] = []

        roi_results = _run_pipeline_to_rois(image)

        if len(roi_results) == 0:
            pytest.skip("No clusters found")

        for roi, nms, angle, bbox, ci in roi_results:
            normals, rhos, scores = hough_vote_peaks(nms, angle)

            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),
                roi_shape=roi.shape,
            )

            fail_list: list[str] = []
            self._assert_peaks_exist(gt_edges, normals, rhos, fail_list, ci)
            self._assert_span_adequate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_span_not_excessive(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_no_phantom(gt_edges, normals, rhos, nms, angle, fail_list, ci)
            self._assert_non_degenerate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )

            all_failures.extend(fail_list)

        if all_failures:
            pytest.fail(
                f"Version-12 clean (expected to pass): {len(all_failures)} assertion failure(s):\n  "
                + "\n  ".join(all_failures)
            )

    def test_fixture_version5_default(self, v5_default_data):
        """Version 5, default difficulty — shows version-dependence."""
        image, metadata = v5_default_data
        all_failures: list[str] = []

        roi_results = _run_pipeline_to_rois(image)

        if len(roi_results) == 0:
            pytest.skip("No clusters found")

        for roi, nms, angle, bbox, ci in roi_results:
            normals, rhos, scores = hough_vote_peaks(nms, angle)

            gt_edges = _compute_finder_edges(
                metadata,
                roi_offset=(bbox[0], bbox[2]),
                roi_shape=roi.shape,
            )

            fail_list: list[str] = []
            self._assert_peaks_exist(gt_edges, normals, rhos, fail_list, ci)
            self._assert_span_adequate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_span_not_excessive(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )
            self._assert_no_phantom(gt_edges, normals, rhos, nms, angle, fail_list, ci)
            self._assert_non_degenerate(
                gt_edges, normals, rhos, nms, angle, fail_list, ci
            )

            all_failures.extend(fail_list)

        if all_failures:
            pytest.fail(
                f"Version-5 default: {len(all_failures)} assertion failure(s):\n  "
                + "\n  ".join(all_failures)
            )


# ===========================================================================
# Phase IV — Instrumentation and diagnostics
# ===========================================================================


def _describe_support(
    seg: LineSegment,
    nms: np.ndarray,
    angle: np.ndarray,
    distance_thresh: float = 1.5,
) -> str:
    """Diagnostic dump describing the support set for a LineSegment.

    Returns a human-readable string with:
      - Total support pixel count
      - Projection range (min, max)
      - List of gaps ≥ 1.5 px in sorted projection
      - 5 strongest / 5 weakest support pixels with (x, y) positions
      - Whether support falls in high‑density vs. isolated NMS regions.

    Parameters
    ----------
    seg : LineSegment
        The refined line segment to diagnose.
    nms : ndarray, shape (H, W)
        NMS edge magnitudes.
    angle : ndarray, shape (H, W)
        Edge-normal angles (unused but available).
    distance_thresh : float
        Distance threshold for support inclusion (default 1.5 px).

    Returns
    -------
    str
    """
    H, W = nms.shape

    # Collect support pixels
    ys, xs = np.nonzero(np.asarray(nms))
    strengths = nms[ys, xs]
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    dists = np.abs(points @ seg.normal - seg.rho)
    mask = dists < distance_thresh

    support_pts = points[mask]
    support_strengths = strengths[mask]

    n_support = len(support_pts)

    if n_support == 0:
        return (
            f"  _describe_support: no support pixels for line "
            f"n=({seg.normal[0]:.3f},{seg.normal[1]:.3f}) rho={seg.rho:.1f}"
        )

    # Direction along the line for projection
    direction = np.array([-seg.normal[1], seg.normal[0]], dtype=np.float64)
    proj = support_pts @ direction

    proj_min = float(proj.min())
    proj_max = float(proj.max())
    proj_span = proj_max - proj_min

    # Sorted projection for gap analysis
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]
    strengths_sorted = support_strengths[sort_idx]
    pts_sorted = support_pts[sort_idx]

    # Gap analysis
    gaps = []
    for i in range(1, len(proj_sorted)):
        gap = float(proj_sorted[i] - proj_sorted[i - 1])
        if gap >= 1.5:
            gaps.append((float(proj_sorted[i - 1]), float(proj_sorted[i]), gap))

    # Top / bottom N by strength (N=5)
    top_n = min(5, n_support)
    strength_order = np.argsort(-strengths_sorted)  # descending
    strongest = strength_order[:top_n]
    weakest = strength_order[-top_n:]

    # Whether support pixels fall in dense vs sparse NMS regions
    # Check density: fraction of support points that have ≥ 3 edge pixels
    # within a 3×3 neighbourhood in nms
    dense_count = 0
    for sx, sy in support_pts:
        x0 = max(0, int(sx) - 1)
        x1 = min(W, int(sx) + 2)
        y0 = max(0, int(sy) - 1)
        y1 = min(H, int(sy) + 2)
        neighbourhood = nms[y0:y1, x0:x1]
        if np.count_nonzero(neighbourhood) >= 3:
            dense_count += 1

    density_ratio = dense_count / n_support if n_support > 0 else 0.0

    lines = [
        f"  _describe_support:  n=({seg.normal[0]:.3f},{seg.normal[1]:.3f})  rho={seg.rho:.1f}",
        f"    total support pixels = {n_support}",
        f"    projection range = [{proj_min:.1f}, {proj_max:.1f}]  (span={proj_span:.1f} px)",
        f"    gaps ≥ 1.5 px: {len(gaps)}",
    ]

    for g_proj_a, g_proj_b, g_width in gaps:
        lines.append(f"      gap {g_width:.1f} px  [{g_proj_a:.1f} → {g_proj_b:.1f}]")

    lines.append(
        f"    density: {dense_count}/{n_support} support pixels in dense NMS neighbourhood (ratio={density_ratio:.2f})"
    )

    lines.append("    strongest support pixels (x, y, strength):")
    for idx in strongest:
        lines.append(
            f"      ({pts_sorted[idx][0]:.0f},{pts_sorted[idx][1]:.0f})  "
            f"strength={strengths_sorted[idx]:.1f}"
        )

    lines.append("    weakest support pixels (x, y, strength):")
    for idx in weakest:
        lines.append(
            f"      ({pts_sorted[idx][0]:.0f},{pts_sorted[idx][1]:.0f})  "
            f"strength={strengths_sorted[idx]:.1f}"
        )

    return "\n".join(lines)
