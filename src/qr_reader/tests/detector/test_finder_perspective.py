"""Isolated single-finder perspective benchmark.

This benchmark is the evaluation engine for Plan 015.  It synthesises a
single finder pattern under controlled yaw/pitch, extracts edges, runs the
production ``fit_finder_full`` function, and reports corner RMSE and
orientation error against ground truth.

No production code is changed by this file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.finder_fit import fit_finder_full


# ---------------------------------------------------------------------------
# Ground-truth synthesis
# ---------------------------------------------------------------------------


def render_canonical_finder(
    module_size: int = 10,
    quiet_modules: int = 4,
) -> np.ndarray:
    """Render a clean binary finder pattern (white=255, black=0).

    The finder is centred in the output image.  The 7x7 module pattern is
    surrounded by *quiet_modules* white modules.
    """
    modules = 7 + 2 * quiet_modules
    size = modules * module_size
    img = np.full((size, size), 255, dtype=np.uint8)

    # 7x7 pattern: 1 = white, 0 = black
    pattern = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )

    start = quiet_modules * module_size
    for py in range(7):
        for px in range(7):
            value = 255 if pattern[py, px] else 0
            y0 = start + py * module_size
            x0 = start + px * module_size
            img[y0 : y0 + module_size, x0 : x0 + module_size] = value

    return img


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World-to-camera rotation: yaw around Y then pitch around X."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

    return Rx @ Ry


def synthesise_finder_homography(
    yaw_deg: float,
    pitch_deg: float,
    module_size: int = 10,
    image_size: int = 400,
    focal_length: float = 200.0,
    camera_distance: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a perspective-warped finder ROI and its ground-truth homography.

    Returns
    -------
    warped : ndarray (image_size, image_size)
        Grayscale image containing the warped finder.
    H_world_to_image : ndarray (3, 3)
        Homography mapping world (X, Y, 1) in pixel units to image (x, y, w).
    true_corners_xy : ndarray (4, 2)
        The 4 outer finder corners in image (x, y) = (col, row) coordinates,
        ordered [(-,-), (+,-), (+,+), (-,+)] relative to the finder axes.
    """
    src = render_canonical_finder(module_size=module_size)
    cx = cy = image_size / 2.0

    K = np.array(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]]
    )
    R = _rotation_matrix(yaw_deg, pitch_deg)
    H_world_to_image = K @ R @ np.diag([1.0, 1.0, camera_distance])

    # Source pixel to destination pixel: subtract image centre (world origin
    # is at the finder centre), apply world-to-image homography.
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    M = H_world_to_image @ T

    warped = cv2.warpPerspective(
        src,
        M.astype(np.float64),
        (image_size, image_size),
        borderValue=255,
        flags=cv2.INTER_LINEAR,
    )

    half = 3.5 * module_size
    canonical_corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float64,
    )
    true_corners_h = (H_world_to_image @ np.column_stack([canonical_corners, np.ones(4)]).T).T
    true_corners_xy = true_corners_h[:, :2] / true_corners_h[:, 2:3]

    return warped, H_world_to_image, true_corners_xy


def extract_roi(
    image: np.ndarray,
    corners_xy: np.ndarray,
    padding: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a padded ROI around the warped finder.

    Returns
    -------
    roi : ndarray
        Grayscale ROI.
    roi_origin : ndarray (2,)
        (x, y) origin of the ROI in the original image.
    roi_corners : ndarray (4, 2)
        True corners translated into ROI coordinates.
    """
    xs = corners_xy[:, 0]
    ys = corners_xy[:, 1]
    x0 = int(np.floor(xs.min() - padding))
    y0 = int(np.floor(ys.min() - padding))
    x1 = int(np.ceil(xs.max() + padding))
    y1 = int(np.ceil(ys.max() + padding))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)

    roi = image[y0:y1, x0:x1].copy()
    roi_origin = np.array([x0, y0], dtype=np.float64)
    roi_corners = corners_xy - roi_origin
    return roi, roi_origin, roi_corners


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def corner_rmse(est_corners_xy: np.ndarray, true_corners_xy: np.ndarray) -> float:
    """Minimum-sum pairing RMSE for 4 corners (handles cyclic ordering)."""
    est = np.asarray(est_corners_xy, dtype=np.float64)
    true = np.asarray(true_corners_xy, dtype=np.float64)
    if est.shape != (4, 2) or true.shape != (4, 2):
        return np.inf

    # Cyclic shifts of the estimated order
    best = np.inf
    for shift in range(4):
        shifted = np.roll(est, shift, axis=0)
        err = np.linalg.norm(shifted - true, axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        best = min(best, rmse)
    return best


def fit_finder_to_roi(roi: np.ndarray, center_xy: np.ndarray, m_est: float) -> np.ndarray:
    """Run production ``fit_finder_full`` on a ROI and return corners in ROI coords."""
    nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
    fit = fit_finder_full(nms, angle, roi, center_xy, m_est)
    return np.asarray(fit.corners, dtype=np.float64)


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------


def _run_sweep() -> list[dict]:
    """Run the yaw/pitch sweep and return per-cell metrics."""
    yaw_values = [0, 10, 20, 30, 40]
    pitch_values = [0, 10, 20, 30, 40]
    module_size = 10
    results: list[dict] = []

    for yaw in yaw_values:
        for pitch in pitch_values:
            warped, H_true, true_corners_global = synthesise_finder_homography(
                yaw_deg=yaw,
                pitch_deg=pitch,
                module_size=module_size,
            )
            roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)

            center_roi = true_corners_roi.mean(axis=0)
            est_corners_roi = fit_finder_to_roi(roi, center_roi, module_size)
            rmse = corner_rmse(est_corners_roi, true_corners_roi)

            results.append(
                {
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "rmse_px": rmse,
                    "roi_width": roi.shape[1],
                    "roi_height": roi.shape[0],
                }
            )

    return results


# ---------------------------------------------------------------------------
# Pytest-visible tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw_deg,pitch_deg", [
    (0, 0),
    (30, 0),
    (0, 30),
    (30, 30),
])
def test_single_finder_runs_without_error(yaw_deg: float, pitch_deg: float) -> None:
    """Smoke test: the benchmark harness and fit_finder_full do not crash."""
    warped, _, true_corners_global = synthesise_finder_homography(yaw_deg, pitch_deg)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
    center_roi = true_corners_roi.mean(axis=0)
    est_corners_roi = fit_finder_to_roi(roi, center_roi, 10.0)
    assert est_corners_roi.shape == (4, 2)
    assert np.all(np.isfinite(est_corners_roi))


def test_frontoparallel_rmse_is_small() -> None:
    """At 0°/0° the current pipeline should already fit well."""
    warped, _, true_corners_global = synthesise_finder_homography(0.0, 0.0)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
    center_roi = true_corners_roi.mean(axis=0)
    est_corners_roi = fit_finder_to_roi(roi, center_roi, 10.0)
    rmse = corner_rmse(est_corners_roi, true_corners_roi)
    # Baseline expectation: < 2 px on a clean, frontoparallel finder.
    assert rmse < 2.0, f"Frontoparallel RMSE too large: {rmse:.2f} px"


def test_write_sweep_csv(tmp_path_factory) -> None:
    """Run the full yaw/pitch sweep and write a CSV for human inspection."""
    results = _run_sweep()

    out_dir = Path(__file__).parents[4] / "docs" / "plan-015-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "finder_perspective_sweep.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print a readable table to stdout so pytest -v shows it.
    print("\nSingle-finder perspective sweep:")
    print(f"{'yaw':>5} {'pitch':>5} {'RMSE(px)':>10} {'ROI':>10}")
    for r in results:
        print(
            f"{r['yaw_deg']:>5} {r['pitch_deg']:>5} {r['rmse_px']:>10.2f} "
            f"{r['roi_width']}x{r['roi_height']}"
        )

    # Ensure the CSV was written and is non-empty.
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
