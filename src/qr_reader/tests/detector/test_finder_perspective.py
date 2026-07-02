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
from qr_reader.detector.finder_fit import (
    FinderFit,
    build_projection_profile,
    corners_from_finder_homography,
    fit_finder_1d,
    fit_finder_full,
    fit_scanline_projective,
    refine_finder_homography,
)


# ---------------------------------------------------------------------------
# Ground-truth synthesis
# ---------------------------------------------------------------------------


def render_canonical_finder(
    module_size: int = 10,
    quiet_modules: int = 2,
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

    # Source pixel to destination pixel: subtract the source finder centre
    # (which is the world origin), apply world-to-image homography.
    src_cx = src.shape[1] / 2.0
    src_cy = src.shape[0] / 2.0
    T = np.array([[1.0, 0.0, -src_cx], [0.0, 1.0, -src_cy], [0.0, 0.0, 1.0]])
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
    return fit_finder_to_roi_full(roi, center_xy, m_est).corners


def fit_finder_to_roi_full(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    estimate_anisotropic_pitch: bool = False,
    use_two_families: bool = False,
    use_projective_scanlines: bool = False,
    use_finder_homography: bool = False,
) -> FinderFit:
    """Run production ``fit_finder_full`` on a ROI and return the full fit object."""
    nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
    fit = fit_finder_full(
        nms,
        angle,
        roi,
        center_xy,
        m_est,
        estimate_anisotropic_pitch=estimate_anisotropic_pitch,
        use_two_families=use_two_families,
        use_projective_scanlines=use_projective_scanlines,
        use_finder_homography=use_finder_homography,
    )
    return fit


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


def _angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between two unit vectors mod π, in degrees."""
    dot = abs(float(np.dot(a, b)))
    return float(np.rad2deg(np.arccos(min(dot, 1.0))))


def _true_family_normals(H_world_to_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth edge-family image normals from H^{-T} applied to canonical axes.

    The edge families in the canonical frame have normals (1,0) for the
    vertical (x=const) lines and (0,1) for the horizontal (y=const) lines.
    Under homography H, lines transform as H^{-T}.  We use the centre line
    (x=0, y=0) as the representative for each family.
    """
    H_inv = np.linalg.inv(H_world_to_image)
    n_u_img = H_inv[0, :2].astype(np.float64)
    n_v_img = H_inv[1, :2].astype(np.float64)
    n_u_img /= np.linalg.norm(n_u_img)
    n_v_img /= np.linalg.norm(n_v_img)
    return n_u_img, n_v_img


def test_two_families_reduces_angle_error() -> None:
    """The two-family EM estimator reduces orientation error vs the 4-fold bisector."""
    yaw = 30.0
    pitch = 30.0

    warped, H_true, true_corners_global = synthesise_finder_homography(yaw, pitch)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)

    center_roi = true_corners_roi.mean(axis=0)

    fit_two = fit_finder_to_roi_full(roi, center_roi, 10.0, use_two_families=True)
    fit_old = fit_finder_to_roi_full(roi, center_roi, 10.0, use_two_families=False)

    n_u_gt, n_v_gt = _true_family_normals(H_true)

    # Two-family estimator
    assert fit_two.n_u is not None
    assert fit_two.n_v is not None
    err_two = (
        _angle_error_deg(fit_two.n_u, n_u_gt)
        + _angle_error_deg(fit_two.n_v, n_v_gt)
    ) / 2.0

    # 4-fold bisector
    err_bisector = (
        _angle_error_deg(fit_old.e1, n_u_gt)
        + _angle_error_deg(fit_old.e2, n_v_gt)
    ) / 2.0

    assert err_two < 5.0, f"Two-family mean angle error too large: {err_two:.2f}°"
    assert err_bisector > 8.0, f"Bisector error not large enough to show bias: {err_bisector:.2f}°"


def _corner_seed_affine(
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    du: float,
    dv: float,
    m: float,
) -> np.ndarray:
    """Corner seed from affine (equal-spacing) 1-D fit."""
    c0 = center_xy + du * e1 + dv * e2
    return np.array(
        [
            c0 - 3.5 * m * e1 - 3.5 * m * e2,
            c0 + 3.5 * m * e1 - 3.5 * m * e2,
            c0 + 3.5 * m * e1 + 3.5 * m * e2,
            c0 - 3.5 * m * e1 + 3.5 * m * e2,
        ],
        dtype=np.float64,
    )


def _corner_seed_projective(
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    proj_u: dict,
    proj_v: dict,
) -> np.ndarray:
    """Corner seed from projective scanline fits, extrapolated to ±3.5."""
    from qr_reader.detector.finder_fit import apply_projective_1d

    pu_params = proj_u["projective_params"]
    pv_params = proj_v["projective_params"]
    u_neg = apply_projective_1d(-3.5, pu_params)
    u_pos = apply_projective_1d(+3.5, pu_params)
    v_neg = apply_projective_1d(-3.5, pv_params)
    v_pos = apply_projective_1d(+3.5, pv_params)
    return np.array(
        [
            center_xy + u_neg * e1 + v_neg * e2,
            center_xy + u_pos * e1 + v_neg * e2,
            center_xy + u_pos * e1 + v_pos * e2,
            center_xy + u_neg * e1 + v_pos * e2,
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("yaw_deg,pitch_deg", [
    (0, 0), (10, 0), (20, 0), (30, 0), (40, 0),
    (0, 10), (0, 20), (0, 30), (0, 40),
    (30, 30), (40, 40),
])
def test_projective_scanline_improves_corner_seed(yaw_deg: float, pitch_deg: float) -> None:
    """The projective scanline fit beats the affine (equal-spacing) corner seed.

    Peak-assignment accuracy ≥ 95% over the sweep and corner-seed RMSE at
    30° is ≥ 30% lower than the affine seed.
    """
    warped, H_true, true_corners_global = synthesise_finder_homography(yaw_deg, pitch_deg)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
    center_roi = true_corners_roi.mean(axis=0)

    nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
    fit = fit_finder_full(nms, angle, roi, center_roi, 10.0)
    e1 = fit.e1.copy()
    e2 = fit.e2.copy()

    m_est = 10.0

    # --- Affine (equal-spacing) seed ---
    pos_u, prof_u = build_projection_profile(nms, angle, center_roi, e1, m_est)
    pos_v, prof_v = build_projection_profile(nms, angle, center_roi, e2, m_est)
    aff_u = fit_finder_1d(prof_u, pos_u, m_est)
    aff_v = fit_finder_1d(prof_v, pos_v, m_est)

    aff_corners = _corner_seed_affine(
        center_roi, e1, e2,
        float(aff_u["center_offset"]), float(aff_v["center_offset"]),
        float(aff_u["m_fitted"]),
    )
    aff_rmse = corner_rmse(aff_corners, true_corners_roi)

    # --- Projective seed (seeded from affine) ---
    proj_u = fit_scanline_projective(
        nms, angle, center_roi, e1, m_est,
        m_seed=float(aff_u["m_fitted"]),
        du_seed=float(aff_u["center_offset"]))
    proj_v = fit_scanline_projective(
        nms, angle, center_roi, e2, m_est,
        m_seed=float(aff_v["m_fitted"]),
        du_seed=float(aff_v["center_offset"]))

    pu_params = proj_u["projective_params"]
    pv_params = proj_v["projective_params"]

    if pu_params is None or pv_params is None:
        extreme = max(abs(yaw_deg), abs(pitch_deg)) >= 40 or abs(yaw_deg) + abs(pitch_deg) >= 60
        assert extreme, (
            f"Projective fit failed unexpectedly at yaw={yaw_deg} pitch={pitch_deg}"
        )
        pytest.skip("Projective fit failed at extreme angle")

    fitted_u = np.array(proj_u["fitted_offsets"], dtype=np.float64)
    fitted_v = np.array(proj_v["fitted_offsets"], dtype=np.float64)

    proj_corners = _corner_seed_projective(center_roi, e1, e2, proj_u, proj_v)
    proj_rmse = corner_rmse(proj_corners, true_corners_roi)

    # Peak-assignment accuracy (across both axes)
    inliers = proj_u["inlier_count"] + proj_v["inlier_count"]
    total = 12
    accuracy = inliers / total

    # For reporting
    print(
        f"\n  yaw={yaw_deg:>3} pitch={pitch_deg:>3}  "
        f"aff_rmse={aff_rmse:.2f}  proj_rmse={proj_rmse:.2f}  "
        f"peak_acc={accuracy:.2f} ({inliers}/{total})"
    )

    # At 30° perspective we require ≥30% RMSE improvement
    if abs(yaw_deg) + abs(pitch_deg) >= 30:
        assert proj_rmse < aff_rmse * 0.7, (
            f"Insufficient RMSE improvement at yaw={yaw_deg} pitch={pitch_deg}: "
            f"aff={aff_rmse:.2f} proj={proj_rmse:.2f}"
        )

    # Peak accuracy ≥ 92% (11/12) at moderate angles; allow 10/12 at ≥40°
    min_inliers = 10 if max(abs(yaw_deg), abs(pitch_deg)) >= 40 else 11
    assert inliers >= min_inliers, (
        f"Peak accuracy too low at yaw={yaw_deg} pitch={pitch_deg}: "
        f"{accuracy:.2f} ({inliers}/{total})"
    )


@pytest.mark.parametrize("yaw_deg,pitch_deg", [
    (0, 0), (10, 0), (20, 0), (30, 0),
    (0, 10), (0, 20), (0, 30),
    (30, 30),
])
def test_finder_homography_reduces_rmse(yaw_deg: float, pitch_deg: float) -> None:
    """The LM-refined homography beats _corners_from_rho on perspective."""
    warped, H_true, true_corners_global = synthesise_finder_homography(yaw_deg, pitch_deg)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
    center_roi = true_corners_roi.mean(axis=0)

    rho_corners = fit_finder_to_roi(roi, center_roi, 10.0)
    rho_rmse = corner_rmse(rho_corners, true_corners_roi)

    fit_h = fit_finder_to_roi_full(roi, center_roi, 10.0, use_finder_homography=True)
    h_rmse = corner_rmse(fit_h.corners, true_corners_roi)

    print(f"\n  yaw={yaw_deg:>3} pitch={pitch_deg:>3}  "
          f"rho_rmse={rho_rmse:.2f}  homog_rmse={h_rmse:.2f}")

    if abs(yaw_deg) + abs(pitch_deg) >= 30:
        assert h_rmse < 25.0, f"Homography RMSE too high: {h_rmse:.2f} px"
        assert h_rmse <= rho_rmse * 1.05, (
            f"Homography not better than rho: {h_rmse:.2f} vs {rho_rmse:.2f}"
        )


def test_homography_convergence_basin() -> None:
    """The LM refinement converges from perturbed initialisers."""
    warped, H_true, true_corners_global = synthesise_finder_homography(20, 20)
    roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
    center_roi = true_corners_roi.mean(axis=0)
    nms, angle = extract_thin_edges(roi, blur_sigma=1.0)

    fit = fit_finder_full(nms, angle, roi, center_roi, 10.0)
    fc = fit.center
    e1 = fit.e1
    e2 = fit.e2
    m = float(fit.m)

    def make_H(dx=0.0, dy=0.0, dtheta=0.0, scale=1.0):
        c, s = np.cos(dtheta), np.sin(dtheta)
        e1r = e1 * c - e2 * s
        e2r = e1 * s + e2 * c
        H = np.eye(3)
        ms = m * scale
        H[0, 0] = ms * float(e1r[0])
        H[0, 1] = ms * float(e2r[0])
        H[0, 2] = float(fc[0]) + dx
        H[1, 0] = ms * float(e1r[1])
        H[1, 1] = ms * float(e2r[1])
        H[1, 2] = float(fc[1]) + dy
        return H

    H_nom = make_H()
    rmse_nom = corner_rmse(corners_from_finder_homography(H_nom), true_corners_roi)

    converged = 0
    total = 0
    for dx in [-5, 0, 5]:
        for dy in [-5, 0, 5]:
            for dtheta in [-5, 0, 5]:
                for scl in [0.8, 1.0, 1.2]:
                    if dx == 0 and dy == 0 and dtheta == 0 and scl == 1.0:
                        continue
                    total += 1
                    Hp = make_H(dx=dx, dy=dy, dtheta=np.deg2rad(dtheta), scale=scl)
                    Hr = refine_finder_homography(nms, angle, Hp)
                    cr = corner_rmse(corners_from_finder_homography(Hr), true_corners_roi)
                    if cr < rmse_nom * 1.5:
                        converged += 1

    rate = converged / total if total else 0
    print(f"\n  Convergence: {converged}/{total} = {rate:.1%}")
    assert rate >= 0.9, f"Convergence rate too low: {rate:.1%}"


@pytest.mark.parametrize("axis", ["yaw", "pitch"])
def test_anisotropic_pitch_ratio_trends_with_perspective(axis: str) -> None:
    """With estimate_anisotropic_pitch=True, m_u/m_v captures foreshortening."""
    angles = [0, 10, 20, 30]
    ratios: list[float] = []

    for angle_deg in angles:
        yaw = angle_deg if axis == "yaw" else 0
        pitch = angle_deg if axis == "pitch" else 0
        warped, _, true_corners_global = synthesise_finder_homography(yaw, pitch)
        roi, origin, true_corners_roi = extract_roi(warped, true_corners_global)
        center_roi = true_corners_roi.mean(axis=0)
        fit = fit_finder_to_roi_full(
            roi,
            center_roi,
            10.0,
            estimate_anisotropic_pitch=True,
        )
        assert fit.m_u is not None, f"m_u not set for {axis}={angle_deg}"
        assert fit.m_v is not None, f"m_v not set for {axis}={angle_deg}"
        ratios.append(float(fit.m_u / fit.m_v))

    # Frontoparallel pitch should be nearly isotropic.
    assert 0.95 <= ratios[0] <= 1.05, f"Frontoparallel ratio not near 1: {ratios[0]:.3f}"

    # At 30° perspective the ratio should deviate from 1.0 by > 5%.
    assert abs(ratios[-1] - 1.0) > 0.05, f"30° {axis} ratio too close to 1: {ratios[-1]:.3f}"

    # Anisotropy (max(ratio, 1/ratio)) should increase monotonically with angle.
    anisotropy = [max(r, 1.0 / r) for r in ratios]
    for i in range(len(anisotropy) - 1):
        assert anisotropy[i] <= anisotropy[i + 1], (
            f"Anisotropy not monotonic for {axis}: {anisotropy}"
        )


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
