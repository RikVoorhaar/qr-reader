"""Debug script for joint refinement producing nonsense results.

Reproduces the pipeline up to edge_data, then runs
refine_finder_edges_joint and prints all intermediate diagnostics.

Usage:
    python src/qr_reader/scripts/debug_joint_refinement.py
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edge_fitting import (
    PITCH_CONSTANT,
    MAX_GAP,
    DISTANCE_THRESHOLD,
    build_pair_distance_matrix,
    cluster_pairs,
    compute_boundary_points,
    compute_corners,
    compute_kappa,
    compute_projective_center,
    compute_transition_distances,
    extract_top_clusters,
    assign_points,
    check_joint_refinement_jacobian,
    joint_refinement_jacobian,
    joint_refinement_residuals,
    precompute_mask,
    refine_finder_edges_joint,
    synthesize_template,
    thetarho_to_homogeneous_line,
    _fit_ols_params,
    _reorder_to_standard,
    _all_candidate_info,
    _assign_rays_to_sides,
)
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

# ── Config (mirrors ray-profile.py) ─────────────────────────────────────────
PRESET = "medium"
VERSION = 8
SAMPLE_SEED = 44
NUM_RAYS = 36
NUM_SAMPLES = 120
RAY_LENGTH = 1.0
MASK_BOUNDARY = 4.5
NUM_GRID = 50
GRID_WIDTH = 2.0


# ── Helpers (copied from ray-profile.py) ────────────────────────────────────

def sample_ray_profiles(roi, center_x, center_y, num_rays=36, num_samples=120,
                        ray_length=1.0):
    H_roi, W_roi = roi.shape
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = ray_length * diag_half
    theta = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    dx = np.cos(theta)
    dy = np.sin(theta)
    xs = center_x + np.linspace(0, max_dist, num_samples)[None, :] * dx[:, None]
    ys = center_y + np.linspace(0, max_dist, num_samples)[None, :] * dy[:, None]
    profiles = np.zeros((num_rays, num_samples), dtype=np.float64)
    for i in range(num_rays):
        ix = np.clip(xs[i].astype(int), 0, W_roi - 1)
        iy = np.clip(ys[i].astype(int), 0, H_roi - 1)
        profiles[i] = roi[iy, ix]
    return profiles, max_dist, theta


def normalize_roi_intensities(roi, center_xy, m_est, sigma_factor=1.0):
    H, W = roi.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs.astype(np.float64) - center_xy[0]) ** 2
                   + (ys.astype(np.float64) - center_xy[1]) ** 2)
    sigma = sigma_factor * 3.5 * m_est
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    vals = roi.ravel().astype(np.float64)
    w = weights.ravel()
    order = np.argsort(vals)
    vals_sorted = vals[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    def _wp(pct):
        target = pct / 100.0 * total_w
        idx = int(np.searchsorted(cum_w, target))
        idx = max(0, min(idx, len(vals_sorted) - 1))
        return float(vals_sorted[idx])

    dark = _wp(10.0)
    bright = _wp(90.0)
    span = bright - dark
    if span < 1.0:
        span = 1.0
    roi_norm = np.clip((roi.astype(np.float64) - dark) / span, 0.0, 1.0)
    return roi_norm, dark, bright


def finder_soft_template(t, m, sigma=1.0):
    u = np.abs(np.asarray(t, dtype=np.float64)) / m
    s = sigma / m
    sqrt2 = np.sqrt(2.0)
    result = 0.5 * erfc(-(u - 1.5) / (s * sqrt2))
    result -= 0.5 * erfc(-(u - 2.5) / (s * sqrt2))
    result += 0.5 * erfc(-(u - 3.5) / (s * sqrt2))
    return result


def _masked_mse(t_valid, p_valid, m, mask_boundary, sigma):
    abs_t = np.abs(t_valid)
    inside = abs_t <= mask_boundary * m
    if np.sum(inside) < 3:
        return np.inf
    template = finder_soft_template(t_valid[inside], m, sigma)
    return float(np.mean((template - p_valid[inside]) ** 2))


def fit_m_half_ray(t_samples, profile, m_est, mask_boundary=MASK_BOUNDARY,
                   num_grid=NUM_GRID, grid_width=GRID_WIDTH, sigma=1.0):
    from scipy.optimize import minimize_scalar
    mask = np.isfinite(profile)
    if np.sum(mask) < 10:
        return m_est
    t_valid = t_samples[mask]
    p_valid = profile[mask]
    m_low = m_est / grid_width
    m_high = m_est * grid_width
    m_grid = np.linspace(m_low, m_high, num_grid)
    losses = np.full(num_grid, np.inf)
    for i, m in enumerate(m_grid):
        losses[i] = _masked_mse(t_valid, p_valid, m, mask_boundary, sigma)
    best_idx = np.argmin(losses)
    m_best = m_grid[best_idx]
    if not np.isfinite(losses[best_idx]):
        return m_best
    abs_t = np.abs(t_valid)
    inside = abs_t <= mask_boundary * m_best
    if np.sum(inside) < 3:
        return m_best
    t_refine = t_valid[inside]
    p_refine = p_valid[inside]

    def cost(m_val):
        return float(np.mean((finder_soft_template(t_refine, m_val, sigma)
                              - p_refine) ** 2))

    step = m_grid[1] - m_grid[0] if num_grid > 1 else m_est * 0.05
    result = minimize_scalar(cost, bounds=(m_best - step, m_best + step),
                             method="bounded")
    return float(result.x)


def fit_all_rays(profiles, m_est, max_dist):
    n_rays = profiles.shape[0]
    t_pos = np.linspace(0, max_dist, profiles.shape[1])
    m = np.full(n_rays, np.nan)
    for i in range(n_rays):
        m[i] = fit_m_half_ray(t_pos, profiles[i], m_est)
    return m, None, None


# ── Phase A: Reproduce pipeline up to edge_data ────────────────────────────

def build_edge_data(img_gray, clusters):
    """Reproduce ray-profile.py cells [8]-[10] for all clusters."""
    edge_data = {}
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(img_gray, bbox)
        if roi.size == 0:
            continue
        r0 = max(0, int(bbox[0]))
        c0 = max(0, int(bbox[2]))
        H_roi, W_roi = roi.shape
        c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
        c_row = float(cluster.row) - r0
        center_xy = np.array([c_col, c_row], dtype=np.float64)
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0
        roi_norm, dark_val, bright_val = normalize_roi_intensities(
            roi, center_xy, m_est)
        profiles, max_dist, theta_rad = sample_ray_profiles(
            roi, c_col, c_row, num_rays=NUM_RAYS, num_samples=NUM_SAMPLES,
            ray_length=RAY_LENGTH)
        span = bright_val - dark_val
        if span < 1.0:
            span = 1.0
        profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)
        m, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)
        bp = compute_boundary_points(center_xy, m, theta_rad, PITCH_CONSTANT)
        valid = np.all(np.isfinite(bp), axis=1)
        valid_indices = np.flatnonzero(valid)
        points = bp[valid]
        if len(points) < 4:
            continue
        D, pairs = build_pair_distance_matrix(points, valid_indices, NUM_RAYS,
                                              max_gap=MAX_GAP)
        labels = cluster_pairs(D, distance_threshold=DISTANCE_THRESHOLD)
        top4 = extract_top_clusters(labels, pairs, points, k=4)
        assignment = assign_points(top4, len(points))
        edge_data[ci] = {
            "roi": roi, "center_xy": center_xy,
            "top4": top4, "assignment": assignment,
            "points": points,
            "H_roi": H_roi, "W_roi": W_roi,
            "profiles_norm": profiles_norm,
            "theta_rad": theta_rad,
            "max_dist": max_dist,
            "m_est": m_est,
        }
    return edge_data


# ── Phase B: Diagnostics ───────────────────────────────────────────────────

def run_diagnostics(data):
    """Run all B1-B9 diagnostics on one cluster's edge_data."""
    top4 = data["top4"]
    center_xy = data["center_xy"]
    profiles_norm = data["profiles_norm"]
    theta_rad = data["theta_rad"]
    max_dist = data["max_dist"]
    H_roi = data["H_roi"]
    W_roi = data["W_roi"]
    m_est = data["m_est"]

    half_dirs = np.column_stack([np.cos(theta_rad), np.sin(theta_rad)])
    n_samples = profiles_norm.shape[1]
    s_samples = np.linspace(0, max_dist, n_samples)

    print("=" * 72)
    print("B1 — Reordering")
    print("=" * 72)
    for k, ec in enumerate(top4):
        print(f"  Edge {k}: normal=({ec.normal[0]:+.4f}, {ec.normal[1]:+.4f}), "
              f"rho={ec.rho:.2f}, direction=({ec.direction[0]:+.4f}, {ec.direction[1]:+.4f})")
    l_idx, r_idx, t_idx, b_idx = _reorder_to_standard(top4)
    print(f"  Reorder: L={l_idx}, R={r_idx}, T={t_idx}, B={b_idx}")
    ordered = [top4[l_idx], top4[r_idx], top4[t_idx], top4[b_idx]]
    for name, ec in zip(["L", "R", "T", "B"], ordered):
        theta = np.arctan2(ec.normal[1], ec.normal[0])
        print(f"    {name}: theta={theta:.4f}, rho={ec.rho:.2f}, "
              f"normal=({ec.normal[0]:+.4f}, {ec.normal[1]:+.4f})")

    print()
    print("=" * 72)
    print("B2 — Line construction")
    print("=" * 72)
    theta0 = np.array([np.arctan2(s.normal[1], s.normal[0])
                       for s in ordered], dtype=np.float64)
    rho0 = np.array([s.rho for s in ordered], dtype=np.float64)
    ells = []
    for name, th, rh in zip(["L", "R", "T", "B"], theta0, rho0):
        ell = thetarho_to_homogeneous_line(th, rh)
        ells.append(ell)
        print(f"  {name}: theta={th:.4f}, rho={rh:.2f} → "
              f"line=[{ell[0]:+.4f}, {ell[1]:+.4f}, {ell[2]:+.4f}]")

    print()
    print("=" * 72)
    print("B3 — Projective geometry")
    print("=" * 72)
    ell_L, ell_R, ell_T, ell_B = ells
    corners = compute_corners(ell_L, ell_R, ell_T, ell_B)
    c = compute_projective_center(*corners)
    R = float(np.mean([np.linalg.norm(corner - c) for corner in corners]))
    kappa_u, kappa_v = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)
    print(f"  Corners (LT, RT, RB, LB):")
    for name, corner in zip(["LT", "RT", "RB", "LB"], corners):
        print(f"    {name}: ({corner[0]:.2f}, {corner[1]:.2f})")
    print(f"  Projective center c: ({c[0]:.2f}, {c[1]:.2f})")
    print(f"  Centerpoint: ({center_xy[0]:.2f}, {center_xy[1]:.2f})")
    print(f"  R (mean corner distance): {R:.2f} px")
    print(f"  Expected R ~ 3.5 * m_est = {3.5 * m_est:.2f} px")
    print(f"  kappa_u = {kappa_u:.4f}  (should be ~ +1.0)")
    print(f"  kappa_v = {kappa_v:.4f}  (should be ~ +1.0)")

    print()
    print("=" * 72)
    print("B4 — Masks")
    print("=" * 72)
    per_ray_side = _assign_rays_to_sides(
        center_xy, half_dirs, ell_L, ell_R, ell_T, ell_B)
    print(f"  Per-ray side assignment: {per_ray_side}")
    pre_masks = np.zeros((NUM_RAYS, n_samples), dtype=bool)
    n_fully_masked = 0
    for k in range(NUM_RAYS):
        si = int(per_ray_side[k]) if per_ray_side[k] >= 0 else None
        s_j = compute_transition_distances(
            center_xy, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si)
        pre_masks[k] = precompute_mask(s_samples, s_j, sigma=1.0)
        if not np.any(pre_masks[k]):
            n_fully_masked += 1
    n_unmasked_samples = int(np.sum(pre_masks))
    print(f"  Fully masked rays: {n_fully_masked}/{NUM_RAYS}")
    print(f"  Total unmasked samples: {n_unmasked_samples}/{NUM_RAYS * n_samples}")

    # Print transition distances for a few representative rays
    for k in [0, 4, 5, 9, 13, 18, 22, 27]:
        if k >= NUM_RAYS:
            continue
        si = int(per_ray_side[k]) if per_ray_side[k] >= 0 else None
        s_j = compute_transition_distances(
            center_xy, half_dirs[k],
            ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v,
            side_idx=si)
        angle_deg = np.rad2deg(theta_rad[k])
        print(f"  Ray {k} ({angle_deg:.0f}°, side={si}): s_j = {s_j}")
        if np.all(np.isfinite(s_j)):
            print(f"    s_j/m_est = {s_j / m_est}")

    print()
    print("=" * 72)
    print("B5 — OLS fit")
    print("=" * 72)
    ab_fixed = _fit_ols_params(
        center_xy, profiles_norm, half_dirs, s_samples, pre_masks,
        ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v, sigma=1.0,
        per_ray_side=per_ray_side)
    print(f"  (a, b) = ({ab_fixed[0]:.4f}, {ab_fixed[1]:.4f})  "
          f"(expect a~1, b~0)")

    print()
    print("=" * 72)
    print("B6 — Residual at x0")
    print("=" * 72)
    x0 = np.zeros(8, dtype=np.float64)
    x0[4:8] = rho0
    r0 = joint_refinement_residuals(
        x0, center_xy, R, theta0, profiles_norm, half_dirs,
        s_samples, pre_masks, sigma=1.0, ab_fixed=ab_fixed,
        per_ray_side=per_ray_side)
    cost0 = 0.5 * np.dot(r0, r0)
    print(f"  ||r(x0)|| = {np.linalg.norm(r0):.4f}")
    print(f"  cost(x0) = {cost0:.4f}")
    n_active = int(np.sum(r0 != 0))
    print(f"  Active residuals: {n_active}/{len(r0)}")

    print()
    print("=" * 72)
    print("B7 — Jacobian at x0")
    print("=" * 72)
    J = joint_refinement_jacobian(
        x0, center_xy, R, theta0, profiles_norm, half_dirs,
        s_samples, pre_masks, sigma=1.0, per_ray_side=per_ray_side)
    col_norms = np.linalg.norm(J, axis=0)
    print(f"  Column norms: {col_norms}")
    print(f"  phi/R cols (0-3): mean={np.mean(col_norms[:4]):.4f}")
    print(f"  rho cols (4-7):   mean={np.mean(col_norms[4:]):.4f}")
    cond = np.linalg.cond(J)
    print(f"  cond(J) = {cond:.2e}")
    # FD check on real data
    J_anal, J_fd, max_err = check_joint_refinement_jacobian(
        x0, center_xy, R, theta0, profiles_norm, half_dirs,
        s_samples, pre_masks, sigma=1.0, per_ray_side=per_ray_side)
    print(f"  FD check max error: {max_err:.2e}  (tol=1e-3)")

    print()
    print("=" * 72)
    print("B8 — First LM step")
    print("=" * 72)
    result1 = least_squares(
        fun=lambda x, *args: joint_refinement_residuals(
            x, *args, ab_fixed=ab_fixed, per_ray_side=per_ray_side),
        x0=x0,
        jac=lambda x, *args: joint_refinement_jacobian(
            x, *args, per_ray_side=per_ray_side),
        method="lm",
        args=(center_xy, R, theta0, profiles_norm, half_dirs,
              s_samples, pre_masks, 1.0),
        xtol=1e-6, ftol=1e-6, max_nfev=1, x_scale="jac",
    )
    dx = result1.x - x0
    print(f"  First step dx = {dx}")
    print(f"  |dx| = {np.linalg.norm(dx):.4f}")
    print(f"  dtheta = dx[:4]*R = {dx[:4] * R}")
    print(f"  drho = dx[4:] = {dx[4:]}")

    print()
    print("=" * 72)
    print("B9 — Full LM trace")
    print("=" * 72)
    trace = []

    def callback(x, state=None):
        r = joint_refinement_residuals(
            x, center_xy, R, theta0, profiles_norm, half_dirs,
            s_samples, pre_masks, sigma=1.0, ab_fixed=ab_fixed)
        cost = 0.5 * np.dot(r, r)
        trace.append((len(trace), cost, x.copy()))
        print(f"  nfev={len(trace)-1}: cost={cost:.6f}, "
              f"phi/R={x[:4]}, rho={x[4:]}")
        return False

    result = least_squares(
        fun=lambda x, *args: joint_refinement_residuals(
            x, *args, ab_fixed=ab_fixed, per_ray_side=per_ray_side),
        x0=x0,
        jac=lambda x, *args: joint_refinement_jacobian(
            x, *args, per_ray_side=per_ray_side),
        method="lm",
        args=(center_xy, R, theta0, profiles_norm, half_dirs,
              s_samples, pre_masks, 1.0),
        xtol=1e-6, ftol=1e-6, max_nfev=200, x_scale="jac",
    )
    print(f"\n  Final: success={result.success}, cost={result.cost:.6f}, "
          f"nfev={result.nfev}")
    theta_opt = theta0 + result.x[:4] * R
    rho_opt = result.x[4:8]
    print(f"\n  Results per edge:")
    for name, th0, rh0, th_o, rh_o in zip(
            ["L", "R", "T", "B"], theta0, rho0, theta_opt, rho_opt):
        print(f"    {name}: theta {th0:.4f} → {th_o:.4f} "
              f"(d={th_o - th0:+.4f}),  rho {rh0:.2f} → {rh_o:.2f} "
              f"(d={rh_o - rh0:+.2f})")

    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SAMPLE_SEED)
    preset = PRESET.lower()
    if preset not in PRESET_MAP:
        preset = "medium"
    config = AugmentationConfig(
        version=VERSION,
        content=f"QR v{VERSION}",
        error_correction="M",
        global_seed=SAMPLE_SEED,
        ppm_range=PRESET_MAP[preset].ppm_range,
        target_ppm_range=PRESET_MAP[preset].target_ppm_range,
        jitter_fraction=PRESET_MAP[preset].jitter_fraction,
        feather_sigma_range=PRESET_MAP[preset].feather_sigma_range,
        blur_sigma_range=PRESET_MAP[preset].blur_sigma_range,
        noise_sigma_range=PRESET_MAP[preset].noise_sigma_range,
        jpeg_quality_range=PRESET_MAP[preset].jpeg_quality_range,
    )
    bg_dir = Path("data/images/train")
    bg_paths = sorted(bg_dir.glob("*.jpg"))
    bg_path = bg_paths[SAMPLE_SEED % len(bg_paths)]
    background = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
    image, metadata = generate_sample(rng, config, background)
    img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)
    print(f"Generated v{metadata['version']} QR ({img_gray.shape[1]}×{img_gray.shape[0]})")

    img_binary = binarize_image(img_gray)
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(
        img_binary, max_error)
    clusters = cluster_candidates(rows_valid, cols_valid_all)
    print(f"{len(rows_valid)} 2-D candidates → {len(clusters)} clusters")

    edge_data = build_edge_data(img_gray, clusters)
    print(f"Built edge_data for clusters: {list(edge_data.keys())}")

    for ci, data in edge_data.items():
        print(f"\n{'#' * 72}")
        print(f"# Cluster {ci}  (m_est={data['m_est']:.2f}px)")
        print(f"{'#' * 72}")
        run_diagnostics(data)


if __name__ == "__main__":
    main()
