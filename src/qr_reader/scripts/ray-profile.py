# %%
"""ray-profile.py — Sample radial intensity profiles from cluster centres.

For each cluster, sample pixel intensities along rays at N-degree increments
from the estimated centre.  Plot the rays on the ROI and the 1-D signals.
Goal: understand whether a better ROI boundary can be inferred from these profiles.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
PRESET = "hard"             # 'easy', 'medium', or 'hard'
VERSION = 5                 # QR version (1–40)
SAMPLE_SEED = 42            # Base seed
NUM_RAYS = 36               # Half-ray count (36 directions in [0, 2π))
NUM_SAMPLES = 120           # Samples per half-ray
RAY_LENGTH = 1.0            # Ray length as fraction of half-ROI diagonal
CLUSTER_INDICES = None      # List of cluster indices to plot, or None for all
TIGHT_LAYOUT = True

# ── m-fitting params ──
MASK_BOUNDARY = 4.5         # Mask |t| > 4.5m during fitting (beyond quiet zone)
NUM_GRID = 50               # Grid-search points per half-ray
GRID_WIDTH = 2.0            # Grid bounds: [m_est / GRID_WIDTH, m_est * GRID_WIDTH]

# Colours
C_GOOD = "#2ca02c"
C_GT = "#d62728"
C_E1 = "#17becf"
C_E2 = "#ff7f0e"

# %% [1] Imports
from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

# %% [2] Generate image + run pipeline up to clustering
rng = np.random.default_rng(SAMPLE_SEED)

preset_name = PRESET.lower()
if preset_name not in PRESET_MAP:
    preset_name = "medium"

config = AugmentationConfig(
    version=VERSION,
    content=f"QR v{VERSION}",
    error_correction="M",
    global_seed=SAMPLE_SEED,
    ppm_range=PRESET_MAP[preset_name].ppm_range,
    target_ppm_range=PRESET_MAP[preset_name].target_ppm_range,
    jitter_fraction=PRESET_MAP[preset_name].jitter_fraction,
    feather_sigma_range=PRESET_MAP[preset_name].feather_sigma_range,
    blur_sigma_range=PRESET_MAP[preset_name].blur_sigma_range,
    noise_sigma_range=PRESET_MAP[preset_name].noise_sigma_range,
    jpeg_quality_range=PRESET_MAP[preset_name].jpeg_quality_range,
)

BG_DIR = Path("data/images/train")
bg_paths = sorted(BG_DIR.glob("*.jpg"))
bg_path = bg_paths[SAMPLE_SEED % len(bg_paths)]
background = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
print(f"Background: {bg_path.name} ({background.shape[1]}×{background.shape[0]})")

image, metadata = generate_sample(rng, config, background)
img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)
H_IMG, W_IMG = img_gray.shape

QR_VERSION = metadata["version"]
QR_CONTENT = metadata["payload"]
gt_corners = np.array([
    metadata["corners_qr"]["TL"],
    metadata["corners_qr"]["TR"],
    metadata["corners_qr"]["BR"],
    metadata["corners_qr"]["BL"],
    metadata["corners_qr"]["TL"],
], dtype=np.float64)

print(f"Generated v{QR_VERSION} QR: '{QR_CONTENT}' ({W_IMG}×{H_IMG})")

# ── Full image ──
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_gray, cmap="gray")
ax.plot(gt_corners[:, 0], gt_corners[:, 1], color=C_GT, linewidth=2, label="GT")
for i, lbl in enumerate(["TL", "TR", "BR", "BL"]):
    ax.text(gt_corners[i, 0] + 3, gt_corners[i, 1] + 3, lbl, color=C_GT, fontsize=8, weight="bold")
ax.set_title(f"Input — v{QR_VERSION} ({PRESET.upper()})  bg: {bg_path.name}")
ax.legend(fontsize=8)
ax.axis("off")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()




# ── Pipeline: binarize → alignment scan → cluster ──
img_binary = binarize_image(img_gray)
max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"{len(rows_valid)} 2-D candidates → {len(clusters)} clusters")

# %% [3] Ray-sampling helper


def sample_ray_profiles(
    roi: np.ndarray,
    center_x: float,
    center_y: float,
    num_rays: int = 36,
    num_samples: int = 120,
    ray_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample pixel intensities along half-rays outward from a centre point.

    Parameters
    ----------
    roi : ndarray (H, W)
        Grayscale ROI.
    center_x, center_y : float
        Ray origin in ROI-local (x=col, y=row) coordinates.
    num_rays : int
        Number of equally-spaced half-rays in ``[0, 2π)``.
    num_samples : int
        Number of sample points per half-ray.
    ray_length : float
        Ray extent as a fraction of half the ROI diagonal length.

    Returns
    -------
    profiles : ndarray (num_rays, num_samples)
        Sampled intensities.  Each row is one half-ray from centre outward
        (distance 0 → max_dist).
    ray_endpoints : ndarray (num_rays, 4)
        Each row: [centre_x, centre_y, end_x, end_y] for drawing the ray.
    angles_deg : ndarray (num_rays,)
        Ray angles in degrees (0 = rightward, CCW).
    """
    H_roi, W_roi = roi.shape
    roi_f = roi.astype(np.float64)
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = ray_length * diag_half

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    angles_deg = np.rad2deg(angles)

    profiles = np.full((num_rays, num_samples), np.nan, dtype=np.float64)
    ray_endpoints = np.zeros((num_rays, 4), dtype=np.float64)

    for i, theta in enumerate(angles):
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Endpoints for drawing the half-ray (centre → outward)
        ray_endpoints[i] = [center_x, center_y,
                            center_x + max_dist * dx,
                            center_y + max_dist * dy]

        # Sample from centre outward
        sample_ts = np.linspace(0, max_dist, num_samples)
        sx = center_x + sample_ts * dx
        sy = center_y + sample_ts * dy

        # Bilinear interpolation
        x0 = np.clip(np.floor(sx).astype(int), 0, W_roi - 1)
        y0 = np.clip(np.floor(sy).astype(int), 0, H_roi - 1)
        x1 = np.clip(x0 + 1, 0, W_roi - 1)
        y1 = np.clip(y0 + 1, 0, H_roi - 1)
        fx = sx - x0.astype(np.float64)
        fy = sy - y0.astype(np.float64)

        profiles[i] = ((1 - fy) * ((1 - fx) * roi_f[y0, x0] + fx * roi_f[y0, x1])
                       + fy * ((1 - fx) * roi_f[y1, x0] + fx * roi_f[y1, x1]))

    profiles = np.clip(profiles, 0, 255)
    return profiles, ray_endpoints, angles_deg


def plot_ray_profiles(
    roi: np.ndarray,
    center_xy: np.ndarray,
    profiles: np.ndarray,
    ray_endpoints: np.ndarray,
    angles_deg: np.ndarray,
    m_est: float,
    ci: int,
):
    """Two-panel figure: half-rays overlaid on ROI, and heatmap of half-ray profiles."""
    H_roi, W_roi = roi.shape
    num_rays = len(angles_deg)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cluster {ci} — {num_rays} half-ray profiles  "
                 f"(m_est={m_est:.2f}px)", fontsize=13, fontweight="bold")

    # ── Left: ROI with half-rays ──
    ax = axes[0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])

    cmap_rays = plt.cm.hsv
    for i, theta_deg in enumerate(angles_deg):
        color = cmap_rays(i / num_rays)
        ep = ray_endpoints[i]
        ax.plot([ep[0], ep[2]], [ep[1], ep[3]],
                color=color, linewidth=1, alpha=0.6)

    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)
    ax.set_title(f"ROI + half-rays (0° = rightward, CCW)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # ── Right: half-ray profile heatmap ──
    ax = axes[1]
    num_samples = profiles.shape[1]
    step = (angles_deg[1] - angles_deg[0]) if num_rays > 1 else 360.0 / num_rays
    # imshow y-extent: row 0 (angle 0) at y=0 (top), last row at y=360 (bottom)
    img_extent = [0, num_samples - 1, angles_deg[-1] + step, angles_deg[0]]

    im = ax.imshow(profiles, aspect="auto", cmap="gray",
                   extent=img_extent, interpolation="nearest")
    ax.set_xlabel("Distance from centre (px)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Half-ray intensity profiles (light = bright pixel)")

    # Horizontal lines at the 4 canonical finder orientations
    for deg in np.arange(0, 360, 90):
        ax.axhline(deg, color=C_GT, linestyle="--", linewidth=1, alpha=0.4)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Intensity")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()


# %% [4] Process each cluster
import matplotlib.colors as mcolors

show_indices = CLUSTER_INDICES or list(range(len(clusters)))

for ci in show_indices:
    if ci >= len(clusters):
        break

    cluster = clusters[ci]
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)
    if roi.size == 0:
        print(f"  Cluster {ci}: empty ROI, skipping")
        continue

    r0 = max(0, int(bbox[0]))
    c0 = max(0, int(bbox[2]))
    H_roi, W_roi = roi.shape

    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
    c_row = float(cluster.row) - r0
    center_xy = np.array([c_col, c_row], dtype=np.float64)
    m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0
    width_px = float(cluster.cols[5] - cluster.cols[0])

    print(f"\nCluster {ci}: centre=({c_col:.1f}, {c_row:.1f}), "
          f"width={width_px:.1f}px, m_est={m_est:.2f}px")

    profiles, ray_endpoints, angles_deg = sample_ray_profiles(
        roi, c_col, c_row,
        num_rays=NUM_RAYS,
        num_samples=NUM_SAMPLES,
        ray_length=RAY_LENGTH,
    )

    # ── Clip half-ray endpoints to ROI boundary ──
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = RAY_LENGTH * diag_half
    clipped_endpoints = ray_endpoints.copy()
    for i in range(NUM_RAYS):
        dx = np.cos(np.deg2rad(angles_deg[i]))
        dy = np.sin(np.deg2rad(angles_deg[i]))
        ts = []
        if abs(dx) > 1e-12:
            ts.append((W_roi - 1 - c_col) / dx)
            ts.append((0 - c_col) / dx)
        if abs(dy) > 1e-12:
            ts.append((H_roi - 1 - c_row) / dy)
            ts.append((0 - c_row) / dy)
        ts_pos = [t for t in ts if t > 0]
        t_pos = min(ts_pos) if ts_pos else max_dist
        if t_pos > max_dist:
            t_pos = max_dist
        clipped_endpoints[i] = [c_col, c_row,
                                c_col + t_pos * dx, c_row + t_pos * dy]

    # ── Inline plot (half-rays + half-ray profile heatmap) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cluster {ci} — {NUM_RAYS} half-ray profiles  "
                 f"(m_est={m_est:.2f}px)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    cmap_rays = plt.cm.hsv
    for i in range(NUM_RAYS):
        ep = clipped_endpoints[i]
        ax.plot([ep[0], ep[2]], [ep[1], ep[3]],
                color=cmap_rays(i / NUM_RAYS), linewidth=1, alpha=0.6)
    ax.plot(c_col, c_row, "r+", markersize=12, markeredgewidth=2)
    ax.set_title("ROI + half-rays")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    ax = axes[1]
    step = (angles_deg[1] - angles_deg[0]) if NUM_RAYS > 1 else 360.0 / NUM_RAYS
    img_extent = [0, max_dist, angles_deg[-1] + step, angles_deg[0]]
    ax.imshow(profiles, aspect="auto", cmap="gray",
              extent=img_extent, interpolation="nearest")
    # Canonical boundary lines at 1.5m, 2.5m, 3.5m
    for k, ls in [(1.5, "--"), (2.5, "--"), (3.5, "-")]:
        ax.axvline(+k * m_est, color=C_GT, linestyle=ls, linewidth=1, alpha=0.5)
    ax.axvline(0, color="white", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Distance from centre (px)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Half-ray intensity profiles")

    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [5] Normalization + template fitting helpers
from scipy.optimize import minimize_scalar
from scipy.special import erfc


def finder_soft_template(
    t: np.ndarray, m: float, sigma: float = 1.0
) -> np.ndarray:
    """Find pattern intensity template along a radial ray.

    The ideal finder pattern cross-section from centre outward (in module
    units) has a 3-module-wide dark centre, then a 1-module white ring, a
    1-module dark ring, and the white quiet zone::

        dark (0) --[1.5]--> white (1) --[2.5]--> dark (0) --[3.5]--> white (1)

    Transitions are smoothed via ``erfc`` (complementary error function).

    Parameters
    ----------
    t : ndarray
        Signed distances from centre in pixels.
    m : float
        Module pitch in pixels.
    sigma : float
        Smoothing scale in pixels (default 1.0).

    Returns
    -------
    template : ndarray
        Expected normalised intensities in [0, 1].
    """
    u = np.abs(np.asarray(t, dtype=np.float64)) / m
    s = sigma / m
    sqrt2 = np.sqrt(2.0)

    result = 0.5 * erfc(-(u - 1.5) / (s * sqrt2))
    result -= 0.5 * erfc(-(u - 2.5) / (s * sqrt2))
    result += 0.5 * erfc(-(u - 3.5) / (s * sqrt2))
    return result


def normalize_roi_intensities(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    sigma_factor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize ROI intensities to [0, 1] using centre-weighted percentiles.

    Pixels near the finder-pattern centre are weighted more heavily so that
    the dark/bright mapping reflects the finder pattern's contrast, not the
    background's.

    Returns
    -------
    roi_norm : ndarray (H, W)
        Normalized intensities in [0, 1].
    dark : float
        Weighted p10 (mapped to 0).
    bright : float
        Weighted p90 (mapped to 1).
    """
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

    def _weighted_percentile(percentile: float) -> float:
        if total_w == 0.0:
            return float(np.percentile(vals, percentile))
        target = percentile / 100.0 * total_w
        idx = int(np.searchsorted(cum_w, target))
        idx = max(0, min(idx, len(vals_sorted) - 1))
        return float(vals_sorted[idx])

    dark = _weighted_percentile(10.0)
    bright = _weighted_percentile(90.0)

    span = bright - dark
    if span < 1.0:
        span = 1.0
    roi_norm = np.clip((roi.astype(np.float64) - dark) / span, 0.0, 1.0)
    return roi_norm, dark, bright


def _masked_mse(t_valid, p_valid, m, mask_boundary, sigma):
    """MSE per unmasked sample between profile and finder soft-template.

    Masked region (not used in loss): |t| > mask_boundary * m.
    """
    abs_t = np.abs(t_valid)
    inside_mask = abs_t <= mask_boundary * m
    n_inside = int(np.sum(inside_mask))
    if n_inside < 3:
        return np.inf
    template = finder_soft_template(t_valid[inside_mask], m, sigma)
    return float(np.mean((template - p_valid[inside_mask]) ** 2))


def fit_m_half_ray(
    t_samples: np.ndarray,
    profile: np.ndarray,
    m_est: float,
    mask_boundary: float = MASK_BOUNDARY,
    num_grid: int = NUM_GRID,
    grid_width: float = GRID_WIDTH,
    sigma: float = 1.0,
) -> dict:
    """Fit *m* to a single half-ray profile via grid search + bounded refine.

    Phase 1  — grid search (masked loss, mask recomputed per grid point).
    Phase 2  — ``minimize_scalar`` within ±1 grid step of winner,
               mask frozen at the winning *m*.
    """
    mask = np.isfinite(profile)
    if np.sum(mask) < 10:
        return {"m_fitted": m_est, "mse": np.inf, "success": False}

    t_valid = t_samples[mask]
    p_valid = profile[mask]

    # ── Phase 1: grid search ──
    m_low = m_est / grid_width
    m_high = m_est * grid_width
    m_grid = np.linspace(m_low, m_high, num_grid)
    losses = np.full(num_grid, np.inf, dtype=np.float64)
    for i, m in enumerate(m_grid):
        losses[i] = _masked_mse(t_valid, p_valid, m, mask_boundary, sigma)

    best_idx = np.argmin(losses)
    m_best = m_grid[best_idx]
    best_loss = losses[best_idx]

    if not np.isfinite(best_loss):
        return {"m_fitted": m_est, "mse": best_loss, "success": False}

    # ── Phase 2: bounded refine around winner, mask frozen ──
    abs_t = np.abs(t_valid)
    inside_mask = abs_t <= mask_boundary * m_best
    n_inside = int(np.sum(inside_mask))
    if n_inside < 3:
        return {"m_fitted": m_best, "mse": best_loss, "success": True}

    t_refine = t_valid[inside_mask]
    p_refine = p_valid[inside_mask]

    def cost(m_val):
        template = finder_soft_template(t_refine, m_val, sigma)
        return float(np.mean((template - p_refine) ** 2))

    step = m_grid[1] - m_grid[0] if num_grid > 1 else m_est * 0.05
    result = minimize_scalar(
        cost,
        bounds=(m_best - step, m_best + step),
        method="bounded",
    )
    return {
        "m_fitted": float(result.x),
        "mse": float(result.fun),
        "success": bool(result.success),
    }


def fit_all_rays(
    profiles: np.ndarray,
    m_est: float,
    max_dist: float,
    mask_boundary: float = MASK_BOUNDARY,
    num_grid: int = NUM_GRID,
    grid_width: float = GRID_WIDTH,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit *m* independently to each half-ray profile.

    Returns
    -------
    m : ndarray (num_rays,)
    mse : ndarray (num_rays,)
    success : ndarray (num_rays,) bool
    """
    n_rays = profiles.shape[0]
    t_pos = np.linspace(0, max_dist, profiles.shape[1])

    m = np.full(n_rays, np.nan, dtype=np.float64)
    mse_arr = np.full(n_rays, np.nan, dtype=np.float64)
    success = np.full(n_rays, False)

    for i in range(n_rays):
        res = fit_m_half_ray(t_pos, profiles[i], m_est,
                             mask_boundary, num_grid, grid_width, sigma)
        m[i] = res["m_fitted"]
        mse_arr[i] = res["mse"]
        success[i] = res["success"]

    return m, mse_arr, success


# %% [6] Normalize ROI + fit per-ray m, overlay on heatmap
show_indices = CLUSTER_INDICES or list(range(len(clusters)))

for ci in show_indices:
    if ci >= len(clusters):
        break

    cluster = clusters[ci]
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
    width_px = float(cluster.cols[5] - cluster.cols[0])

    # ── Normalize ROI ──
    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est)

    # ── Sample rays ──
    profiles, ray_endpoints, angles_deg = sample_ray_profiles(
        roi, c_col, c_row,
        num_rays=NUM_RAYS,
        num_samples=NUM_SAMPLES,
        ray_length=RAY_LENGTH,
    )
    span = bright_val - dark_val
    if span < 1.0:
        span = 1.0
    profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)

    # ── Fit half-rays ──
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = RAY_LENGTH * diag_half
    m, mse, success = fit_all_rays(profiles_norm, m_est, max_dist)

    ok = success & np.isfinite(m)
    n_ok = int(np.sum(ok))
    n_total = NUM_RAYS
    m_median = float(np.median(m[ok])) if n_ok > 0 else m_est
    print(f"\nCluster {ci}: m_est={m_est:.2f}px, m_fitted median={m_median:.2f}px  "
          f"({n_ok}/{n_total})")

    # ── Figure: heatmap with fitted m overlays ──
    fig, ax_heatmap = plt.subplots(figsize=(11, 7))
    fig.suptitle(f"Cluster {ci} — m_est={m_est:.2f} → m_fit median={m_median:.2f}  "
                 f"(grey dashed = m_est)",
                 fontsize=12, fontweight="bold")

    step = (angles_deg[1] - angles_deg[0]) if NUM_RAYS > 1 else 360.0 / NUM_RAYS
    img_extent = [0, max_dist, angles_deg[-1] + step, angles_deg[0]]

    im = ax_heatmap.imshow(profiles_norm, aspect="auto", cmap="gray",
                           extent=img_extent, interpolation="nearest")
    ax_heatmap.set_xlabel("Distance from centre (px)")
    ax_heatmap.set_ylabel("Angle (deg)")

    # Angles aligned to imshow cell centres (half-step down) + wrap-around
    half = step / 2.0
    angles_ctr = angles_deg + half
    angles_plot = np.append(angles_ctr, angles_ctr[0] + 360.0)

    def _wrap(a: np.ndarray) -> np.ndarray:
        return np.append(a, a[0])

    ax_heatmap.plot(1.5 * _wrap(m), angles_plot,
                    "o-", color=C_E2, markersize=3, linewidth=1.0, alpha=0.6,
                    label="1.5m")
    ax_heatmap.plot(2.5 * _wrap(m), angles_plot,
                    "o-", color=C_E1, markersize=3, linewidth=1.0, alpha=0.6,
                    label="2.5m")
    ax_heatmap.plot(3.5 * _wrap(m), angles_plot,
                    "o-", color=C_GOOD, markersize=3, linewidth=1.5, alpha=0.8,
                    label="3.5m")
    ax_heatmap.set_ylim(img_extent[3], img_extent[2])  # [top, bottom]
    ax_heatmap.legend(fontsize=7)

    plt.colorbar(im, ax=ax_heatmap, shrink=0.85, label="Normalized intensity")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [7] Estimate new ROI from 3.5m boundary points
show_indices = CLUSTER_INDICES or list(range(len(clusters)))
PITCH_CONSTANT=3.5
for ci in show_indices:
    if ci >= len(clusters):
        break

    cluster = clusters[ci]
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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est)
    profiles, ray_endpoints, angles_deg = sample_ray_profiles(
        roi, c_col, c_row,
        num_rays=NUM_RAYS,
        num_samples=NUM_SAMPLES,
        ray_length=RAY_LENGTH,
    )
    span = bright_val - dark_val
    if span < 1.0:
        span = 1.0
    profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)

    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = RAY_LENGTH * diag_half
    m, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)

    # ── Compute 3.5m boundary points for each half-ray ──
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    boundary_points = []
    for i in range(NUM_RAYS):
        dx = np.cos(theta_rad[i])
        dy = np.sin(theta_rad[i])
        if np.isfinite(m[i]):
            r = PITCH_CONSTANT * m[i]
            boundary_points.append(center_xy + r * np.array([dx, dy]))

    if not boundary_points:
        print(f"Cluster {ci}: no valid boundary points")
        continue

    boundary_pts = np.array(boundary_points)
    x_min, x_max = boundary_pts[:, 0].min(), boundary_pts[:, 0].max()
    y_min, y_max = boundary_pts[:, 1].min(), boundary_pts[:, 1].max()
    new_width = x_max - x_min
    new_height = y_max - y_min

    print(f"\nCluster {ci}: original ROI {W_roi}×{H_roi}, "
          f"new bbox {new_width:.0f}×{new_height:.0f}px")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)

    # Boundary points — one per half-ray
    ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1],
               c=angles_deg[:len(boundary_pts)], cmap="hsv", s=20, 
               edgecolors="white", linewidths=0.3, zorder=3)

    # New bounding box
    rect = plt.Rectangle((x_min, y_min), new_width, new_height,
                          fill=False, edgecolor=C_GOOD, linewidth=2)
    ax.add_patch(rect)
    ax.set_title(f"Cluster {ci}: 3.5m boundary points + estimated ROI  "
                 f"({new_width:.0f}×{new_height:.0f}px)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [8] Phase 0 — Pairwise σ₂/σ₁ distance matrix + initial pair clusters
from qr_reader.detector.edge_fitting import (
    DISTANCE_THRESHOLD,
    MAX_GAP,
    PITCH_CONSTANT,
    assign_points,
    build_pair_distance_matrix,
    cluster_pairs,
    compute_boundary_points,
    extract_top_clusters,
)

show_indices = CLUSTER_INDICES or list(range(len(clusters)))
edge_data: dict[int, dict] = {}

for ci in show_indices:
    if ci >= len(clusters):
        break

    cluster = clusters[ci]
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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est)
    profiles, ray_endpoints, angles_deg = sample_ray_profiles(
        roi, c_col, c_row, num_rays=NUM_RAYS, num_samples=NUM_SAMPLES,
        ray_length=RAY_LENGTH,
    )
    span = bright_val - dark_val
    if span < 1.0:
        span = 1.0
    profiles_norm = np.clip((profiles - dark_val) / span, 0.0, 1.0)

    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = RAY_LENGTH * diag_half
    m, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)

    bp = compute_boundary_points(center_xy, m, theta_rad,
                                 PITCH_CONSTANT)
    valid = np.all(np.isfinite(bp), axis=1)
    valid_indices = np.flatnonzero(valid)
    points = bp[valid]
    M = len(points)
    if M < 4:
        print(f"Cluster {ci}: only {M} valid boundary points, skipping")
        continue

    D, pairs = build_pair_distance_matrix(points, valid_indices, NUM_RAYS,
                                          max_gap=MAX_GAP)

    edge_data[ci] = {
        "roi": roi, "center_xy": center_xy,
        "bp": bp, "points": points, "valid_indices": valid_indices,
        "D": D, "pairs": pairs,
        "H_roi": H_roi, "W_roi": W_roi,
        "profiles_norm": profiles_norm,
        "theta_rad": theta_rad,
        "max_dist": max_dist,
    }

    n_comparable = int((np.sum(D < 1.0) - M) // 2)  # off-diagonal, symmetric
    print(f"Cluster {ci}: {M}/{NUM_RAYS} valid boundary points, "
          f"{n_comparable} comparable pair-pairs")

    # ── Plot 0: distance-matrix heatmap + initial pairs on ROI ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Cluster {ci} — Phase 0: pairwise σ₂/σ₁ distances",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    im = ax.imshow(D, cmap="viridis", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xlabel("Initial pair index")
    ax.set_ylabel("Initial pair index")
    ax.set_title(f"σ₂/σ₁ distance matrix  (1.0 = gap > {MAX_GAP})")
    plt.colorbar(im, ax=ax, shrink=0.8, label="σ₂/σ₁")

    ax = axes[1]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)
    ax.scatter(points[:, 0], points[:, 1],
               c=np.arange(M), cmap="hsv", s=30, edgecolors="white",
               linewidths=0.4, zorder=3)
    for j in range(M):
        seg = points[pairs[j]]
        ax.plot(seg[:, 0], seg[:, 1], "-", color="white", linewidth=0.8,
                alpha=0.6)
    ax.set_title(f"{M} boundary points + {M} initial pairs")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [9] Phase 1 — Sklearn single-linkage agglomerative clustering
for ci, data in edge_data.items():
    D = data["D"]
    pairs = data["pairs"]
    points = data["points"]
    roi = data["roi"]
    center_xy = data["center_xy"]
    H_roi, W_roi = data["H_roi"], data["W_roi"]
    M = len(points)

    labels = cluster_pairs(D, distance_threshold=DISTANCE_THRESHOLD)
    data["labels"] = labels
    n_clusters_found = len(set(labels.tolist()))
    print(f"Cluster {ci}: sklearn found {n_clusters_found} clusters "
          f"(threshold={DISTANCE_THRESHOLD})")

    # ── Plot 1: points coloured by their leading pair's label ──
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)

    cmap = plt.cm.tab20
    for j in range(M):
        color = cmap(labels[j] % 20)
        ax.plot(points[j, 0], points[j, 1], "o", color=color, markersize=7,
                markeredgewidth=0.5, markeredgecolor="white")

    # Overlay TLS lines for the 4 largest clusters (preview of Phase 2)
    top4_preview = extract_top_clusters(labels, pairs, points, k=4)
    for ec in top4_preview:
        support_pts = points[ec.support]
        proj = support_pts @ ec.direction
        lo, hi = float(proj.min()), float(proj.max())
        ext = (hi - lo) * 0.2
        t_vals = np.array([lo - ext, hi + ext])
        line_pts = ec.rho * ec.normal + t_vals[:, None] * ec.direction
        ax.plot(line_pts[:, 0], line_pts[:, 1], "-", color="white",
                linewidth=1.5, alpha=0.7)

    ax.set_title(f"Phase 1 — sklearn labels  (cluster {ci}, "
                 f"{n_clusters_found} clusters)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [10] Phase 2 — Top-4 extraction + TLS lines + tie-broken assignment
seg_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

for ci, data in edge_data.items():
    if "labels" not in data:
        continue
    pairs = data["pairs"]
    points = data["points"]
    labels = data["labels"]
    roi = data["roi"]
    center_xy = data["center_xy"]
    H_roi, W_roi = data["H_roi"], data["W_roi"]
    M = len(points)

    top4 = extract_top_clusters(labels, pairs, points, k=4)
    assignment = assign_points(top4, M)
    data["top4"] = top4
    data["assignment"] = assignment

    print(f"\nCluster {ci}: top-4 edge clusters")
    for ei, ec in enumerate(top4):
        print(f"  Edge {ei}: {len(ec.pair_indices)} pairs, "
              f"{len(ec.support)} support pts, "
              f"σ₂/σ₁={ec.sigma_ratio:.4f}, "
              f"n=({ec.normal[0]:+.3f},{ec.normal[1]:+.3f}), ρ={ec.rho:.1f}")

    # ── Plot 2: tie-broken point-to-edge assignment + fitted lines ──
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)

    for j in range(M):
        a = assignment[j]
        color = seg_colors[a] if a >= 0 else "gray"
        ax.plot(points[j, 0], points[j, 1], "o", color=color, markersize=7,
                markeredgewidth=0.5, markeredgecolor="white")

    for ei, ec in enumerate(top4):
        assigned_pts = points[assignment == ei]
        if len(assigned_pts) == 0:
            continue
        proj = assigned_pts @ ec.direction
        lo, hi = float(proj.min()), float(proj.max())
        ext = (hi - lo) * 0.2
        t_vals = np.array([lo - ext, hi + ext])
        line_pts = ec.rho * ec.normal + t_vals[:, None] * ec.direction
        ax.plot(line_pts[:, 0], line_pts[:, 1], "-", color=seg_colors[ei],
                linewidth=2.5, alpha=0.9, label=f"Edge {ei}")

    ax.set_title(f"Phase 2 — tie-broken assignment  (cluster {ci})")
    ax.legend(fontsize=8)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()


# %% [11] Step 1 — Actual vs theoretical template profiles

for ci, data in edge_data.items():
    if "top4" not in data:
        continue
    top4 = data["top4"]
    profiles_norm = data["profiles_norm"]
    center_xy = data["center_xy"]
    theta_rad = data["theta_rad"]
    max_dist = data["max_dist"]
    H_roi = data["H_roi"]
    W_roi = data["W_roi"]

    half_dirs = np.column_stack([np.cos(theta_rad), np.sin(theta_rad)])
    n_rays = len(theta_rad)
    n_samples = profiles_norm.shape[1]
    t_samples = np.linspace(0, max_dist, n_samples)

    # Assign each half-ray to the segment with smallest positive t
    assignment = np.full(n_rays, -1, dtype=int)
    t_assigned = np.full(n_rays, np.nan, dtype=np.float64)

    for j in range(n_rays):
        d = half_dirs[j]
        best_t = np.inf
        best_idx = -1
        for si, seg in enumerate(top4):
            denom = seg.normal @ d
            if abs(denom) < 1e-12:
                continue
            t = (seg.rho - seg.normal @ center_xy) / denom
            if 0 < t < best_t:
                best_t = t
                best_idx = si
        if best_idx >= 0:
            assignment[j] = best_idx
            t_assigned[j] = best_t

    # Build theoretical profiles
    theoretical = np.zeros((n_rays, n_samples), dtype=np.float64)
    for j in range(n_rays):
        if assignment[j] >= 0:
            m = t_assigned[j] / PITCH_CONSTANT
            theoretical[j] = finder_soft_template(t_samples, m, sigma=1.0)

    # ── Plot: actual (left) vs theoretical (right) ──
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cluster {ci} — actual vs theoretical profiles",
                 fontsize=13, fontweight="bold")

    step = (np.rad2deg(theta_rad[1] - theta_rad[0]) if n_rays > 1
            else 360.0 / n_rays)
    img_extent = [0, max_dist,
                  np.rad2deg(theta_rad[-1]) + step, np.rad2deg(theta_rad[0])]

    ax_l.imshow(profiles_norm, aspect="auto", cmap="gray",
                extent=img_extent, interpolation="nearest")
    ax_l.set_title("Actual profiles")
    ax_l.set_xlabel("Distance from centre (px)")
    ax_l.set_ylabel("Angle (deg)")

    ax_r.imshow(theoretical, aspect="auto", cmap="gray",
                vmin=0.0, vmax=1.0, extent=img_extent, interpolation="nearest")
    ax_r.set_title("Theoretical templates")
    ax_r.set_xlabel("Distance from centre (px)")
    ax_r.set_ylabel("Angle (deg)")

    counts = [int(np.sum(assignment == si)) for si in range(len(top4))]
    print(f"Cluster {ci}: half-ray assignment — " + ", ".join(
        f"Edge {si}: {c}" for si, c in enumerate(counts)))

    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()
