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
PRESET = "easy"             # 'easy', 'medium', or 'hard'
VERSION = 5                 # QR version (1–40)
SAMPLE_SEED = 42            # Base seed
NUM_RAYS = 36               # Angular increments (10° spacing)
NUM_SAMPLES = 120           # Samples per half-ray (total profile length = 2×NUM_SAMPLES−1)
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
    """Sample pixel intensities along rays from a centre point.

    Parameters
    ----------
    roi : ndarray (H, W)
        Grayscale ROI.
    center_x, center_y : float
        Ray origin in ROI-local (x=col, y=row) coordinates.
    num_rays : int
        Number of equally-spaced rays (full circle).
    num_samples : int
        Number of sample points per half-ray (total samples = 2×num_samples−1 on the full line).
    ray_length : float
        Ray extent as a fraction of half the ROI diagonal length.

    Returns
    -------
    profiles : ndarray (num_rays, 2 * num_samples - 1)
        Sampled intensities.  Rows = rays, columns = signed distance from centre
        (negative = opposite direction, centre at column `num_samples − 1`).
    ray_endpoints : ndarray (num_rays, 4)
        Each row: [start_x, start_y, end_x, end_y] for drawing the full line.
    angles_deg : ndarray (num_rays,)
        Ray angles in degrees (0 = rightward, CCW).
    """
    H_roi, W_roi = roi.shape
    roi_f = roi.astype(np.float64)
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = ray_length * diag_half

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    angles_deg = np.rad2deg(angles)

    n_full = 2 * num_samples - 1
    profiles = np.full((num_rays, n_full), np.nan, dtype=np.float64)
    ray_endpoints = np.zeros((num_rays, 4), dtype=np.float64)

    for i, theta in enumerate(angles):
        dx = np.cos(theta)
        dy = np.sin(theta)

        # Endpoints for drawing the full line (both directions)
        ray_endpoints[i] = [center_x - max_dist * dx,
                            center_y - max_dist * dy,
                            center_x + max_dist * dx,
                            center_y + max_dist * dy]

        # Sample along the full line: −max_dist … 0 … +max_dist
        sample_ts = np.linspace(-max_dist, max_dist, n_full)
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
    """Two-panel figure: rays overlaid on ROI, and polar heatmap of profiles."""
    H_roi, W_roi = roi.shape
    num_rays = len(angles_deg)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cluster {ci} — {num_rays} radial profiles  "
                 f"(m_est={m_est:.2f}px)", fontsize=13, fontweight="bold")

    # ── Left: ROI with rays ──
    ax = axes[0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])

    cmap_rays = plt.cm.hsv
    for i, theta_deg in enumerate(angles_deg):
        color = cmap_rays(i / num_rays)
        ep = ray_endpoints[i]
        ax.plot([ep[0], ep[2]], [ep[1], ep[3]],
                color=color, linewidth=1, alpha=0.6)

    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)
    ax.set_title(f"ROI + sample rays (0° = rightward, CCW)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # ── Right: profile heatmap (horizontal = angle, vertical = distance from centre) ──
    ax = axes[1]

    n_full = profiles.shape[1]
    centre_idx = n_full // 2
    img_extent = [-centre_idx, centre_idx,
                  angles_deg[-1] + (360.0 / num_rays), angles_deg[0]]

    im = ax.imshow(profiles, aspect="auto", cmap="gray",
                   extent=img_extent, interpolation="nearest")
    ax.set_xlabel("Sample index (± from centre)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Radial intensity profiles (light = bright pixel)")

    # Vertical lines at the 4 canonical finder orientations
    for deg in np.arange(0, 360, 90):
        ax.axvline(deg, color=C_GT, linestyle="--", linewidth=1, alpha=0.4)

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

    # ── Clip ray endpoints to ROI boundary ──
    clipped_endpoints = ray_endpoints.copy()
    for i in range(NUM_RAYS):
        dx = np.cos(np.deg2rad(angles_deg[i]))
        dy = np.sin(np.deg2rad(angles_deg[i]))
        # Distance to each ROI edge from centre
        ts = []
        if abs(dx) > 1e-12:
            ts.append((0 - c_col) / dx)
            ts.append((W_roi - 1 - c_col) / dx)
        if abs(dy) > 1e-12:
            ts.append((0 - c_row) / dy)
            ts.append((H_roi - 1 - c_row) / dy)
        ts_pos = [t for t in ts if t > 0]
        ts_neg = [t for t in ts if t < 0]
        t_pos = min(ts_pos) if ts_pos else max_dist
        t_neg = max(ts_neg) if ts_neg else -max_dist
        max_dist = np.hypot(W_roi, H_roi) * 0.5
        if t_pos > max_dist:
            t_pos = max_dist
        if t_neg < -max_dist:
            t_neg = -max_dist
        clipped_endpoints[i] = [c_col + t_neg * dx, c_row + t_neg * dy,
                                c_col + t_pos * dx, c_row + t_pos * dy]

    # ── Inline plot (clipped rays + correct boundary lines) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Cluster {ci} — {NUM_RAYS} radial profiles  "
                 f"(m_est={m_est:.2f}px)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    cmap_rays = plt.cm.hsv
    for i in range(NUM_RAYS):
        ep = clipped_endpoints[i]
        ax.plot([ep[0], ep[2]], [ep[1], ep[3]],
                color=cmap_rays(i / NUM_RAYS), linewidth=1, alpha=0.6)
    ax.plot(c_col, c_row, "r+", markersize=12, markeredgewidth=2)
    ax.set_title("ROI + sample rays")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    ax = axes[1]
    n_full = profiles.shape[1]
    centre_idx = n_full // 2
    img_extent = [-centre_idx, centre_idx,
                  angles_deg[-1] + (360.0 / NUM_RAYS), angles_deg[0]]
    ax.imshow(profiles, aspect="auto", cmap="gray",
              extent=img_extent, interpolation="nearest")
    # Canonical boundary lines at ±1.5m, ±2.5m, ±3.5m
    for k, ls in [(1.5, "--"), (2.5, "--"), (3.5, "-")]:
        ax.axvline(+k * m_est, color=C_GT, linestyle=ls, linewidth=1, alpha=0.5)
        ax.axvline(-k * m_est, color=C_GT, linestyle=ls, linewidth=1, alpha=0.5)
    ax.axvline(0, color="white", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Sample index (± from centre)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Radial intensity profiles")

    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [5] Normalization + template fitting helpers
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
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


def normalize_roi_intensities(roi: np.ndarray, nbins: int = 64) -> tuple[np.ndarray, float, float]:
    """Normalize ROI intensities to [0, 1] using histogram bimodal fitting.

    Finds the two dominant peaks in the ROI intensity histogram, maps the
    darker peak to 0 and the brighter peak to 1.

    Returns
    -------
    roi_norm : ndarray (H, W)
        Normalized intensities in [0, 1].
    dark : float
        Intensity value mapped to 0.
    bright : float
        Intensity value mapped to 1.
    """
    vals = roi.ravel().astype(np.float64)
    hist, edges = np.histogram(vals, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Find peaks in the histogram (smoothed to avoid noise)
    peaks, props = find_peaks(hist, prominence=0.02 * hist.max())
    if len(peaks) < 2:
        # Fallback: p5 and p95
        dark = float(np.percentile(vals, 5))
        bright = float(np.percentile(vals, 95))
    else:
        # Sort peaks by height, take two tallest
        sorted_idx = np.argsort(hist[peaks])[::-1]
        p1 = peaks[sorted_idx[0]]
        p2 = peaks[sorted_idx[1]]
        dark = float(centers[min(p1, p2)])
        bright = float(centers[max(p1, p2)])

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit *m* independently to each half-ray in a profile stack.

    Returns
    -------
    m_pos, m_neg : ndarray (num_rays,)
    mse_pos, mse_neg : ndarray (num_rays,)
    success_pos, success_neg : ndarray (num_rays,) bool
    """
    n_rays = profiles.shape[0]
    n_full = profiles.shape[1]
    centre_idx = n_full // 2
    t_full = np.linspace(-max_dist, max_dist, n_full)

    # Split into positive and negative halves, both mapped to [0, max_dist]
    t_pos = t_full[centre_idx:]          # 0 … max_dist
    t_neg = -t_full[:centre_idx + 1][::-1]  # 0 … max_dist

    m_pos = np.full(n_rays, np.nan, dtype=np.float64)
    m_neg = np.full(n_rays, np.nan, dtype=np.float64)
    mse_pos = np.full(n_rays, np.nan, dtype=np.float64)
    mse_neg = np.full(n_rays, np.nan, dtype=np.float64)
    success_pos = np.full(n_rays, False)
    success_neg = np.full(n_rays, False)

    for i in range(n_rays):
        prof = profiles[i]
        res_pos = fit_m_half_ray(t_pos, prof[centre_idx:], m_est,
                                 mask_boundary, num_grid, grid_width, sigma)
        m_pos[i] = res_pos["m_fitted"]
        mse_pos[i] = res_pos["mse"]
        success_pos[i] = res_pos["success"]

        res_neg = fit_m_half_ray(t_neg, prof[:centre_idx + 1][::-1], m_est,
                                 mask_boundary, num_grid, grid_width, sigma)
        m_neg[i] = res_neg["m_fitted"]
        mse_neg[i] = res_neg["mse"]
        success_neg[i] = res_neg["success"]

    return m_pos, m_neg, mse_pos, mse_neg, success_pos, success_neg


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
    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi)

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
    m_pos, m_neg, mse_pos, mse_neg, success_pos, success_neg = fit_all_rays(
        profiles_norm, m_est, max_dist,
    )

    ok_pos = success_pos & np.isfinite(m_pos)
    ok_neg = success_neg & np.isfinite(m_neg)
    n_ok = int(np.sum(ok_pos)) + int(np.sum(ok_neg))
    n_total = 2 * NUM_RAYS
    m_all = np.concatenate([m_pos[ok_pos], m_neg[ok_neg]])
    m_median = float(np.median(m_all)) if len(m_all) > 0 else m_est
    print(f"\nCluster {ci}: m_est={m_est:.2f}px, m_fitted median={m_median:.2f}px  "
          f"({n_ok}/{n_total})")

    # ── Figure: heatmap + narrow strip above showing m per angle ──
    fig, ax_heatmap = plt.subplots(figsize=(11, 7))
    fig.suptitle(f"Cluster {ci} — m_est={m_est:.2f} → m_fit median={m_median:.2f}  "
                 f"(grey dashed = m_est)",
                 fontsize=12, fontweight="bold")

    # profiles_norm is (num_rays, n_full).  imshow: rows=rays→y, cols=index→x.
    n_full = profiles_norm.shape[1]
    centre_idx = n_full // 2
    # imshow: rows (rays=36) → y, cols (samples=239) → x
    # y maps to angle range (descending: 360 at top), x maps to sample index (centred)
    img_extent = [-centre_idx, centre_idx,
                  angles_deg[-1] + (360.0 / NUM_RAYS), angles_deg[0]]

    im = ax_heatmap.imshow(profiles_norm, aspect="auto", cmap="gray",
                           extent=img_extent, interpolation="nearest")
    ax_heatmap.set_xlabel("Sample index (± from centre)")
    ax_heatmap.set_ylabel("Angle (deg)")

    # Overlay fitted boundary positions (convert pixels → sample index)
    px_to_idx = centre_idx / max_dist
    # ── Inner boundary at ±1.5m (edge of 3×3 dark square) ──
    ax_heatmap.plot(+1.5 * m_pos * px_to_idx, angles_deg,
                    "-", color=C_E2, linewidth=1.5, alpha=0.5, label="1.5m (pos)")
    ax_heatmap.plot(-1.5 * m_neg * px_to_idx, angles_deg,
                    "-", color=C_E2, linewidth=1.5, alpha=0.5, label="1.5m (neg)")

    # ── Outer boundary at ±3.5m ──
    ax_heatmap.plot(+3.5 * m_pos * px_to_idx, angles_deg,
                    "o-", color=C_GOOD, markersize=3, linewidth=1.5, label="3.5m (pos)")
    ax_heatmap.plot(-3.5 * m_neg * px_to_idx, angles_deg,
                    "o-", color=C_GOOD, markersize=3, linewidth=1.5, label="3.5m (neg)")
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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi)
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
    m_pos, m_neg, _, _, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)

    # ── Compute 3.5m boundary points for each ray ──
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    boundary_points = []
    for i in range(NUM_RAYS):
        dx = np.cos(theta_rad[i])
        dy = np.sin(theta_rad[i])
        if np.isfinite(m_pos[i]):
            rp = PITCH_CONSTANT * m_pos[i]
            boundary_points.append(center_xy + rp * np.array([dx, dy]))
        if np.isfinite(m_neg[i]):
            rn = PITCH_CONSTANT * m_neg[i]
            boundary_points.append(center_xy - rn * np.array([dx, dy]))

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

    # Boundary points
    ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1],
               c=angles_deg.repeat(2), cmap="hsv", s=20, edgecolors="white",
               linewidths=0.3, zorder=3)

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

# %% [8] Phase 0 — Local TLS init + degeneracy cull
H_SMOOTH = 2
DEGENERACY_THRESHOLD = 0.3
MIN_POINTS = 18
PITCH_CONSTANT = 3.5


def compute_boundary_points(
    center_xy: np.ndarray,
    m_pos: np.ndarray,
    m_neg: np.ndarray,
    theta_rad: np.ndarray,
    pitch_constant: float = 3.5,
) -> np.ndarray:
    """1N cyclic boundary points: prefer m_pos, fall back to m_neg."""
    N = len(theta_rad)
    half_N = N // 2
    points = np.full((N, 2), np.nan, dtype=np.float64)
    for i in range(N):
        dx = float(np.cos(theta_rad[i]))
        dy = float(np.sin(theta_rad[i]))
        dir_vec = np.array([dx, dy])
        if np.isfinite(m_pos[i]):
            points[i] = center_xy + pitch_constant * m_pos[i] * dir_vec
        else:
            neg_idx = (i + half_N) % N
            if np.isfinite(m_neg[neg_idx]):
                points[i] = center_xy + pitch_constant * m_neg[neg_idx] * dir_vec
    return points


def tls_line(pts: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """TLS line fit via SVD.  Returns (normal, rho, dir_vec, singular_values)."""
    mu = pts.mean(axis=0)
    _, s, Vt = np.linalg.svd(pts - mu, full_matrices=False)
    n = Vt[1]
    d = Vt[0]
    r = float(mu @ n)
    if r < 0:
        n = -n
        r = -r
    return n, r, d, s


def perp_dist(points: np.ndarray, normal: np.ndarray, rho: float) -> np.ndarray:
    return np.abs(points @ normal - rho)


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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi)
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
    m_pos, m_neg, _, _, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)
    theta_deg = np.rad2deg(theta_rad)

    bp = compute_boundary_points(center_xy, m_pos, m_neg, theta_rad, PITCH_CONSTANT)
    N = len(bp)

    # ── Local TLS on 5-point windows ──
    ratios = np.full(N, np.nan, dtype=np.float64)
    init_normals = np.zeros((N, 2), dtype=np.float64)
    init_rhos = np.zeros(N, dtype=np.float64)
    init_dirs = np.zeros((N, 2), dtype=np.float64)
    init_support = [[] for _ in range(N)]

    for i in range(N):
        idx = [(i + k) % N for k in range(-H_SMOOTH, H_SMOOTH + 1)]
        pts = bp[idx]
        if len(pts) < 2:
            continue
        n, rho, d, s = tls_line(pts)
        init_normals[i] = n
        init_rhos[i] = rho
        init_dirs[i] = d
        init_support[i] = idx
        ratios[i] = s[1] / s[0] if s[0] > 1e-12 else np.nan

    # ── Cull degenerate points ──
    kept_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if np.isfinite(ratios[i]) and ratios[i] > DEGENERACY_THRESHOLD:
            kept_mask[i] = False
    n_kept = int(np.sum(kept_mask))
    if n_kept < MIN_POINTS:
        order = np.argsort(ratios)
        kept_mask[:] = False
        kept_mask[order[:MIN_POINTS]] = True
        n_kept = MIN_POINTS

    print(f"Cluster {ci}: kept {n_kept}/{N} points "
          f"(threshold={DEGENERACY_THRESHOLD})")

    # ── Diagnostic plot: angle vs σ₂/σ₁ ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(DEGENERACY_THRESHOLD, color="black", linestyle="--", linewidth=1,
               alpha=0.5, label=f"threshold={DEGENERACY_THRESHOLD}")
    for i in range(N):
        color = C_GOOD if kept_mask[i] else C_GT
        ax.plot(theta_deg[i], ratios[i], "o", color=color, markersize=6)
    ax.set_xlabel("Angle (deg CCW from +x)")
    ax.set_ylabel("σ₂/σ₁")
    ax.set_title(f"Phase 0 — Local TLS degeneracy ratios  (cluster {ci}, "
                 f"kept {n_kept}/{N})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(1.0, np.nanmax(ratios) * 1.15))
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [9] Phase 1 — Agglomerative clustering with merge-order plot
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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi)
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
    m_pos, m_neg, _, _, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)

    bp = compute_boundary_points(center_xy, m_pos, m_neg, theta_rad, PITCH_CONSTANT)
    N = len(bp)

    # ── Phase 0: local TLS + cull ──
    ratios = np.full(N, np.nan)
    for i in range(N):
        idx = [(i + k) % N for k in range(-H_SMOOTH, H_SMOOTH + 1)]
        pts = bp[idx]
        if len(pts) < 2:
            continue
        _, _, _, s = tls_line(pts)
        ratios[i] = s[1] / s[0] if s[0] > 1e-12 else np.nan

    kept_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if np.isfinite(ratios[i]) and ratios[i] > DEGENERACY_THRESHOLD:
            kept_mask[i] = False
    n_kept = int(np.sum(kept_mask))
    if n_kept < MIN_POINTS:
        order = np.argsort(ratios)
        kept_mask[:] = False
        kept_mask[order[:MIN_POINTS]] = True
        n_kept = MIN_POINTS

    kept_indices = [i for i in range(N) if kept_mask[i]]

    # ── Agglomerative clustering ──
    clusters_list: list[dict] = []
    for i in kept_indices:
        supp_idx = [(i + k) % N for k in range(-H_SMOOTH, H_SMOOTH + 1)]
        pts = bp[supp_idx]
        n, rho, d, _ = tls_line(pts)
        clusters_list.append({
            "start": i,
            "end": i,
            "support": supp_idx,
            "normal": n,
            "rho": rho,
            "dir": d,
        })

    absorb_step = np.full(N, -1, dtype=int)
    for cl in clusters_list:
        absorb_step[cl["start"]] = 0

    merge_step = 0
    while len(clusters_list) > 4:
        nc = len(clusters_list)
        if nc < 2:
            break

        distances = np.full(nc, np.inf)
        pair_indices = []
        for j in range(nc):
            j_next = (j + 1) % nc
            ca = clusters_list[j]
            cb = clusters_list[j_next]

            pts_a = bp[ca["support"]]
            pts_b = bp[cb["support"]]
            da = np.mean(perp_dist(pts_a, cb["normal"], cb["rho"]))
            db = np.mean(perp_dist(pts_b, ca["normal"], ca["rho"]))
            distances[j] = (da + db) / 2.0
            pair_indices.append((j, j_next))

        best_j = int(np.argmin(distances))
        j1, j2 = pair_indices[best_j]
        j_prev = (j1 - 1) % nc
        j_next_next = (j2 + 1) % nc

        ca = clusters_list[j1]
        cb = clusters_list[j2]

        new_support = sorted(set(ca["support"] + cb["support"]))
        pts = bp[new_support]
        n, rho, d, _ = tls_line(pts)

        new_cl = {
            "start": ca["start"],
            "end": cb["end"],
            "support": new_support,
            "normal": n,
            "rho": rho,
            "dir": d,
        }

        merge_step += 1
        for idx in ca["support"] + cb["support"]:
            absorb_step[idx] = merge_step

        if j1 < j2:
            del clusters_list[j2]
            del clusters_list[j1]
        else:
            del clusters_list[max(j1, j2)]
            del clusters_list[min(j1, j2)]

        clusters_list.insert(min(j1, j2), new_cl)

    # ── Diagnostic plot: merge order ──
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)

    valid = np.all(np.isfinite(bp), axis=1)
    max_step = merge_step if merge_step > 0 else 1
    for i in range(N):
        if valid[i]:
            if absorb_step[i] >= 0:
                c = plt.cm.plasma(absorb_step[i] / max_step)
            else:
                c = "gray"
            ax.plot(bp[i, 0], bp[i, 1], "o", color=c, markersize=6,
                    markeredgewidth=0.5, markeredgecolor="white")

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, max_step))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Merge step (dark=early, bright=late)")

    ax.set_title(f"Phase 1 — Agglomerative merge order  (cluster {ci}, "
                 f"{merge_step} merges)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [10] Phase 2 — Final 4-line assignment
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

    roi_norm, dark_val, bright_val = normalize_roi_intensities(roi)
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
    m_pos, m_neg, _, _, _, _ = fit_all_rays(profiles_norm, m_est, max_dist)
    theta_rad = np.linspace(0, 2 * np.pi, NUM_RAYS, endpoint=False)

    bp = compute_boundary_points(center_xy, m_pos, m_neg, theta_rad, PITCH_CONSTANT)
    N = len(bp)

    # ── Phase 0 + 1: cull + cluster ──
    ratios = np.full(N, np.nan)
    for i in range(N):
        idx = [(i + k) % N for k in range(-H_SMOOTH, H_SMOOTH + 1)]
        pts = bp[idx]
        if len(pts) < 2:
            continue
        _, _, _, s = tls_line(pts)
        ratios[i] = s[1] / s[0] if s[0] > 1e-12 else np.nan

    kept_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if np.isfinite(ratios[i]) and ratios[i] > DEGENERACY_THRESHOLD:
            kept_mask[i] = False
    n_kept = int(np.sum(kept_mask))
    if n_kept < MIN_POINTS:
        order = np.argsort(ratios)
        kept_mask[:] = False
        kept_mask[order[:MIN_POINTS]] = True
        n_kept = MIN_POINTS

    kept_indices = [i for i in range(N) if kept_mask[i]]

    clusters_list: list[dict] = []
    for i in kept_indices:
        supp_idx = [(i + k) % N for k in range(-H_SMOOTH, H_SMOOTH + 1)]
        pts = bp[supp_idx]
        n, rho, d, _ = tls_line(pts)
        clusters_list.append({
            "start": i, "end": i,
            "support": supp_idx,
            "normal": n, "rho": rho, "dir": d,
        })

    while len(clusters_list) > 4 and len(clusters_list) >= 2:
        nc = len(clusters_list)
        distances = np.full(nc, np.inf)
        pair_indices = []
        for j in range(nc):
            j_next = (j + 1) % nc
            ca = clusters_list[j]
            cb = clusters_list[j_next]
            pts_a = bp[ca["support"]]
            pts_b = bp[cb["support"]]
            da = np.mean(perp_dist(pts_a, cb["normal"], cb["rho"]))
            db = np.mean(perp_dist(pts_b, ca["normal"], ca["rho"]))
            distances[j] = (da + db) / 2.0
            pair_indices.append((j, j_next))

        best_j = int(np.argmin(distances))
        j1, j2 = pair_indices[best_j]
        ca = clusters_list[j1]
        cb = clusters_list[j2]

        new_support = sorted(set(ca["support"] + cb["support"]))
        pts_new = bp[new_support]
        n, rho, d, _ = tls_line(pts_new)
        new_cl = {
            "start": ca["start"], "end": cb["end"],
            "support": new_support,
            "normal": n, "rho": rho, "dir": d,
        }

        if j1 < j2:
            del clusters_list[j2]
            del clusters_list[j1]
        else:
            del clusters_list[max(j1, j2)]
            del clusters_list[min(j1, j2)]
        clusters_list.insert(min(j1, j2), new_cl)

    # ── Diagnostic plot: final 4 lines ──
    seg_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy[0], center_xy[1], "r+", markersize=12, markeredgewidth=2)

    if len(clusters_list) == 4:
        for seg_idx, cl in enumerate(clusters_list):
            pts_seg = bp[cl["support"]]
            ax.scatter(pts_seg[:, 0], pts_seg[:, 1],
                       color=seg_colors[seg_idx], s=30, edgecolors="white",
                       linewidths=0.5, zorder=3, label=f"Edge {seg_idx}")

            n, rho, d = cl["normal"], cl["rho"], cl["dir"]
            proj = pts_seg @ n
            lo, hi = float(proj.min()), float(proj.max())
            ext = (hi - lo) * 0.2
            t_vals = np.array([lo - ext, hi + ext])
            line_pts = rho * n + t_vals[:, None] * d
            ax.plot(line_pts[:, 0], line_pts[:, 1],
                    color=seg_colors[seg_idx], linewidth=2.5, alpha=0.9)
    else:
        valid_bp = np.all(np.isfinite(bp), axis=1)
        if np.any(valid_bp & kept_mask):
            ax.scatter(bp[valid_bp & kept_mask, 0], bp[valid_bp & kept_mask, 1],
                       color="gray", s=20, zorder=3)

    ax.set_title(f"Phase 2 — Initial 4-line assignment  (cluster {ci})")
    ax.legend(fontsize=8)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()


