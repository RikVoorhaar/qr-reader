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

        profiles[i] = ((1 - fy) * ((1 - fx) * roi[y0, x0] + fx * roi[y0, x1])
                       + fy * ((1 - fx) * roi[y1, x0] + fx * roi[y1, x1]))

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
    # profiles is (num_rays, n_full).  imshow: row=ray → y, col=distance → x.
    img_extent = [angles_deg[0], angles_deg[-1] + (360.0 / num_rays),
                  centre_idx, -centre_idx]

    im = ax.imshow(profiles, aspect="auto", cmap="gray",
                   extent=img_extent, interpolation="nearest")
    ax.set_xlabel("Angle (deg, 0° = rightward, CCW)")
    ax.set_ylabel("Sample index (± from centre)")
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

    plot_ray_profiles(roi, center_xy, profiles, ray_endpoints,
                      angles_deg, m_est, ci)
