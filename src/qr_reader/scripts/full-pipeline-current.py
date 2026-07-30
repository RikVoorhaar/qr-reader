
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
PRESET = "easy"  # 'easy', 'medium', or 'hard'
VERSION = 5  # QR version (1–40)
SAMPLE_SEED = 42  # Base seed for the random generator
TOP_N = 5  # Number of top clusters to visualize in the per-cluster finder fitting
TIGHT_LAYOUT = True  # Apply plt.tight_layout() before each plt.show()

# Colours
C_GOOD = "#2ca02c"       # green  — detected / fitted
C_GT = "#d62728"         # red    — ground truth
C_INTERMEDIATE = "#1f77b4"  # blue   — intermediate estimates
C_E1 = "#17becf"         # cyan   — e1 axis
C_E2 = "#ff7f0e"         # orange — e2 axis
C_CLUSTER_BG = "#7f7f7f"  # gray   — cluster bounding boxes

# %% [1] Imports
from qr_reader.decoder.decoder import DecodeError, decode
from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
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
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.detector.sample import sample_qr_bits
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

# %% [2] Generate composited QR image
rng = np.random.default_rng(SAMPLE_SEED)

preset_name = PRESET.lower()
if preset_name not in PRESET_MAP:
    print(f"Unknown preset '{PRESET}', using 'medium'")
    preset_name = "medium"

config = AugmentationConfig(
    version=VERSION,
    content=f"QR v{VERSION} — pipeline test",
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

import cv2

# Load a real background image from the HomeObjects dataset
BG_DIR = Path("data/images/train")
if BG_DIR.is_dir():
    bg_paths = sorted(BG_DIR.glob("*.jpg"))
    if bg_paths:
        bg_path = bg_paths[SAMPLE_SEED % len(bg_paths)]
        background = cv2.imread(str(bg_path))
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        print(f"Background: {bg_path.name} ({background.shape[1]}×{background.shape[0]})")
    else:
        raise FileNotFoundError(f"No .jpg files in {BG_DIR}")
else:
    raise FileNotFoundError(f"Background directory not found: {BG_DIR}")

image, metadata = generate_sample(rng, config, background)
img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)

QR_VERSION = metadata["version"]
QR_CONTENT = metadata["payload"]
print(f"Generated v{QR_VERSION} QR with content: '{QR_CONTENT}'")

# ── Extract ground truth corners (x, y) ──
gt_corners = np.array([
    metadata["corners_qr"]["TL"],
    metadata["corners_qr"]["TR"],
    metadata["corners_qr"]["BR"],
    metadata["corners_qr"]["BL"],
    metadata["corners_qr"]["TL"],  # close loop
], dtype=np.float64)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap="gray")
ax.plot(gt_corners[:, 0], gt_corners[:, 1], color=C_GT, linewidth=2, label="GT corners")
for i, label in enumerate(["TL", "TR", "BR", "BL"]):
    ax.text(gt_corners[i, 0] + 3, gt_corners[i, 1] + 3, label, color=C_GT, fontsize=8, weight="bold")
ax.set_title(f"Input image — v{QR_VERSION} ({PRESET.upper()})")
ax.legend(fontsize=8)
ax.axis("off")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()

# %% [3] Binarization
img_binary = binarize_image(img_gray)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_gray, cmap="gray")
ax1.set_title("Grayscale")
ax1.axis("off")
ax2.imshow(img_binary, cmap="gray")
ax2.set_title(f"Otsu binarized (black={np.sum(~img_binary):,}, white={np.sum(img_binary):,})")
ax2.axis("off")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()

# %% [4] Alignment pattern scan (2-D cross-validated)
max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
print(f"2-D validated candidates: {len(rows_valid)}")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_binary, cmap="gray")

# Draw all 2-D validated candidates as coloured horizontal segments
if len(rows_valid) > 0:
    seg_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i in range(len(rows_valid)):
        r = rows_valid[i]
        c = cols_valid_all[i]
        for k in range(5):
            ax.hlines(float(r), float(c[k]), float(c[k + 1]),
                      colors=seg_palette[k], linewidth=3, alpha=0.7)
    ax.set_title(f"2-D validated alignment pattern candidates ({len(rows_valid)})")
else:
    ax.set_title("No 2-D validated candidates found")
ax.axis("off")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()

# %% [5] Clustering
clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"Clusters: {len(clusters)}")

# Sort by width (descending) for visual order
sorted_clusters = sorted(enumerate(clusters),
                         key=lambda x: float(x[1].cols[5] - x[1].cols[0]),
                         reverse=True)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")

cmap = plt.cm.tab20
for rank, (_, cluster) in enumerate(sorted_clusters):
    color = cmap(rank % 20)
    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0
    c_row = float(cluster.row)
    width = float(cluster.cols[5] - cluster.cols[0])
    height = float(cluster.height)

    rect = plt.Rectangle(
        (cluster.cols[0], cluster.row - height / 2),
        width, height,
        fill=False, edgecolor=color, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.plot(c_col, c_row, "o", color=color, markersize=5)
    ax.text(c_col + 3, c_row + 3, str(rank), color=color, fontsize=7, weight="bold")

ax.set_title(f"{len(clusters)} clustered regions")
ax.axis("off")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()

# %% [6] Per-cluster finder fitting (top-N by width)
# TODO: Reimplement using fit_finder_ray from ray_fit.py
# The old implementation using finder_fit was deleted.
