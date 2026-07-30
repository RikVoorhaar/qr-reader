"""Interactive visual diagnostic for composited QR pipeline failures.

Usage:
    python diagnose_visual.py V=1 S=02    # specific case
    python diagnose_visual.py             # shows first interesting cases
"""
# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.detector import _run_detection, _score_timing_pattern
from qr_reader.detector.finder_pattern import FinderPattern, find_valid_triplets
from qr_reader.detector.homography import (
    compute_qr_corners,
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)
from qr_reader.detector.ray_fit import fit_finder_ray
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

# %%
BG_DIR = Path("data/images/train")

PRESET = "medium"
VERSION = 1
SEED = 0  # 0, 1, 5, 12, 13, 14, 16, 18 all have 3-4 finders but no triplet at V=1

# %%
bg_paths = sorted(BG_DIR.glob("*.jpg"))
base_cfg = PRESET_MAP[PRESET]
config = AugmentationConfig(**base_cfg.__dict__)
config.version = VERSION
config.content = f"v{VERSION}"
config.error_correction = "M"
config.global_seed = SEED

rng = np.random.default_rng(SEED)
bg_path = bg_paths[SEED % len(bg_paths)]
bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
img_rgb, meta = generate_sample(
    rng, config, bg, sample_index=0, background_path=str(bg_path),
)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
h, w = gray.shape

gt_corners = np.array([
    meta["corners_qr"]["TL"], meta["corners_qr"]["TR"],
    meta["corners_qr"]["BR"], meta["corners_qr"]["BL"],
], dtype=np.float64)

# ── Alignment scan + clustering ──
img_binary = binarize_image(gray)
rows_v, cols_v_all = find_alignment_patterns_2d(img_binary, np.log(1.3))
clusters = cluster_candidates(rows_v, cols_v_all)
print(f"{len(rows_v)} RLE candidates → {len(clusters)} clusters")

# ── Per-cluster finder fitting ──
fps = []; score_map = {}; global_corners_xy = {}
for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(gray, bbox)
    if roi.size == 0: continue
    r0 = max(0, int(bbox[0])); c0 = max(0, int(bbox[2]))
    cx = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
    cy = float(cluster.row) - r0
    m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0
    result = fit_finder_ray(roi, np.array([cx, cy]), m_est)
    if not result.valid: continue
    cxy = result.corners + np.array([c0, r0], dtype=np.float64)
    wh = np.ptp(cxy, axis=0)
    if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est: continue
    fps.append(FinderPattern(cluster_idx=ci, outer_corners=cxy[:, ::-1]))
    score_map[ci] = result.score
    global_corners_xy[ci] = cxy

print(f"  {len(fps)} finders fitted")

# ── Dedup ──
keep_mask = np.ones(len(fps), dtype=bool)
for i in range(len(fps)):
    if not keep_mask[i]: continue
    ci = fps[i].outer_corners.mean(axis=0)
    seg_i = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
    for j in range(i + 1, len(fps)):
        if not keep_mask[j]: continue
        cj = fps[j].outer_corners.mean(axis=0)
        seg_j = float(np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1]))
        if float(np.linalg.norm(ci - cj)) < 1.0 * min(seg_i, seg_j):
            if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                keep_mask[j] = False
            else:
                keep_mask[i] = False; break
fps = [fp for fp, keep in zip(fps, keep_mask) if keep]
print(f"  {len(fps)} after dedup")

# ── Triplets ──
triplets = find_valid_triplets(fps, score_map)
print(f"  {len(triplets)} triplets")

# ── Try running full detection ──
try:
    H, dv = _run_detection(gray)
    print(f"  Full detection: V={dv}")
    det_ok = True
except Exception as e:
    print(f"  Full detection FAILED: {e}")
    dv = VERSION
    det_ok = False

# %%
# Plot: image with all finder fits overlaid, GT corners, clusters
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_rgb)

# GT corners
gt_poly = np.vstack([gt_corners, gt_corners[:1]])
ax.plot(gt_poly[:, 0], gt_poly[:, 1], "-", color="#d62728", linewidth=2, label="GT QR")

# Cluster bboxes
import matplotlib.patches as patches
for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    r0, r1, c0, c1 = bbox
    rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0,
                              fill=False, edgecolor="cyan", linewidth=0.5, alpha=0.4)
    ax.add_patch(rect)

# Fitted finders
colors = plt.cm.tab10(np.linspace(0, 1, len(fps)))
for i, fp in enumerate(fps):
    corners_xy = fp.outer_corners[:, ::-1]
    poly = np.vstack([corners_xy, corners_xy[:1]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color=colors[i], linewidth=2,
            label=f"FP {i} (score={score_map[fp.cluster_idx]:.2f})")
    ax.plot(corners_xy[0, 0], corners_xy[0, 1], "o", color=colors[i], markersize=8)

# If detection succeeded, plot detected QR
if det_ok:
    N = 4 * dv + 17
    det_corners = compute_qr_corners(H, N)
    det_poly = np.vstack([det_corners, det_corners[:1]])
    ax.plot(det_poly[:, 0], det_poly[:, 1], "-", color="#2ca02c", linewidth=2,
            label=f"Detected V={dv}")

# RLE candidate positions
for row, col_val in zip(rows_v, cols_v_all):
    ax.plot(col_val, row, ".", color="yellow", markersize=3, alpha=0.5)

ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"{PRESET} V={VERSION} seed={SEED}  bg={bg_path.name}  "
             f"{len(fps)} finders → {len(triplets)} triplets")
ax.axis("off")

# %%
# Show a few ROI cutouts with their finder fits
n_show = min(len(clusters), 4)
if n_show > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    for idx in range(4):
        ax = axes[idx]
        if idx >= len(clusters):
            ax.axis("off"); continue
        cluster = clusters[idx]
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        r0 = max(0, int(bbox[0])); c0 = max(0, int(bbox[2]))
        if roi.size == 0:
            ax.axis("off"); continue
        ax.imshow(roi, cmap="gray")

        # Check if this cluster has a fitted finder
        for fp in fps:
            if fp.cluster_idx == idx:
                corners_rc = fp.outer_corners
                poly_rc = np.vstack([corners_rc, corners_rc[:1]])
                poly_xy = poly_rc[:, ::-1] - np.array([c0, r0])
                ax.plot(poly_xy[:, 0], poly_xy[:, 1], "r-", linewidth=2)
                break

        ax.set_title(f"Cluster {idx}")
        ax.axis("off")

plt.show()
