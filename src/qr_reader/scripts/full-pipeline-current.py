
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
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.edges import extract_thin_edges  # deprecated — kept for backward compat in display sections
# finder_fit module removed; use ray_fit.fit_finder_ray instead
try:
    from qr_reader.detector.finder_fit import (
        FinderFit,
        build_projection_profile,
        estimate_orientation,
        fit_finder_1d,
        fit_finder_full,
        fit_scanline_projective,
        estimate_m_from_edges,
        refine_outer_line,
        _sample_1d_cross_section,
    )
except ImportError:
    FinderFit = None  # type: ignore
    build_projection_profile = None  # type: ignore
    estimate_orientation = None  # type: ignore
    fit_finder_1d = None  # type: ignore
    fit_finder_full = None  # type: ignore
    fit_scanline_projective = None  # type: ignore
    estimate_m_from_edges = None  # type: ignore
    refine_outer_line = None  # type: ignore
    _sample_1d_cross_section = None  # type: ignore
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

# Clone the pipeline logic from detector.py
fps: list[FinderPattern] = []
fit_map: dict[int, FinderFit] = {}
global_corners_xy: dict[int, np.ndarray] = {}

# We'll build our own list, indexed the same way as detector.py
for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    r0_orig, r1_orig, c0_orig, c1_orig = bbox
    roi = cutout(img_gray, bbox)
    if roi.size == 0:
        continue

    r0 = max(0, r0_orig)
    c0 = max(0, c0_orig)

    nms, angle_npy = extract_thin_edges(roi, blur_sigma=1.0)
    if nms.size == 0 or np.count_nonzero(nms) == 0:
        continue

    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
    c_row = float(cluster.row) - r0
    center_xy = np.array([c_col, c_row], dtype=np.float64)
    m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

    fit = fit_finder_full(nms, angle_npy, roi, center_xy, m_est)

    corners_xy_global = fit.corners + np.array([c0, r0], dtype=np.float64)
    corners_rc = corners_xy_global[:, ::-1]
    fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
    fit_map[ci] = fit
    global_corners_xy[ci] = corners_xy_global


def _draw_infinite_line(normal, rho, H, W):
    nx, ny = normal
    eps = 1e-9
    points = []
    if abs(ny) > eps:
        y0 = rho / ny
        if 0 <= y0 < H:
            points.append((0.0, y0))
    if abs(ny) > eps:
        yw = (rho - nx * (W - 1)) / ny
        if 0 <= yw < H:
            points.append((float(W - 1), yw))
    if abs(nx) > eps:
        x0 = rho / nx
        if 0 <= x0 < W:
            points.append((x0, 0.0))
    if abs(nx) > eps:
        xh = (rho - ny * (H - 1)) / nx
        if 0 <= xh < W:
            points.append((xh, float(H - 1)))
    if len(points) < 2:
        return np.array([[0, 0], [0, 0]])
    return np.array(points[:2])


# Pick top-N clusters with valid fits (non-zero m) for detailed visualization
valid_clusters = [
    (ci, cluster) for ci, cluster in enumerate(clusters)
    if ci in fit_map and fit_map[ci].m > 0
]
# Sort by module pitch (larger = more prominent finder)
valid_clusters.sort(key=lambda x: fit_map[x[0]].m, reverse=True)

top_n = min(TOP_N, len(valid_clusters))
print(f"Visualizing top {top_n} clusters out of {len(valid_clusters)} with valid fits")

for rank in range(top_n):
    ci, cluster = valid_clusters[rank]
    fit = fit_map[ci]

    bbox = cluster_to_bbox(cluster, scale=1.5)
    r0_orig, r1_orig, c0_orig, c1_orig = bbox
    roi = cutout(img_gray, bbox)
    r0 = max(0, r0_orig)
    c0 = max(0, c0_orig)
    H_roi, W_roi = roi.shape

    nms, angle_npy = extract_thin_edges(roi, blur_sigma=1.0)

    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
    c_row = float(cluster.row) - r0
    center_xy_input = np.array([c_col, c_row], dtype=np.float64)
    m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

    # Re-run the intermediate phases for richer diagnostics
    phi_diag, e1_diag, e2_diag = estimate_orientation(nms, angle_npy, center_xy_input)
    phi_deg = np.rad2deg(phi_diag)

    m_edge = estimate_m_from_edges(nms, angle_npy, center_xy_input, e1_diag, e2_diag)
    m_init = max(m_est, m_edge)

    pos_u, prof_u = build_projection_profile(nms, angle_npy, center_xy_input, e1_diag, m_init)
    pos_v, prof_v = build_projection_profile(nms, angle_npy, center_xy_input, e2_diag, m_init)

    aff_u = fit_finder_1d(prof_u, pos_u, m_init)
    aff_v = fit_finder_1d(prof_v, pos_v, m_init)

    proj_u = fit_scanline_projective(
        nms, angle_npy, center_xy_input, e1_diag, m_init,
        m_seed=float(aff_u["m_fitted"]),
        du_seed=float(aff_u["center_offset"]),
    )
    proj_v = fit_scanline_projective(
        nms, angle_npy, center_xy_input, e2_diag, m_init,
        m_seed=float(aff_v["m_fitted"]),
        du_seed=float(aff_v["center_offset"]),
    )

    m_fit = fit.m
    fitted_center = fit.center
    corners_final = fit.corners  # in ROI-local (x, y)

    # ── Print diagnostics ──
    width_px = float(cluster.cols[5] - cluster.cols[0])
    print(f"\n  Cluster {ci} (rank {rank+1}/{top_n}): "
          f"width={width_px:.1f}px, m_est={m_est:.2f}px, m_fit={m_fit:.2f}px, "
          f"score={fit.score:.3f}, phi={phi_deg:.1f}°")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure A: Orientation estimation
    # ══════════════════════════════════════════════════════════════════════════
    figA, axesA = plt.subplots(2, 2, figsize=(12, 10))
    figA.suptitle(f"Cluster {ci} — Orientation Estimation  (φ={phi_deg:.1f}°)",
                  fontsize=13, fontweight="bold")

    # A1 — Grayscale cutout
    ax = axesA[0, 0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    ax.plot(center_xy_input[0], center_xy_input[1], "rx", markersize=8)
    ax.set_title("ROI grayscale cutout")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # A2 — NMS edge map
    ax = axesA[0, 1]
    ax.imshow(nms, cmap="hot", extent=[0, W_roi, H_roi, 0])
    ax.set_title(f"NMS thin edges ({np.count_nonzero(nms)} nonzero)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # A3 — 4-fold orientation histogram
    ax = axesA[1, 0]
    ys_nz, xs_nz = np.nonzero(nms)
    if len(ys_nz) > 0:
        strengths = nms[ys_nz, xs_nz]
        alpha_mod = np.fmod(angle_npy[ys_nz, xs_nz], np.pi)
        alpha_mod = np.where(alpha_mod < 0, alpha_mod + np.pi, alpha_mod)
        alpha_mod = np.fmod(alpha_mod, np.pi / 2.0)
        ax.hist(np.rad2deg(alpha_mod), bins=45, range=(0, 90),
                weights=strengths, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(phi_deg, color=C_GOOD, linestyle="-", linewidth=2,
               label=f"φ = {phi_deg:.1f}°")
    ax.set_xlabel("Angle mod π/2 (deg)")
    ax.set_ylabel("Weighted count")
    ax.set_title("4-fold orientation histogram")
    ax.legend(fontsize=9)

    # A4 — Estimated axes overlaid on ROI
    ax = axesA[1, 1]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    arrow_len = 3.5 * m_fit
    deg_90 = np.pi / 2
    for k, (axis, color, label) in enumerate([
        (e1_diag, C_E1, "e1"), (e2_diag, C_E2, "e2"),
        # Also show the 90°-rotated families (4-fold ambiguity resolved)
    ]):
        ax.arrow(center_xy_input[0], center_xy_input[1],
                 arrow_len * axis[0], arrow_len * axis[1],
                 color=color, width=2, head_width=5, alpha=0.8, label=label)
        # Also show opposite directions (the 180° partners)
        ax.arrow(center_xy_input[0], center_xy_input[1],
                 -arrow_len * axis[0], -arrow_len * axis[1],
                 color=color, width=1, head_width=3, alpha=0.3)
    ax.plot(center_xy_input[0], center_xy_input[1], "r+", markersize=10)
    ax.set_title("Estimated axes (C=e1, O=e2) ± both directions")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    if TIGHT_LAYOUT:
        figA.tight_layout()
    plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # Figure B: Profile fitting + corner extraction
    # ══════════════════════════════════════════════════════════════════════════
    figB, axesB = plt.subplots(2, 2, figsize=(14, 10))
    figB.suptitle(f"Cluster {ci} — Profile Fitting & Corner Extraction  "
                  f"(m={m_fit:.2f}px)",
                  fontsize=13, fontweight="bold")

    # B1 — e1 profile with affine peaks + projective offsets
    ax = axesB[0, 0]
    if len(pos_u) > 0 and len(prof_u) > 0:
        bw = (pos_u[1] - pos_u[0]) if len(pos_u) > 1 else m_est / 4
        ax.bar(pos_u, prof_u, width=bw, color="steelblue", alpha=0.7)
    # Affine peaks (green dashed)
    if aff_u["profile_values"] is not None and len(aff_u["peak_positions"]) > 0:
        for p in aff_u["peak_positions"]:
            ax.axvline(p, color=C_GOOD, linestyle="--", linewidth=1.5, alpha=0.6)
    # Projective fitted offsets (green solid)
    if proj_u["projective_params"] is not None:
        for offset in proj_u["fitted_offsets"]:
            if np.isfinite(offset):
                ax.axvline(offset, color=C_GOOD, linestyle="-", linewidth=2, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Position along e1 (px)")
    ax.set_ylabel("Edge strength")
    ax.set_title(f"Profile along e1  (affine m={aff_u['m_fitted']:.2f}px, proj m={proj_u['m_effective']:.2f}px)")

    # B2 — e2 profile with affine peaks + projective offsets
    ax = axesB[0, 1]
    if len(pos_v) > 0 and len(prof_v) > 0:
        bw = (pos_v[1] - pos_v[0]) if len(pos_v) > 1 else m_est / 4
        ax.bar(pos_v, prof_v, width=bw, color="steelblue", alpha=0.7)
    if aff_v["profile_values"] is not None and len(aff_v["peak_positions"]) > 0:
        for p in aff_v["peak_positions"]:
            ax.axvline(p, color=C_GOOD, linestyle="--", linewidth=1.5, alpha=0.6)
    if proj_v["projective_params"] is not None:
        for offset in proj_v["fitted_offsets"]:
            if np.isfinite(offset):
                ax.axvline(offset, color=C_GOOD, linestyle="-", linewidth=2, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Position along e2 (px)")
    ax.set_ylabel("Edge strength")
    ax.set_title(f"Profile along e2  (affine m={aff_v['m_fitted']:.2f}px, proj m={proj_v['m_effective']:.2f}px)")

    # B3 — Refined outer lines + NMS edges
    ax = axesB[1, 0]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    # Show NMS edges faintly
    nms_mask = nms > 0
    ys_nz, xs_nz = np.nonzero(nms_mask)
    if len(ys_nz) > 0:
        ax.scatter(xs_nz, ys_nz, s=1, c="white", alpha=0.3, marker=".")
    # Outer lines
    colors = {"u+": C_E1, "u-": C_E1, "v+": C_E2, "v-": C_E2}
    for label, (normal, rho) in fit.outer_lines.items():
        pts = _draw_infinite_line(normal, rho, H_roi, W_roi)
        ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]],
                color=colors.get(label, "white"), linewidth=2, alpha=0.9)
    # Fitted center
    ax.plot(fitted_center[0], fitted_center[1], "r+", markersize=12, markeredgewidth=2)
    ax.set_title("Refined outer lines + NMS edges")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # B4 — Final fitted corners on ROI
    ax = axesB[1, 1]
    ax.imshow(roi, cmap="gray", extent=[0, W_roi, H_roi, 0])
    idx_loop = [0, 1, 2, 3, 0]
    ax.plot(corners_final[idx_loop, 0], corners_final[idx_loop, 1],
            color=C_GOOD, linewidth=2, marker="o", markersize=5, label="Fitted")
    ax.plot(fitted_center[0], fitted_center[1], "r+", markersize=12)
    # Label corners
    corner_labels = ["(-,-)", "(+,-)", "(+,+)", "(-,+)"]
    for k in range(4):
        ax.text(corners_final[k, 0] + 2, corners_final[k, 1] + 2,
                corner_labels[k], color=C_GOOD, fontsize=7)
    ax.set_title("Fitted finder corners")
    ax.legend(fontsize=8)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    if TIGHT_LAYOUT:
        figB.tight_layout()
    plt.show()

print(f"\nTotal: {len(fps)} finder patterns fitted")

# %% [7] Deduplication
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
            if fit_map[fps[i].cluster_idx].score >= fit_map[fps[j].cluster_idx].score:
                keep_mask[j] = False
            else:
                keep_mask[i] = False
                break
fps_dedup = [fp for fp, keep in zip(fps, keep_mask) if keep]

dedup_map: dict[int, int] = {}
for i, fp in enumerate(fps_dedup):
    dedup_map[fp.cluster_idx] = i

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")

cmap = plt.cm.tab10
# Draw all finders (faded if removed)
for i, fp in enumerate(fps):
    ci = fp.cluster_idx
    corners_rc = fp.outer_corners
    corners_xy = np.column_stack([corners_rc[:, 1], corners_rc[:, 0]])
    idx_loop = [0, 1, 2, 3, 0]
    if ci in dedup_map:
        color = cmap(dedup_map[ci] % 10)
        lw, alpha_ = 2.5, 0.9
        label = f"Finder {dedup_map[ci]}"
    else:
        color = C_CLUSTER_BG
        lw, alpha_ = 1.0, 0.3
        label = None
    ax.plot(corners_xy[idx_loop, 0], corners_xy[idx_loop, 1],
            color=color, linewidth=lw, alpha=alpha_, label=label)
    if ci in dedup_map:
        ctr = corners_xy.mean(axis=0)
        ax.text(ctr[0], ctr[1], str(dedup_map[ci]),
                color=color, fontsize=8, weight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle", facecolor="white", alpha=0.7))
ax.set_title(f"Finder patterns: {len(fps_dedup)} kept / {len(fps)} total  "
             f"({len(fps) - len(fps_dedup)} removed)")
ax.axis("off")
# Deduplicate handles for legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower right")
if TIGHT_LAYOUT:
    plt.tight_layout()
plt.show()

# %% [8] Triplet finding + version estimation
best_H = None  # will be set by homography cell if successful
raw_triplets = find_valid_triplets(fps_dedup, fit_map)
if not raw_triplets:
    print("No triplet found!")
else:
    raw = raw_triplets[0]
    tl_idx = raw.top_left_idx
    tr_idx = raw.top_right_idx
    bl_idx = raw.bottom_left_idx

    fp_map = {fp.cluster_idx: fp for fp in fps_dedup}
    rows = {idx: float(fp_map[idx].outer_corners.mean(axis=0)[0]) for idx in [tl_idx, tr_idx, bl_idx]}
    cols = {idx: float(fp_map[idx].outer_corners.mean(axis=0)[1]) for idx in [tl_idx, tr_idx, bl_idx]}

    center_tl_xy = np.array([cols[tl_idx], rows[tl_idx]], dtype=np.float64)
    c_tr = np.array([cols[tr_idx], rows[tr_idx]], dtype=np.float64)
    c_bl = np.array([cols[bl_idx], rows[bl_idx]], dtype=np.float64)

    m_avg = (fit_map[tl_idx].m + fit_map[tr_idx].m + fit_map[bl_idx].m) / 3.0
    dx = float(np.linalg.norm(c_tr - center_tl_xy))
    dy = float(np.linalg.norm(c_bl - center_tl_xy))
    dh = float(np.linalg.norm(c_tr - c_bl))
    s_hat = (dx + dy + dh / np.sqrt(2)) / (3.0 * m_avg)
    N_est = int(round(s_hat + 7))
    N_legal = ((N_est - 17) // 4) * 4 + 21
    N_legal = max(21, min(177, N_legal))
    V_est = (N_legal - 17) // 4

    print(f"Triplet: TL={tl_idx}, TR={tr_idx}, BL={bl_idx}")
    print(f"  m_avg={m_avg:.2f}px, dx={dx:.1f}px, dy={dy:.1f}px, dh={dh:.1f}px")
    print(f"  N_est={N_est}, N_legal={N_legal}, V_est={V_est} (true={QR_VERSION})")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_gray, cmap="gray")

    for idx, label, color in [(tl_idx, "TL", C_GOOD), (tr_idx, "TR", C_E1), (bl_idx, "BL", C_E2)]:
        corners_rc = fp_map[idx].outer_corners
        corners_xy = np.column_stack([corners_rc[:, 1], corners_rc[:, 0]])
        idx_loop = [0, 1, 2, 3, 0]
        ax.plot(corners_xy[idx_loop, 0], corners_xy[idx_loop, 1],
                color=color, linewidth=2.5, label=f"{label} (cluster {idx})")
        ctr = corners_xy.mean(axis=0)
        ax.text(ctr[0] + 8, ctr[1], label, color=color, fontsize=10, weight="bold")

    # Inter-finder lines
    ax.plot([center_tl_xy[0], c_tr[0]], [center_tl_xy[1], c_tr[1]],
            "w--", linewidth=1, alpha=0.5)
    ax.plot([center_tl_xy[0], c_bl[0]], [center_tl_xy[1], c_bl[1]],
            "w--", linewidth=1, alpha=0.5)

    ax.set_title(f"Triplet: {len(raw_triplets)} candidates, "
                 f"selected v{V_est} (true v{QR_VERSION})")
    ax.legend(fontsize=8)
    ax.axis("off")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [9] Global homography + version search
if raw_triplets:
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

    grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64)
    tl_c = _canonicalize_corners(global_corners_xy[tl_idx])
    tr_c = _canonicalize_corners(global_corners_xy[tr_idx])
    bl_c = _canonicalize_corners(global_corners_xy[bl_idx])

    best_err = np.inf
    best_H = None
    best_N = N_legal
    results: list[dict] = []

    for N_cand in range(max(21, N_legal - 4), min(181, N_legal + 5), 4):
        src_xy = []
        dst_xy = []
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
        reproj_err = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))

        if reproj_err < best_err:
            best_err = reproj_err
            best_H = H
            best_N = N_cand

        results.append({
            "N": N_cand,
            "V": (N_cand - 17) // 4,
            "reproj_err": reproj_err,
        })

    if best_H is not None:
        V_best = (best_N - 17) // 4
        print(f"\nBest: N={best_N}, V={V_best}, reproj_err={best_err:.2f}px")
        print(f"  Candidates: {', '.join(f'N{r['N']}(V{r['V']})={r['reproj_err']:.1f}px' for r in results)}")

        qr_corners = compute_qr_corners(best_H, best_N)
        qr_corners_closed = np.vstack([qr_corners, qr_corners[0]])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_gray, cmap="gray")

        # Draw GT
        ax.plot(gt_corners[:, 0], gt_corners[:, 1],
                color=C_GT, linestyle="--", linewidth=2, label="GT")

        # Draw estimated QR boundary
        ax.plot(qr_corners_closed[:, 0], qr_corners_closed[:, 1],
                color=C_GOOD, linewidth=2, label=f"Estimated v{V_best}")

        # Draw finder correspondences as circles
        for corners, origin in [
            (tl_c, (0, 0)),
            (tr_c, (best_N - 7, 0)),
            (bl_c, (0, best_N - 7)),
        ]:
            ax.plot(corners[:, 0], corners[:, 1], "o", color=C_GOOD, markersize=4, alpha=0.7)

        ax.set_title(f"Global homography — v{V_best} (true v{QR_VERSION}), "
                     f"reproj_err={best_err:.2f}px")
        ax.legend(fontsize=8)
        ax.axis("off")
        if TIGHT_LAYOUT:
            plt.tight_layout()
        plt.show()

# %% [10] Bit sampling
if best_H is not None:
    bits = sample_qr_bits(img_gray, best_H, best_N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bits.T, cmap="gray", interpolation="nearest")
    ax.set_title(f"Sampled bit matrix ({best_N} × {best_N})")
    ax.axis("off")
    if TIGHT_LAYOUT:
        plt.tight_layout()
    plt.show()

# %% [11] Decode
if best_H is not None:
    try:
        decoded = decode(bits)
        print(f"Decoded: '{decoded}'")
    except DecodeError as e:
        print(f"Decode failed: {e}")
