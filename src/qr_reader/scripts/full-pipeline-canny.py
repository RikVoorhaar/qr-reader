# %%
"""Diagnostic script — pipeline up to clusters, then Sobel+NMS edge extraction + Hough line detection per ROI.

Three figures per cluster:
  1. Edge extraction: grayscale cutout | L2 magnitude (raw) | NMS edges | angle histogram with GT
  2. Hough accumulator heatmap with GT markers + per-GT zoom panel
  3. ROI overlay with detected segments + GT edges
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.hough import hough_vote_peaks, refine_line


def _draw_infinite_line(normal, rho, H, W):
    """Return two (x, y) points where the infinite line intersects the ROI boundary.

    Line equation:  normal · p = rho   where p = (x, y).
    """
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

    pts = np.array(points[:2])
    return pts


def _normal_theta_deg(normal):
    rad = float(np.arctan2(normal[1], normal[0]))
    if rad < 0:
        rad += np.pi
    return np.rad2deg(rad)


def _compute_gt_edges(metadata, roi_offset, roi_shape):
    """Compute 36 GT finder-pattern edges via module-grid homography.

    12 per finder (TL, TR, BL): 4 sides × 3 module boundaries.
    Inner segments clipped: k_vis = min(k, 7-k) — visible feature span only.
    Returns list of {label, normal, rho} in ROI-local coordinates.
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

    r0_off, c0_off = int(roi_offset[0]), int(roi_offset[1])
    offset_xy = np.array([c0_off, r0_off], dtype=np.float64)

    results = []

    for finder_name, (r0, c0) in finder_positions.items():

        for side, offsets in [("top", TOP), ("bot", BOTTOM)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k), float(c0 + k_vis))
                b = _grid_to_image(float(r0 + k), float(c0 + 7 - k_vis))
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

                rho_local = float(rho - normal @ offset_xy)
                if rho_local < 0:
                    rho_local = -rho_local
                    normal_local = -normal
                else:
                    normal_local = normal.copy()

                results.append({
                    "label": f"{finder_name}_{side}{k}",
                    "normal": normal_local,
                    "rho": rho_local,
                })

        for side, offsets in [("left", LEFT), ("right", RIGHT)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k_vis), float(c0 + k))
                b = _grid_to_image(float(r0 + 7 - k_vis), float(c0 + k))
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

                rho_local = float(rho - normal @ offset_xy)
                if rho_local < 0:
                    rho_local = -rho_local
                    normal_local = -normal
                else:
                    normal_local = normal.copy()

                results.append({
                    "label": f"{finder_name}_{side}{k}",
                    "normal": normal_local,
                    "rho": rho_local,
                })

    return results


from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image

# %%
# Generate sample with GT metadata (so we can mark which edges should be found)
#
# To use a saved sample instead, put it at data/synth/ and uncomment the
# "Load from disk" cell below.
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

rng = np.random.default_rng(42)
config = AugmentationConfig(
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
bg_h, bg_w = 640, 640
xx = np.linspace(0, 1, bg_w, dtype=np.float32).reshape(1, -1)
yy = np.linspace(0, 1, bg_h, dtype=np.float32).reshape(-1, 1)
bg_val = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
background = np.stack([bg_val] * 3, axis=-1)

image, metadata = generate_sample(rng, config, background)
img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)
QR_VERSION = metadata["version"]
QR_CONTENT = metadata["payload"]
print(f"v{QR_VERSION} — {QR_CONTENT}")

# %%
# Binarize (Otsu)

img_binary = binarize_image(img_gray)

# %%
# Find alignment patterns (2-D: horizontal + vertical cross-validation)

max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)

# %%
# Cluster candidates

clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"Found {len(clusters)} clusters.")

# %%
# Hough tuning — best config from ablation sweeps (E6)
THETA_STEP_DEG = 0.5
RHO_STEP = 1.0
NMS_RHO = 2
NMS_THETA = 2
GAP_TOLERANCE = 3.0
DISTANCE_THRESH = 1.5

# %%
# Per-cluster ROI → extract_thin_edges → Hough → display
#
# Figure 1 (four subplots):
#   1. Grayscale cutout
#   2. Raw L2 gradient magnitude (Sobel)
#   3. NMS-thinned edges
#   4. Edge-normal angle histogram with GT markers
#
# Figure 2 (one plot):
#   Hough accumulator heatmap with GT edges (circles) + detected peaks (crosses)
#   + per-GT-edge accumulator zoom inset
#
# Figure 3 (one plot):
#   Grayscale cutout with:
#     - GT edges (red dashed)
#     - Detected refined segments (solid, colored)
#     - All Hough infinite lines (dotted, faint)

for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)

    if roi.size == 0:
        print(f"  Cluster {ci}: empty ROI, skipping")
        continue

    nms, angle_npy = extract_thin_edges(roi, blur_sigma=1.0)
    H_roi, W_roi = roi.shape

    # ---- GT edges ------------------------------------------------------------
    gt_edges = _compute_gt_edges(metadata, (bbox[0], bbox[2]), roi.shape)

    # ---- Figure 1: edge extraction view ---------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f"Cluster {ci} — edge extraction (v{QR_VERSION})")

    # 1: Grayscale cutout
    ax0 = axes[0, 0]
    ax0.imshow(roi, cmap="gray")
    ax0.set_title("Grayscale cutout")
    ax0.axis("off")

    # 2: Raw L2 magnitude (before NMS)
    ax1 = axes[0, 1]
    from scipy import ndimage

    roi_f = roi.astype(np.float64)
    blurred = ndimage.gaussian_filter(roi_f, sigma=1.0, mode="reflect")
    gx = ndimage.sobel(blurred, axis=1, mode="constant")
    gy = ndimage.sobel(blurred, axis=0, mode="constant")
    mag = np.hypot(gx, gy)
    ax1.imshow(mag, cmap="gray")
    ax1.set_title("Sobel L2 magnitude")
    ax1.axis("off")

    # 3: NMS edges
    ax2 = axes[1, 0]
    ax2.imshow(nms, cmap="gray")
    n_max = nms.max()
    n_nonzero = int(np.count_nonzero(nms))
    ax2.set_title(f"NMS edges (max={n_max:.1f}, nonzero={n_nonzero})")
    ax2.axis("off")

    # 4: NMS angle histogram with GT edge normals
    ax3 = axes[1, 1]
    ys_nz, xs_nz = np.nonzero(nms)
    thetas = np.fmod(angle_npy[ys_nz, xs_nz], np.pi)
    thetas = np.where(thetas < 0, thetas + np.pi, thetas)
    strengths = nms[ys_nz, xs_nz]
    ax3.hist(np.rad2deg(thetas), bins=90, range=(0, 180),
             weights=strengths, color="steelblue", edgecolor="white", alpha=0.8)
    for i, gt in enumerate(gt_edges):
        gt_th = _normal_theta_deg(gt["normal"])
        ax3.axvline(gt_th, color=f"C{i}", linestyle="--",
                    linewidth=2, label=gt["label"])
    ax3.set_xlabel("Angle (deg)")
    ax3.set_title("NMS angle histogram + GT normals")
    ax3.legend(fontsize=7, loc="upper right")

    plt.tight_layout()

    # ---- Hough line detection -------------------------------------------------
    normals, rhos, scores, acc_data = hough_vote_peaks(
        nms, angle_npy,
        theta_step_deg=THETA_STEP_DEG,
        rho_step=RHO_STEP,
        nms_radius_rho=NMS_RHO,
        nms_radius_theta=NMS_THETA,
        theta_window_deg=0.0,
        vote_scheme="onebin",
        return_acc=True,
    )
    acc = acc_data["acc"]
    n_theta = acc_data["n_theta"]
    n_rho = acc_data["n_rho"]

    segments: list = []
    for normal, rho, score in zip(normals, rhos, scores):
        seg = refine_line(
            normal,
            float(rho),
            float(score),
            nms,
            angle_npy,
            gap_tolerance=GAP_TOLERANCE,
            distance_thresh=DISTANCE_THRESH,
        )
        if not np.all(seg.endpoints == 0):
            segments.append(seg)

    print(
        f"  Cluster {ci}: {len(normals)} Hough peaks → {len(segments)} refined segments"
    )
    for i, seg in enumerate(segments):
        ep = seg.endpoints
        print(
            f"    [{i}] score={seg.vote_score:.1f}  "
            f"θ={_normal_theta_deg(seg.normal):.1f}°  ρ={seg.rho:.1f}  "
            f"ep=({ep[0, 0]:.1f},{ep[0, 1]:.1f})→({ep[1, 0]:.1f},{ep[1, 1]:.1f})"
        )

    # ---- Figure 2: Hough accumulator heatmap + GT markers -------------------
    fig2, ax_acc = plt.subplots(figsize=(12, 6))
    fig2.suptitle(f"Cluster {ci} — Hough Accumulator (θ-step={THETA_STEP_DEG}°)")

    acc_extent = [0, min(n_theta * THETA_STEP_DEG, 180), 0, n_rho * RHO_STEP]
    im = ax_acc.imshow(acc.T, origin="lower", aspect="auto", extent=acc_extent,
                       cmap="inferno", vmax=acc.max() * 0.3)
    plt.colorbar(im, ax=ax_acc, label="Votes", fraction=0.046, pad=0.04)
    ax_acc.set_xlabel("θ (deg)")
    ax_acc.set_ylabel("ρ (px)")

    # GT edge positions (circles)
    for i, gt in enumerate(gt_edges):
        gt_th = _normal_theta_deg(gt["normal"])
        ax_acc.plot(gt_th, gt["rho"], "o", color=f"C{i}",
                    markersize=12, markeredgewidth=2, markeredgecolor="white",
                    label=f"GT {gt['label']}")

    # Detected peaks (crosses)
    for i, (normal, rho, score) in enumerate(zip(normals, rhos, scores)):
        p_th = _normal_theta_deg(normal)
        ax_acc.plot(p_th, rho, "x", color="cyan", markersize=10, markeredgewidth=2)
        ax_acc.annotate(f"{i}", (p_th, rho), fontsize=6, color="white",
                        xytext=(3, 3), textcoords="offset points")

    ax_acc.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    # ---- Print GT edge analysis -----------------------------------------------
    threshold_rel = 0.25
    threshold = threshold_rel * acc.max()
    angle_tol_deg = 5.0
    rho_tol = 5.0
    print(f"\n  GT edges in ROI ({len(gt_edges)}):")
    for gt in gt_edges:
        gt_theta = np.deg2rad(_normal_theta_deg(gt["normal"]))
        gt_ti = int(np.round(gt_theta / acc_data["theta_step_rad"])) % n_theta
        gt_ri = int(np.round(gt["rho"] / RHO_STEP))
        gt_ri = max(0, min(n_rho - 1, gt_ri))
        gt_val = float(acc[gt_ti, gt_ri])

        # Check for matching peak
        matched = False
        for pi, (normal, rho) in enumerate(zip(normals, rhos)):
            ang_dist = min(
                abs(_normal_theta_deg(normal) - np.rad2deg(gt_theta)) % 180,
                180 - abs(_normal_theta_deg(normal) - np.rad2deg(gt_theta)) % 180,
            )
            rho_dist = abs(rho - gt["rho"])
            if ang_dist <= angle_tol_deg and rho_dist <= rho_tol:
                matched = True
                break

        # Window sum
        window_sum = 0.0
        dth = max(1, int(np.ceil(angle_tol_deg / THETA_STEP_DEG)))
        dr = max(1, int(np.ceil(rho_tol / RHO_STEP)))
        for dt in range(-dth, dth + 1):
            tt = (gt_ti + dt) % n_theta
            window_sum += float(
                acc[tt, max(0, gt_ri - dr):min(n_rho, gt_ri + dr + 1)].sum()
            )

        status = "HIT" if matched else "MISS"
        print(
            f"    {gt['label']:>10s} [{status}]  "
            f"θ={_normal_theta_deg(gt['normal']):.1f}°  ρ={gt['rho']:.1f}  "
            f"GT-bin={gt_val:.0f}  window_sum={window_sum:.0f}"
            + (f"  (threshold={threshold:.0f})" if not matched else "")
        )

    # ---- Figure 3: Hough lines overlaid on grayscale --------------------------
    fig3, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(roi, cmap="gray")
    ax.set_title(f"Cluster {ci} — Hough lines (v{QR_VERSION})")

    # Draw GT edges (dashed red)
    for i, gt in enumerate(gt_edges):
        normal = gt["normal"]
        rho = gt["rho"]
        inf_pts = _draw_infinite_line(normal, rho, H_roi, W_roi)
        ax.plot(
            [inf_pts[0, 0], inf_pts[1, 0]],
            [inf_pts[0, 1], inf_pts[1, 1]],
            linestyle="--", color=f"C{i}", linewidth=2, alpha=0.6,
            label=f"GT {gt['label']}",
        )

    # Draw refined support segments (thick, solid)
    for i, seg in enumerate(segments):
        ep = seg.endpoints
        ax.plot(
            [ep[0, 0], ep[1, 0]],
            [ep[0, 1], ep[1, 1]],
            linewidth=4, alpha=0.9, color=f"C{i}",
            label=f"S{i}: θ={_normal_theta_deg(seg.normal):.0f}° "
                  f"s={seg.vote_score:.0f}",
        )

    # Draw infinite Hough lines (dotted, faint — for all peaks)
    for i, (normal, rho, score) in enumerate(zip(normals, rhos, scores)):
        inf_pts = _draw_infinite_line(normal, rho, H_roi, W_roi)
        ax.plot(
            [inf_pts[0, 0], inf_pts[1, 0]],
            [inf_pts[0, 1], inf_pts[1, 1]],
            linestyle=":", linewidth=0.5, alpha=0.2, color=f"C{i}",
        )

    ax.legend(fontsize=7, loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
