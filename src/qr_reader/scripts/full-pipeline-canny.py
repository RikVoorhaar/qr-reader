# %%
"""Diagnostic script — pipeline up to clusters, then Sobel+NMS edge extraction + Hough line detection per ROI.

Two figures per cluster:
  1. Four subplots: grayscale cutout | L2 magnitude (raw) | NMS edges | angle
  2. Grayscale cutout with Hough line segments overlaid
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.hough import hough_vote_peaks, refine_line


def _draw_infinite_line(normal, rho, H, W):
    """Return two (x, y) points where the infinite line intersects the ROI boundary.

    Line equation:  normal · p = rho   where p = (x, y).
    """
    nx, ny = normal
    eps = 1e-9
    points = []

    # Check intersection with the four image edges: x=0, x=W-1, y=0, y=H-1.
    # x = 0  →  ny * y = rho - nx * 0  →  y = rho / ny
    if abs(ny) > eps:
        y0 = rho / ny
        if 0 <= y0 < H:
            points.append((0.0, y0))
    # x = W-1
    if abs(ny) > eps:
        yw = (rho - nx * (W - 1)) / ny
        if 0 <= yw < H:
            points.append((float(W - 1), yw))
    # y = 0
    if abs(nx) > eps:
        x0 = rho / nx
        if 0 <= x0 < W:
            points.append((x0, 0.0))
    # y = H-1
    if abs(nx) > eps:
        xh = (rho - ny * (H - 1)) / nx
        if 0 <= xh < W:
            points.append((xh, float(H - 1)))

    if len(points) < 2:
        return np.array([[0, 0], [0, 0]])

    # Take the first two intersection points.
    pts = np.array(points[:2])
    return pts


from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image

# %%
# Load synthetic image (or generate one if data/synth is unavailable)

metadata_path = Path("data/synth/metadata.jsonl")
if metadata_path.exists():
    import json

    with open(metadata_path, "r") as f:
        metadata = [json.loads(line.strip()) for line in f]
    sample = metadata[0]
    QR_VERSION = sample["version"]
    QR_CONTENT = sample["payload"]
    img_path = Path("data/synth/images") / f"{sample['sample_index']:06d}.jpg"
    import cv2

    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    from qr_reader.qr_gen import generate_test_image

    QR_VERSION = 12
    QR_CONTENT = "https://www.rikvoorhaar.com"
    img_gray = generate_test_image(version=QR_VERSION, content=QR_CONTENT, border=15)

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
# Per-cluster ROI → extract_thin_edges → Hough → display
#
# Figure 1 (four subplots):
#   1. Grayscale cutout
#   2. Raw L2 gradient magnitude (Sobel)
#   3. NMS-thinned edges
#   4. Edge-normal angle (atan2), color-mapped
#
# Figure 2: grayscale cutout with Hough line segments overlaid

for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)

    if roi.size == 0:
        print(f"  Cluster {ci}: empty ROI, skipping")
        continue

    nms, angle = extract_thin_edges(roi, blur_sigma=1.0)

    # ---- Figure 1: edge extraction view ---------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
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

    # 4: Angle (color-mapped, modulo π for Hough relevance)
    ax3 = axes[1, 1]
    angle_mod_pi = np.where(nms > 0, angle % np.pi, np.nan)
    im3 = ax3.imshow(angle_mod_pi, cmap="twilight", vmin=0, vmax=np.pi)
    ax3.set_title("Edge angle (mod π)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="radians")

    plt.tight_layout()

    # ---- Hough line detection -------------------------------------------------
    # Tune these to adjust segment extraction behavior.
    GAP_TOLERANCE =4.0  # px — max gap to bridge in support projection
    DISTANCE_THRESH = 1.5  # px — max perpendicular distance from candidate line

    normals, rhos, scores = hough_vote_peaks(nms, angle)

    segments: list = []
    for normal, rho, score in zip(normals, rhos, scores):
        seg = refine_line(
            normal,
            rho,
            score,
            nms,
            angle,
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
            f"n=({seg.normal[0]:.3f}, {seg.normal[1]:.3f})  ρ={seg.rho:.1f}  "
            f"ep=({ep[0, 0]:.1f},{ep[0, 1]:.1f})→({ep[1, 0]:.1f},{ep[1, 1]:.1f})"
        )

    # ---- Figure 2: Hough lines overlaid on grayscale --------------------------
    H_roi, W_roi = roi.shape
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(roi, cmap="gray")
    ax.set_title(f"Cluster {ci} — Hough lines (v{QR_VERSION})")

    # Draw infinite Hough lines (dashed) for all peaks.
    for i, (normal, rho, score) in enumerate(zip(normals, rhos, scores)):
        inf_pts = _draw_infinite_line(normal, rho, H_roi, W_roi)
        ax.plot(
            [inf_pts[0, 0], inf_pts[1, 0]],
            [inf_pts[0, 1], inf_pts[1, 1]],
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            color=f"C{i}",
            label=f"H{i}: ρ={rho:.0f}, s={score:.0f}"
            if i < len(segments)
            else f"H{i}: ρ={rho:.0f}, s={score:.0f}",
        )

    # Draw refined support segments (thick, solid).
    for i, seg in enumerate(segments):
        ep = seg.endpoints
        if np.all(ep == 0):
            continue
        ax.plot(
            [ep[0, 0], ep[1, 0]],
            [ep[0, 1], ep[1, 1]],
            linewidth=4,
            alpha=0.9,
            color=f"C{i}",
            label=f"S{i}: ρ={seg.rho:.0f}, s={seg.vote_score:.0f}",
        )

    ax.legend(fontsize=7, loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
