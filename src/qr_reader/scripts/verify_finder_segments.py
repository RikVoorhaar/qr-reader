# %%
"""Verify GT finder-pattern segments — Step 0 of Plan 009.

Generates a v12 test image, computes 36 GT edge segments (12 per finder pattern:
4 sides × 3 module boundaries), and plots them for visual verification.

Inner segments (k=1,2,5,6) are clipped to the visible feature span:
- k=0, k=7: full 7-module width (outer boundary — always visible)
- k=1, k=6: 5 modules (dark↔white ring transition in finder interior)
- k=2, k=5: 3 modules (white ring↔center dark square transition)

Run as a notebook-style script::

    .venv/bin/python src/qr_reader/scripts/verify_finder_segments.py

**HARD GATE** — do not proceed past this step until all 36 segments are visually
confirmed correct.
"""

# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import sys

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

# %% Generate v12-default test image

rng = np.random.default_rng(84)

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
    global_seed=84,
)

H_bg, W_bg = 640, 640
xx = np.linspace(0, 1, W_bg, dtype=np.float32).reshape(1, -1)
yy = np.linspace(0, 1, H_bg, dtype=np.float32).reshape(-1, 1)
bg = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
background = np.stack([bg] * 3, axis=-1)

image, metadata = generate_sample(rng, config, background)
img_gray: np.ndarray = image.mean(axis=-1).astype(np.uint8)

print("Image shape:", image.shape)
print("Version:", metadata["version"])
print("N:", metadata["N"])
print("Payload:", metadata["payload"])


# %% Compute homography from QR module grid to image

N = metadata["N"]
corners = metadata["corners_qr"]

src_xy = np.array(
    [
        [0.0, 0.0],
        [float(N), 0.0],
        [float(N), float(N)],
        [0.0, float(N)],
    ],
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
    """Map a (row, col) module-grid position to an image (x, y) point."""
    pt = np.array([[col, row]], dtype=np.float64)
    return project_points(H, pt)[0]


def _edge_from_endpoints(
    a: np.ndarray,
    b: np.ndarray,
    label: str,
) -> dict:
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
    return {"label": label, "normal": normal, "rho": rho, "segment": np.array([a, b])}


# %% Compute 36 GT finder-pattern segments

FINDER_POSITIONS: dict[str, tuple[int, int]] = {
    "TL": (0, 0),
    "TR": (0, N - 7),
    "BL": (N - 7, 0),
}

TOP_OFFSETS = [0, 1, 2]
BOTTOM_OFFSETS = [5, 6, 7]
LEFT_OFFSETS = [0, 1, 2]
RIGHT_OFFSETS = [5, 6, 7]


def _compute_36_edges() -> list[dict]:
    """Compute 36 GT segments with inner ones clipped to visible features.

    The finder pattern is 7×7 modules. Edges at offset k from the edge span:
    - k=0, k=7: full 7 modules (outer boundary)
    - k=1, k=6: 5 modules (ring transition, skip corners)
    - k=2, k=5: 3 modules (center square transition)
    """
    segments: list[dict] = []
    for finder_name, (r0, c0) in FINDER_POSITIONS.items():
        r1 = r0 + 7
        c1 = c0 + 7

        for side, offsets in [("top", TOP_OFFSETS), ("bot", BOTTOM_OFFSETS)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k), float(c0 + k_vis))
                b = _grid_to_image(float(r0 + k), float(c0 + 7 - k_vis))
                segments.append(_edge_from_endpoints(a, b, f"{finder_name}_{side}{k}"))

        for side, offsets in [("left", LEFT_OFFSETS), ("right", RIGHT_OFFSETS)]:
            for k in offsets:
                k_vis = min(k, 7 - k)
                a = _grid_to_image(float(r0 + k_vis), float(c0 + k))
                b = _grid_to_image(float(r0 + 7 - k_vis), float(c0 + k))
                segments.append(_edge_from_endpoints(a, b, f"{finder_name}_{side}{k}"))

    return segments


all_edges = _compute_36_edges()
assert len(all_edges) == 36, f"Expected 36 edges, got {len(all_edges)}"

print(f"Computed {len(all_edges)} GT edges")
for e in all_edges:
    a, b = e["segment"]
    normal = e["normal"]
    angle_deg = float(np.rad2deg(np.arctan2(normal[1], normal[0])))
    mid = (a + b) / 2
    print(
        f"  {e['label']:20s}  mid=({mid[0]:6.1f},{mid[1]:6.1f})  "
        f"normal_angle={angle_deg:6.1f}°  rho={e['rho']:.1f}"
    )

# %% Full-image plot with all 36 segments

FINDER_COLORS = {
    "TL": "#3388ff",
    "TR": "#33cc66",
    "BL": "#ff8833",
}

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(img_gray, cmap="gray", extent=(0, img_gray.shape[1], img_gray.shape[0], 0))

for e in all_edges:
    finder = e["label"].split("_")[0]
    color = FINDER_COLORS.get(finder, "white")
    a, b = e["segment"]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.0, alpha=0.9)

for name, color in FINDER_COLORS.items():
    ax.plot([], [], color=color, linewidth=2, label=f"{name} finder")
ax.legend(loc="lower left", fontsize=8, framealpha=0.8)

ax.set_title("All 36 GT finder-pattern edge segments")
ax.set_xlim(0, img_gray.shape[1])
ax.set_ylim(img_gray.shape[0], 0)
fig.tight_layout()
plt.show()

# %% Run clustering pipeline to extract ROIs

img_binary = binarize_image(img_gray)
max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
print(f"Alignment candidates: {len(rows_valid)}")

if len(rows_valid) == 0:
    print("No alignment candidates found — cannot extract ROIs.")
    sys.exit(1)

clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"Clusters: {len(clusters)}")

for ci, cluster in enumerate(clusters):
    print(f"  Cluster {ci}: row={cluster.row:.1f}, cols=[{cluster.cols[0]:.1f},...,{cluster.cols[5]:.1f}]")


def _clip_segment(a, b, xmin, xmax, ymin, ymax):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _code(x, y):
        c = INSIDE
        if x < xmin:
            c |= LEFT
        elif x > xmax:
            c |= RIGHT
        if y < ymin:
            c |= TOP
        elif y > ymax:
            c |= BOTTOM
        return c

    x0, y0 = float(a[0]), float(a[1])
    x1, y1 = float(b[0]), float(b[1])
    c0, c1 = _code(x0, y0), _code(x1, y1)
    while True:
        if (c0 | c1) == 0:
            return np.array([[x0, y0], [x1, y1]], dtype=np.float64)
        if (c0 & c1) != 0:
            return None
        oc = c0 if c0 != 0 else c1
        if oc & TOP:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymin
        elif oc & BOTTOM:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymax
        elif oc & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmax
        elif oc & LEFT:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmin
        if oc == c0:
            x0, y0 = x, y
            c0 = _code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = _code(x1, y1)


# %% Per-ROI plots with GT edge overlay

n_clusters = len(clusters)
n_cols = 2
n_rows = (n_clusters + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 5 * n_rows))
if n_clusters == 1:
    axes = np.array([[axes]])
if n_rows == 1:
    axes = axes.reshape(1, -1)

for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)
    if roi.size == 0:
        continue
    r0, r1, c0, c1 = bbox
    r0c = max(0, r0)
    c0c = max(0, c0)

    ax = axes[ci // n_cols, ci % n_cols]
    ax.imshow(roi, cmap="gray", extent=(c0c, c0c + roi.shape[1], r0c + roi.shape[0], r0c))
    ax.set_title(f"Cluster {ci} ROI")

    n_in_roi = 0
    for e in all_edges:
        finder = e["label"].split("_")[0]
        color = FINDER_COLORS.get(finder, "white")
        a, b = e["segment"]
        clipped = _clip_segment(a, b, float(c0c), float(c0c + roi.shape[1]), float(r0c), float(r0c + roi.shape[0]))
        if clipped is not None:
            n_in_roi += 1
            ax.plot([clipped[0, 0], clipped[1, 0]], [clipped[0, 1], clipped[1, 1]], color=color, linewidth=1.0, alpha=0.8)
            ax.plot(clipped[:, 0], clipped[:, 1], "o", color=color, markersize=2)

    ax.axvline(x=c0c, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(x=c0c + roi.shape[1], color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(y=r0c, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(y=r0c + roi.shape[0], color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(c0c, c0c + roi.shape[1])
    ax.set_ylim(r0c + roi.shape[0], r0c)
    print(f"Cluster {ci}: {n_in_roi} GT edges in ROI")

for idx in range(n_clusters, n_rows * n_cols):
    axes[idx // n_cols, idx % n_cols].axis("off")

fig.tight_layout()
plt.show()


# %% Per-finder summary table

print("\n=== Per-finder segment summary ===")
for finder_name in ["TL", "TR", "BL"]:
    print(f"\n{finder_name} finder:")
    finder_edges = [e for e in all_edges if e["label"].startswith(finder_name)]
    assert len(finder_edges) == 12, f"Expected 12 edges, got {len(finder_edges)}"
    for e in finder_edges:
        a, b = e["segment"]
        length = float(np.linalg.norm(b - a))
        angle = float(np.rad2deg(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180
        rho = e["rho"]
        print(
            f"  {e['label']:20s}  len={length:5.1f}px  angle={angle:6.1f}°  "
            f"rho={rho:5.1f}  n=({e['normal'][0]:.4f}, {e['normal'][1]:.4f})"
        )

print("\n=== Done ===")
print("Visually verify all 36 segment positions before proceeding to Step 1.")
