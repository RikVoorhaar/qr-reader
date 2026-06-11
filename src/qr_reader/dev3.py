# %%
"""dev3.py — QR code reader using modular components."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from qr_reader.alignment import (
    find_alignment_patterns,
    find_alignment_patterns_2d,
)
from qr_reader.clustering import cluster_candidates
from qr_reader.corner import angular_nms_top_radial_indices
from qr_reader.geometry import polygon_area
from qr_reader.qr_gen import binarize_image, generate_test_image
from qr_reader.region import (
    boundary_connected_components_ndimage,
    boundary_connected_components_networkx,
    get_neighbors,
    region_boundary_8,
    region_fill_wave_front,
)

# %%
# Generate test image (grayscale)
QR_VERSION = 3
QR_CONTENT = "https://www.rikvoorhaar.com"


img_gray = generate_test_image(version=QR_VERSION, content=QR_CONTENT)

plt.imshow(img_gray, cmap="gray")
plt.title("Noisy QR Code (grayscale)")
plt.show()

# %%
# Binarize (Otsu)

img_binary = binarize_image(img_gray)

plt.imshow(img_binary, cmap="gray")
plt.title("Binary QR Code (Otsu)")
plt.show()

# %%
# Find alignment patterns (horizontal only, for demo)

max_error = np.log(1.3)  # 30% error
rows_x, cols_x_all = find_alignment_patterns(img_binary, max_error)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for row, cols in zip(rows_x, cols_x_all):
    img_plot[row, cols[0] : cols[-1]] = (255, 0, 0)
    img_plot[row, cols[2] : cols[3]] = (255, 150, 0)

plt.imshow(img_plot)
plt.title("Candidate alignment patterns (horizontal)")
plt.show()

# %%
# Find alignment patterns (2-D: horizontal + vertical cross-validation)

rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for row, cols in zip(rows_valid, cols_valid_all):
    img_plot[row, cols[0] : cols[-1]] = (255, 0, 0)
    img_plot[row, cols[2] : cols[3]] = (255, 150, 0)

plt.imshow(img_plot)
plt.title("Candidate alignment patterns (2-D validated)")
plt.show()

# %%
# Cluster candidates

clusters = cluster_candidates(rows_valid, cols_valid_all)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for cluster in clusters:
    row = int(cluster.row)
    cols = cluster.cols.astype(int)
    img_plot[row - 2 : row + 2, cols[0] : cols[1]] = (255, 0, 0)
    img_plot[row - 2 : row + 2, cols[1] : cols[2]] = (0, 255, 0)
    img_plot[row - 2 : row + 2, cols[2] : cols[3]] = (0, 0, 255)
    img_plot[row - 2 : row + 2, cols[3] : cols[4]] = (0, 255, 0)
    img_plot[row - 2 : row + 2, cols[4] : cols[5]] = (255, 0, 0)

plt.imshow(img_plot)
plt.title("Clustered alignment patterns")
plt.show()

# %%
# Queue-based flood fill (demo: cluster[0], centre seed)

cluster = clusters[0]
seed_pixel = (int(cluster.row), int((cluster.cols[2] + cluster.cols[3]) // 2))
print(seed_pixel)

region_mask_queue = np.zeros_like(img_binary, dtype=bool)
region_mask_queue[seed_pixel[0], seed_pixel[1]] = True
queue = {seed_pixel}

while queue:
    pixel = queue.pop()
    for neighbor in get_neighbors(pixel, img_binary.shape):
        if (
            not region_mask_queue[neighbor[0], neighbor[1]]
            and img_binary[neighbor[0], neighbor[1]]
            == img_binary[seed_pixel[0], seed_pixel[1]]
        ):
            region_mask_queue[neighbor[0], neighbor[1]] = True
            queue.add(neighbor)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[region_mask_queue] = (0, 255, 0)
plt.imshow(img_plot)
plt.title("Region mask (queue-based)")
plt.show()

# %%
# Wave-front flood fill (demo: cluster[1], left-edge seed)

cluster = clusters[1]
seed_pixel = (int(cluster.row), int((cluster.cols[0] + cluster.cols[1]) // 2))

region_mask_wf = region_fill_wave_front(
    np.asarray(img_binary), seed_pixel[0], seed_pixel[1]
)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[np.asarray(region_mask_wf)] = (0, 255, 0)
plt.imshow(img_plot)
plt.title("Region mask (NumPy wave front)")
plt.show()

# %%
# Boundary extraction (demo)

boundary_mask = region_boundary_8(region_mask_wf)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[np.asarray(boundary_mask)] = (0, 0, 255)
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.title("Region boundary (8-neighbor)")
plt.show()

# %%
# Connected components (NetworkX + ndimage, demo)

boundary_np = np.asarray(boundary_mask)
components_nx = boundary_connected_components_networkx(boundary_np)
components_nd = boundary_connected_components_ndimage(boundary_np)
assert {frozenset(c) for c in components_nx} == {frozenset(c) for c in components_nd}

cmap = cm.tab20(np.linspace(0, 1, 20))
rgb = np.stack([img_binary.astype(np.float32)] * 3, axis=-1) / 255.0
for i, comp in enumerate(components_nd):
    color = cmap[i % len(cmap)][:3]
    for y, x in comp:
        rgb[y, x] = color
plt.imshow(np.clip(rgb, 0, 1))
plt.title(f"Boundary connected components (ndimage), n={len(components_nd)}")
plt.show()

# %%
# Angular NMS corner finding on component[0] (demo)

comp = components_nd[0]
comp_np = np.array(comp)
centroid = comp_np.mean(axis=0)
radial_distances = np.linalg.norm(comp_np - centroid, axis=1)
angles = np.arctan2(comp_np[:, 1] - centroid[1], comp_np[:, 0] - centroid[0])
angle_order = np.argsort(angles)
angles_ordered = angles[angle_order]
radial_distances_ordered = radial_distances[angle_order]

angular_distance_nms = 10 * 2 * np.pi / 360  # 10 degrees
max_inds = angular_nms_top_radial_indices(
    radial_distances,
    angles,
    angular_nms_rad=angular_distance_nms,
    k=4,
)

plt.plot(angles_ordered, radial_distances_ordered)
plt.scatter(angles[max_inds], radial_distances[max_inds], color="red", s=60, zorder=5)
plt.xlabel("angle (rad)")
plt.ylabel("radial distance")
plt.title("Radial distance vs angle (ordered) + angular NMS maxima")
plt.show()

# %%
# ——— All corners for every detected cluster ——————————————————————————————

# Per-cluster corner-finding: fill → boundary → components → angular NMS.
# We seed in the first black segment (cols[0:1]) to fill the outer black
# ring, whose boundary has two connected components (inner + outer edges).

all_corners: list[tuple[int, np.ndarray]] = []  # (cluster_idx, corners array)

for ci, cluster in enumerate(clusters):
    seed_row = int(cluster.row)
    seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2)

    # Flood fill the outer black ring
    region_mask = region_fill_wave_front(
        np.asarray(img_binary),
        seed_row,
        seed_col,
    )

    # Boundary → connected components
    boundary = region_boundary_8(region_mask)
    components = boundary_connected_components_ndimage(np.asarray(boundary))

    for comp in components:
        comp_arr = np.asarray(comp, dtype=np.float64)
        if comp_arr.shape[0] < 4:
            continue
        centroid_i = comp_arr.mean(axis=0)
        rd = np.linalg.norm(comp_arr - centroid_i, axis=1)
        ang = np.arctan2(comp_arr[:, 1] - centroid_i[1], comp_arr[:, 0] - centroid_i[0])
        try:
            idx = angular_nms_top_radial_indices(
                rd,
                ang,
                angular_nms_rad=angular_distance_nms,
                k=4,
            )
        except ValueError:
            print("ValueError: angular_nms_top_radial_indices failed")
            continue
        all_corners.append((ci, comp_arr[idx]))

# ——— Plot ——————————————————————————————————————————————————————————————————

rgb_all = np.stack([img_binary.astype(np.float32)] * 3, axis=-1) / 255.0
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.clip(rgb_all, 0, 1))

for ci, corners in all_corners:
    color = cmap[ci % len(cmap)]
    ax.scatter(
        corners[:, 1],
        corners[:, 0],
        s=250,
        marker="X",
        c=[color[:3]],
        edgecolors="white",
        linewidths=2.0,
        zorder=10,
        label=f"cluster {ci}"
        if ci == 0 or ci not in {c for c, _ in all_corners[:ci]}
        else "",
    )

ax.set_title(f"All corners across {len(clusters)} clusters")
ax.legend(loc="upper right", fontsize=7)
plt.show()

# %%
all_corners


def area(corners):
    diag1 = corners[0] - corners[2]
    diag2 = corners[1] - corners[3]
    return np.abs(np.linalg.det(np.vstack([diag1, diag2])))


for ci, corners in all_corners:
    print(f"Cluster {ci} area: {area(corners)}")
from qr_reader.finder_pattern import (
    extract_finder_patterns,
    find_all_associations,
    find_triplets,
)

fps = extract_finder_patterns(all_corners)
print(f"Extracted {len(fps)} finder patterns.")

associations = find_all_associations(fps)
print(f"Found {len(associations)} associations:")
for a in associations:
    print(
        f" - FP {a.fp1_idx} <-> FP {a.fp2_idx}: segs {a.colinear_segments_1} and {a.colinear_segments_2}"
    )

triplets = find_triplets(fps, associations)
print(f"Found {len(triplets)} triplets:")
for t in triplets:
    print(
        f" - Top-Left: FP {t.top_left_idx}, Top-Right: FP {t.top_right_idx}, Bottom-Left: FP {t.bottom_left_idx}"
    )

# %%
# Step A — Verify inner corners are carried on FinderPattern

for fp in fps:
    has_inner = fp.inner_corners is not None
    print(
        f"FP {fp.cluster_idx}: outer area={polygon_area(fp.outer_corners):.1f}, "
        f"{'has inner' if has_inner else 'no inner'}"
    )

# %%
# Step B — Build named landmarks from the first triplet

from qr_reader.landmarks import (
    build_named_landmarks,
    get_colinear_quadruples,
)

triplet = triplets[0]
landmarks = build_named_landmarks(triplet, fps)

print("Named landmarks built:")
for name, pts in [("A", landmarks.A), ("C", landmarks.C), ("E", landmarks.E)]:
    print(f"  {name}: {pts.tolist()}")
for name, pts in [("B", landmarks.B), ("D", landmarks.D), ("F", landmarks.F)]:
    has = pts is not None
    print(f"  {name}: {'present' if has else 'None'}")

# %%
# Plot the 6 ordered squares on the image

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")

colors = {
    "A": "red",
    "B": "orange",
    "C": "blue",
    "D": "cyan",
    "E": "green",
    "F": "lime",
}
for name, pts in [
    ("A", landmarks.A),
    ("B", landmarks.B),
    ("C", landmarks.C),
    ("D", landmarks.D),
    ("E", landmarks.E),
    ("F", landmarks.F),
]:
    if pts is None:
        continue
    # Close the quad for plotting: 0-1-2-3-0
    quad = np.vstack([pts, pts[0:1]])
    ax.plot(
        quad[:, 1], quad[:, 0], color=colors.get(name, "white"), linewidth=2, label=name
    )
    # Label the corners
    for i, (label, offset) in enumerate(
        [("TL", (-8, -8)), ("BL", (5, -8)), ("BR", (5, 5)), ("TR", (-8, 5))]
    ):
        ax.annotate(
            f"{name}{i} ({label})",
            (pts[i, 1], pts[i, 0]),
            textcoords="offset pixels",
            xytext=offset,
            fontsize=6,
            color=colors.get(name, "white"),
        )

ax.legend(fontsize=7, loc="upper right")
ax.set_title("Ordered finder-pattern squares (A-F)")
plt.show()

# %%
# Step C — Get colinear quadruples from image landmarks

quads = get_colinear_quadruples(landmarks)
print(
    f"Colinear quadruples: {len(quads)} total "
    f"({sum(1 for q in quads if q.type == 'outer')} outer, "
    f"{sum(1 for q in quads if q.type == 'inner')} inner)"
)

# Plot quadruples on the image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")
for q in quads:
    pts = q.points
    style = "--" if q.type == "inner" else "-"
    color = "cyan" if q.type == "inner" else "yellow"
    ax.plot(pts[:, 1], pts[:, 0], style, color=color, linewidth=1.5, alpha=0.8)
    ax.scatter(
        pts[:, 1],
        pts[:, 0],
        s=30,
        c=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
    )
ax.set_title("Colinear quadruples (yellow=outer, cyan=inner)")
plt.show()

# %%
# Step D — Version estimation via cross-ratios

from qr_reader.version import (
    build_constraints,
    estimate_version,
    expected_cross_ratio_by_N,
    filter_constraints,
)

constraints = build_constraints(landmarks)
print(f"Built {len(constraints)} constraints:")
for c in constraints:
    print(
        f"  {c.label} ({c.type}): r={c.r_measured:.4f}, line_error={c.line_error:.4f}, span={c.span:.1f}"
    )

# Filter and estimate
usable = filter_constraints(constraints, k=4, min_span=1.0)
print(f"\nAfter filtering: {len(usable)} constraints kept")

V_best, scores = estimate_version(usable)
N_best = 4 * V_best + 17
print(f"\nInferred version: V={V_best}  (N={N_best})")

# Show top-5 scores
sorted_idx = np.argsort(scores)
print("\nTop 5 version scores (lower is better):")
for rank, idx in enumerate(sorted_idx[:5]):
    V = idx + 1  # v_range starts at 1
    print(f"  #{rank + 1}: V={V} (N={4 * V + 17}), score={scores[idx]:.6f}")

# Plot constraints colored by line_error / filter status
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")
kept_labels = {c.label for c in usable}
for c in constraints:
    pts = next(q.points for q in quads if q.label == c.label)
    kept = c.label in kept_labels
    color = "lime" if kept else "red"
    alpha = 1.0 if kept else 0.3
    ax.plot(pts[:, 1], pts[:, 0], "-", color=color, linewidth=2, alpha=alpha)
    ax.scatter(
        pts[:, 1],
        pts[:, 0],
        s=40,
        c=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        alpha=alpha,
    )
ax.set_title(
    f"Constraints for version estimation: kept={len(usable)}/{len(constraints)}  →  V={V_best}"
)
plt.show()

# %%
# Smoke check: inferred version matches generator's version (1)

assert V_best == QR_VERSION, f"Expected V={QR_VERSION}, got V={V_best}"
print(f"✓ Version check passed: inferred V={V_best} matches generator's V={QR_VERSION}")

# Show expected cross-ratios for reference
outer_exp, inner_exp = expected_cross_ratio_by_N(N_best)
print(f"\nExpected cross-ratios for V={V_best} (N={N_best}):")
print(f"  outer: {outer_exp:.6f}")
print(f"  inner: {inner_exp:.6f}")

# %%
# Step E — Homography estimation (DLT + RANSAC + LM)

from qr_reader.homography import (
    compute_qr_corners,
    ransac_homography,
    refine_homography_lm,
)
from qr_reader.landmarks import build_named_landmarks, canonical_grid_landmarks

# Build correspondences: canonical grid landmarks → detected image landmarks
# N_best was computed in the version-estimation step above.
grid_lm = canonical_grid_landmarks(N_best)
image_lm = build_named_landmarks(triplet, fps)


def rc_to_xy(pts: np.ndarray) -> np.ndarray:
    """Convert (row, col) → (x, y) by swapping columns."""
    return pts[:, ::-1]


# Gather the 24 correspondences (up to 24 if all inner corners available)
src_xy = []  # grid (canonical)
dst_xy = []  # image (detected)
for attr in ["A", "B", "C", "D", "E", "F"]:
    g = getattr(grid_lm, attr)
    i = getattr(image_lm, attr)
    if g is not None and i is not None:
        src_xy.append(rc_to_xy(g))
        dst_xy.append(rc_to_xy(i))
src_xy = np.vstack(src_xy)
dst_xy = np.vstack(dst_xy)
print(f"Built {len(src_xy)} correspondences for homography.")

# RANSAC homography
H_ransac, inliers = ransac_homography(src_xy, dst_xy, threshold=3.0, iters=2000)
inlier_count = np.sum(inliers)
print(f"RANSAC inliers: {inlier_count} / {len(src_xy)}")

# LM refinement
H_refined = refine_homography_lm(H_ransac, src_xy, dst_xy, loss="linear")

# Compute QR corners
corners_xy = compute_qr_corners(H_refined, N_best)
print(f"\nQR corners (x, y):")
for i, label in enumerate(["TL", "TR", "BR", "BL"]):
    print(f"  {label}: ({corners_xy[i, 0]:.1f}, {corners_xy[i, 1]:.1f})")

# --- Plot: overlay QR corners and boundary on the image ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_gray, cmap="gray")

# Draw the quadrangular QR boundary
quad_closed = np.vstack([corners_xy, corners_xy[0:1]])
ax.plot(quad_closed[:, 0], quad_closed[:, 1], "r-", linewidth=2, label="QR boundary")
ax.scatter(
    corners_xy[:, 0],
    corners_xy[:, 1],
    s=200,
    marker="X",
    c="red",
    edgecolors="white",
    linewidths=2.0,
    zorder=10,
)

# Also overlay all inlier landmark correspondences
ax.scatter(
    dst_xy[inliers, 0],
    dst_xy[inliers, 1],
    s=40,
    c="lime",
    edgecolors="black",
    linewidths=0.5,
    alpha=0.6,
    label=f"RANSAC inliers ({inlier_count})",
)
if inlier_count < len(src_xy):
    ax.scatter(
        dst_xy[~inliers, 0],
        dst_xy[~inliers, 1],
        s=60,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        alpha=0.6,
        label=f"Outliers ({len(src_xy) - inlier_count})",
    )

for i, label in enumerate(["TL", "TR", "BR", "BL"]):
    ax.annotate(
        label,
        (corners_xy[i, 0], corners_xy[i, 1]),
        textcoords="offset pixels",
        xytext=(10, 10),
        fontsize=10,
        color="red",
        fontweight="bold",
    )

ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"QR corners via homography (V={V_best}, N={N_best})")
ax.set_xlabel("x (col)")
ax.set_ylabel("y (row)")
plt.show()

# %%
# Step G — Supersample QR bits from grayscale & decode via OpenCV
from qr_reader.decode import decode_qr
from qr_reader.sample import sample_qr_bits

bits = sample_qr_bits(img_gray, H_refined, N_best)
print(f"Sampled grid shape: {bits.shape}, white fraction: {bits.mean():.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(bits, cmap="gray", interpolation="nearest")
ax.set_title(f"Sampled QR bits (V={V_best}, N={N_best})")
plt.show()

# Build a clean uint8 image for OpenCV
# Up-scale with box_size=10 and add a white quiet-zone border (4 modules)
box_size = 10
border = 4
img_clean = np.full(
    ((N_best + 2 * border) * box_size, (N_best + 2 * border) * box_size),
    255,
    dtype=np.uint8,
)
for r in range(N_best):
    for c in range(N_best):
        val = 255 if bits[r, c] else 0
        img_clean[
            (r + border) * box_size : (r + border + 1) * box_size,
            (c + border) * box_size : (c + border + 1) * box_size,
        ] = val

decoded_text, ok = decode_qr(img_clean, corners_xy=None)  # let OpenCV find corners

if ok:
    print(f'✓ Decoded from sampled bits: "{decoded_text}"')
else:
    print("✗ Decode failed from sampled bits")

assert ok, f"Decode failed for V={V_best}"
assert decoded_text == QR_CONTENT, (
    f"Content mismatch: expected '{QR_CONTENT}', got '{decoded_text}'"
)
print(f"✓ Content check passed: '{decoded_text}' == '{QR_CONTENT}'")

# %%
# Step F — Decode the QR code using OpenCV (with corners)

from qr_reader.decode import decode_qr

decoded_text, ok = decode_qr(img_gray, corners_xy)

if ok:
    print(f'✓ Decoded: "{decoded_text}"')
else:
    print(f"✗ Decode failed")

# Final assertion: decoded text matches the generated content
assert ok, f"Decode failed for V={V_best}"
assert decoded_text == QR_CONTENT, (
    f"Content mismatch: expected '{QR_CONTENT}', got '{decoded_text}'"
)
print(
    f"✓ Content check passed: decoded '{decoded_text}' matches generator's '{QR_CONTENT}'"
)
