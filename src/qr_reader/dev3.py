# %%
"""dev3.py — QR code reader using modular components."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from qr_reader.qr_gen import binarize_image, generate_test_image
from qr_reader.alignment import (
    find_alignment_patterns,
    find_alignment_patterns_2d,
)
from qr_reader.clustering import cluster_candidates
from qr_reader.region import (
    boundary_connected_components_ndimage,
    boundary_connected_components_networkx,
    get_neighbors,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.corner import angular_nms_top_radial_indices

# %%
# Generate test image (grayscale)

img_gray = generate_test_image()

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
            and img_binary[neighbor[0], neighbor[1]] == img_binary[seed_pixel[0], seed_pixel[1]]
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
        np.asarray(img_binary), seed_row, seed_col,
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
        ang = np.arctan2(
            comp_arr[:, 1] - centroid_i[1], comp_arr[:, 0] - centroid_i[0]
        )
        try:
            idx = angular_nms_top_radial_indices(
                rd, ang, angular_nms_rad=angular_distance_nms, k=4,
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
        corners[:, 1], corners[:, 0],
        s=250, marker="X",
        c=[color[:3]],
        edgecolors="white", linewidths=2.0,
        zorder=10,
        label=f"cluster {ci}" if ci == 0 or ci not in {c for c, _ in all_corners[:ci]} else "",
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
from qr_reader.finder_pattern import extract_finder_patterns, find_all_associations, find_triplets

fps = extract_finder_patterns(all_corners)
print(f"Extracted {len(fps)} finder patterns.")

associations = find_all_associations(fps)
print(f"Found {len(associations)} associations:")
for a in associations:
    print(f" - FP {a.fp1_idx} <-> FP {a.fp2_idx}: segs {a.colinear_segments_1} and {a.colinear_segments_2}")

triplets = find_triplets(fps, associations)
print(f"Found {len(triplets)} triplets:")
for t in triplets:
    print(f" - Top-Left: FP {t.top_left_idx}, Top-Right: FP {t.top_right_idx}, Bottom-Left: FP {t.bottom_left_idx}")
"""
Next we can use the correspondences to put all the 24 corners of each triplet into the right order. 
We use CCW ordering, and the top-left corner is always the first corner. I.e., we have ordering
 0 3
 1 2

If we arange this furhter,  we have ordering

A0 - - A3       C0 - - C3
| B0 B3 |       | D0 D3 |
| B1 B2 |       | D1 D2 |
A1 - - A2       C1 - - C2

E0 - - E3
| F0 F3 |
| F1 F2 |
E1 - - E2


Now we use the fact that N = 4V+17 is the number of cells in the QR code, with V the version number.

If we look at e.g. the left-edge we have points A0, A1, E0, E1 at positions 0, 7, N-7, N. 
They are colinear, but have been perspective transformed. But they should still satisfy the cross ratio rule:

(A0, A1; E0, E1) = ||A0-E0|| ||A1-E1|| / ||A0-E1|| ||E1-A0|| = (N-7)(N-7)/(N)(N-14)

Thus we get two ratios: r_measured and r(N). We can then find the value of V that minimizes
abs(log(r_measured/r(N)))

The neat thing is, that we get the exact same result for (A3, A2; E3, E2), (A0,A3;C0,C3), (A1,A2;C1,C2)

In fact, we also get a similar relation for (B0, B1; F0, F1), but they are at positions 
2,5,N-5,N-2; with cross ratio (N-7)(N-7)/(N-10)(N-4).
This also applies to (B3, B2; F3, F2), (B0, B3; D0, D3), (B1, B2; D1, D2)

Thus we get 8 expected cross ratio relations

Then finally we also have diagonal cross ratio relations with the set of 8 colinear points E1,F1,F3,C1,D1,D3,C3
but I'm not sure what to do with those exactly.

Anyway, we can find the value of V that minimizes the e.g. mean or weighted sum of all these cross ratio errors.
Especially for smaller version numbers that should give a very good estimate of V.
"""