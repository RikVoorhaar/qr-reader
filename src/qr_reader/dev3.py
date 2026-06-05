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
    diag1 = corners[0] - corners[1]
    diag2 = corners[2] - corners[3]
    return np.abs(np.linalg.det(np.vstack([diag1, diag2])))

for ci, corners in all_corners:
    print(f"Cluster {ci} area: {area(corners)}")
all_corners[0][1]
"""We need a primitive that checks how far two line segments are from being co-linear. 

Naive way: we have (p1,p2), (q1,q2) line segments. then the P line is given by p1+t*(p2-p1), and the Q line is given by q1+s*(q2-q1). 

You could like take the max distance between the P line and q1/q2 for example as a colinearity test. 

Yeah, so we want to compute the anglular distance _and_ the max offset. The latter is basically: max(d(q1,P), d(q2,P), d(p1,Q), d(p2,Q)) / L, where L is the distance between (p1+p2)/2 and (q1+q2)/2. To avoid pathological situations, we filter out associations where the finder patterns intersect. 

So TOOD:
    - angular distnace
    - distance between point and line
    - intersection test
    - move area computation above to a module


The idea is then:
    - take all the finder patterns. For each pair, compute an 'association score' or threshold
    - Use area to find the _outer_ corners in each case, and only compare the outer part of the finder patterns
    - For each pair of finder patterns, compute the angular distnaces between all pairs of segments. We should find that each segment is roughly parallel to 2 others, so 8 pairs of parallel segments. For those, compute the offset, and we should find exactly two pairs with low offset -> this tells us how they are alligned. Record the pair of colinear segments
    - Find situations where e.g. A is similar to B, and B is similar to C, and A is not colinear to C, and the colinear segments of A-B are not the same as the colinear segments of B-C. Then B must be the top left corner
    - Extract the 24 landmarks, and map them to the QR code grid. Each finder pattern has 4 outer corners and 4 inner corners.
    - Compute the homography, use some simplified RANSAC to filter outliers.
"""