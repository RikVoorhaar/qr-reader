# %%
from typing import NamedTuple

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import qrcode
from scipy import ndimage

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data("Some data")
qr.make(fit=True)

img = qr.make_image()

img = np.array(img).astype(np.uint8) * 255

plt.imshow(img, cmap="gray")

# %%

rows, cols = img.shape

# cols-1 and rows-1 are the coordinate limits.
M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 20, 1)
img_rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

plt.imshow(img_rotated, cmap="gray")

# %%

# Define source points (corners of the original QR code image)
src_pts = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])

# Define destination points to apply a perspective transformation
dst_pts = np.float32(
    [
        [20, 50],  # top-left is shifted to the right and down
        [cols - 25, 0],  # top-right a bit left and unshifted vertically
        [cols - 10, rows - 30],  # bottom-right up and in
        [40, rows - 20],  # bottom-left in and up
    ]
)

# Compute the perspective transform matrix
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective warp
img_persp = cv2.warpPerspective(img, M_persp, (cols, rows), borderValue=(255, 255, 255))

# Visualize
plt.imshow(img_persp, cmap="gray")
plt.title("Perspective Transformed QR Code")
plt.show()

# %%
noise = np.random.normal(0, 50, img.shape)
spatial_noise = cv2.GaussianBlur(noise, (3, 3), 0)
img_noisy = np.clip(img_persp * 0.8 + spatial_noise, 0, 255).astype(np.uint8)
plt.imshow(img_noisy, cmap="gray")
plt.title("Noisy QR Code")
plt.show()

# %%
img_noisy = cv2.GaussianBlur(img_noisy, (5, 5), 0)
plt.imshow(img_noisy, cmap="gray")
plt.title("Blurred Noisy QR Code")
plt.show()

# %%
threshold = 128
img_binary = cv2.threshold(img_noisy, threshold, 255, cv2.THRESH_BINARY)[1].astype(bool)
plt.imshow(img_binary, cmap="gray")
plt.title("Binary Noisy QR Code")
plt.show()

# %%
first_row = img_binary[0, :]


def run_length_encoding(row):
    run_lengths = []
    current_run = 0
    current_value = row[0]
    for i in range(len(row)):
        if row[i] == current_value:
            current_run += 1
        else:
            run_lengths.append((current_value, current_run))
            current_value = row[i]
            current_run = 1
    return run_lengths


run_length_encoding(first_row)

# True is white, False is black
# We want to find a ratio of 1:1:3:1:1 for white:black:white:black:white, so True:False:True:False:True
# but within a tolerance of ~10% for each value.
# so for a window of 5 RLE values, if they start with True, we compute the total length, and then check if each member is in the expected range.

row_num = 100
run_lengths = run_length_encoding(img_binary[row_num, :])
run_lengths_smart = np.diff(
    np.where(np.diff(img_binary[row_num, :]) != 0)[0], prepend=0
)

offset = 1
seq = run_lengths_smart[offset : 5 + offset] / sum(
    run_lengths_smart[offset : 5 + offset]
)
expected = np.array([1, 1, 3, 1, 1])
expected = expected / sum(expected)
log_expected = np.log(expected)

score = np.abs(np.max(np.log(seq) - log_expected))

# %%
from numpy.lib.stride_tricks import sliding_window_view

windows = sliding_window_view(run_lengths_smart, window_shape=5)
np.max(
    np.abs(np.log(windows / np.sum(windows, axis=1, keepdims=True)) - log_expected),
    axis=1,
)

# %%
max_error = np.log(1.3)  # 30% error


def find_alignment_patterns(img_binary, max_error):
    run_lengths_smart = np.diff(np.where(np.diff(img_binary) != 0)[0], prepend=0)
    rows, columns = np.where(np.diff(img_binary))

    row_changes = np.diff(rows) > 0
    run_lengths_smart = np.diff(columns)
    run_lengths_smart[row_changes] = 0
    run_lengths_smart

    seqs = sliding_window_view(run_lengths_smart, window_shape=5) + 1e-8
    scores = np.max(
        np.abs(np.log(seqs / np.sum(seqs, axis=1, keepdims=True)) - log_expected),
        axis=1,
    )

    (candidate_indices,) = np.where(scores < max_error)
    candidate_rows = rows[candidate_indices]
    candidate_column_starts = columns[candidate_indices]

    # candidate_column_ends = columns[candidate_indices + 5]
    indices_to_add = np.arange(6)
    candidate_indices_add = candidate_indices.reshape(-1, 1) + indices_to_add
    candidate_columns_all = columns[candidate_indices_add]
    return candidate_rows, candidate_columns_all


rows_x, cols_x_all = find_alignment_patterns(img_binary, max_error)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for row, cols in zip(rows_x, cols_x_all):
    img_plot[row, cols[0] : cols[-1]] = (255, 0, 0)
    img_plot[row, cols[2] : cols[3]] = (255, 150, 0)

plt.imshow(img_plot)
plt.title("Candidate alignment patterns")
plt.show()

# %%

x_values = (cols_x_all[:, 2] + cols_x_all[:, 3]) // 2  # center of the alignment pattern

x_values_unique = np.unique(x_values)
lookup = np.searchsorted(x_values_unique, x_values)


img_flipped = np.ascontiguousarray(img_binary[:, x_values_unique].T)
cols_y, rows_y_all = find_alignment_patterns(img_flipped, max_error)
cols_y = x_values_unique[cols_y]

for col, rows in zip(cols_y, rows_y_all):
    img_plot[rows[0] : rows[-1], col] = (0, 255, 0)
    img_plot[rows[2] : rows[3], col] = (0, 150, 255)

plt.imshow(img_plot)
plt.title("Candidate alignment patterns")
plt.show()


# %%

# check if the middle segments of the x values intersect with any middle segments of the y values

# We need that row_x is between row_y_all[2] and row_y_all[3] and col_x is between col_y_all[2] and col_y_all[3]. That is just a bunch of boolean checks, that fit in a matrix.
rows_x.shape, rows_y_all.shape
cond1 = rows_x.reshape(-1, 1) >= rows_y_all[:, 2].reshape(1, -1)
cond2 = rows_x.reshape(-1, 1) <= rows_y_all[:, 3].reshape(1, -1)
cond3 = cols_y.reshape(1, -1) >= cols_x_all[:, 2].reshape(-1, 1)
cond4 = cols_y.reshape(1, -1) <= cols_x_all[:, 3].reshape(-1, 1)
valid = (cond1 & cond2 & cond3 & cond4).any(axis=1)

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for row, cols in zip(rows_x[valid], cols_x_all[valid]):
    img_plot[row, cols[0] : cols[-1]] = (255, 0, 0)
    img_plot[row, cols[2] : cols[3]] = (255, 150, 0)

plt.imshow(img_plot)
plt.title("Candidate alignment patterns")
plt.show()
# %%

# next step: clustering


class CandidateCluster(NamedTuple):
    row: float
    cols: jnp.ndarray  # shape (6,)
    height: float
    num_candidates: int


rows_cand = jnp.array(rows_x[valid], dtype=jnp.float32)
cols_cand = jnp.array(cols_x_all[valid], dtype=jnp.float32)
height_cand = jnp.array(cols_x_all[valid, 3] - cols_x_all[valid, 2], dtype=jnp.float32)

candidates = CandidateCluster(
    rows_cand, cols_cand, height_cand, jnp.ones(rows_cand.shape, dtype=jnp.int32)
)
num_candidates = candidates.num_candidates.shape[0]


def candidate_length(candidate: CandidateCluster):
    return candidate.cols[5] - candidate.cols[0]


def candidate_lengths_match(
    candidate1: CandidateCluster,
    candidate2: CandidateCluster,
    length_thresh: float = 0.20,
):
    log_length1 = jnp.log(candidate_length(candidate1))
    log_length2 = jnp.log(candidate_length(candidate2))
    return jnp.abs(log_length1 - log_length2) < length_thresh


def candidate_overlaps(
    candidate1: CandidateCluster,
    candidate2: CandidateCluster,
    length_thresh: float = 0.20,
    dist_thresh: float = 1.20,
) -> bool:
    length_match = candidate_lengths_match(candidate1, candidate2, length_thresh)
    # y_match = (candidate1.row - candidate1.height <= candidate2.row) & (
    #     candidate2.row <= candidate1.row + candidate1.height
    # )
    y_match = jnp.abs(candidate1.row - candidate2.row) < dist_thresh * candidate1.height
    candidate2_center = (candidate2.cols[2] + candidate2.cols[3]) / 2
    x_match = (candidate1.cols[2] <= candidate2_center) & (
        candidate2_center <= candidate1.cols[3]
    )
    return length_match & y_match & x_match


def merge_candidates(
    candidate1: CandidateCluster, candidate2: CandidateCluster
) -> CandidateCluster:
    num_candidates = candidate1.num_candidates + candidate2.num_candidates
    row = (
        candidate1.row * candidate1.num_candidates
        + candidate2.row * candidate2.num_candidates
    ) / num_candidates
    cols = (
        candidate1.cols * candidate1.num_candidates
        + candidate2.cols * candidate2.num_candidates
    ) / num_candidates
    height = (
        candidate1.height * candidate1.num_candidates
        + candidate2.height * candidate2.num_candidates
    ) / num_candidates
    return CandidateCluster(row, cols, height, num_candidates)


def choose_ref_candidate(candidates, processed_mask, rand_key):
    rand_key, sub_key = jax.random.split(rand_key)
    # unprocessed_indices = jnp.where(~processed_mask)[0]
    # ref_index = jax.random.randint(sub_key, (1,), 0, unprocessed_indices.size)[0]
    w = (~processed_mask).astype(jnp.float32)
    w = w / w.sum()
    index = jax.random.choice(sub_key, jnp.arange(num_candidates), (1,), p=w)[0]
    ref_candidate = CandidateCluster(
        candidates.row[index],
        candidates.cols[index],
        candidates.height[index],
        candidates.num_candidates[index],
    )
    processed_mask = processed_mask.at[index].set(True)
    return ref_candidate, processed_mask, rand_key


def step_fn(carry, x):
    i, processed_mask, ref_candidate = carry
    overlap = candidate_overlaps(ref_candidate, x)
    # jax.debug.print("x={x}, overlap={overlap}, i={i}, ref_candidate={ref_candidate}", x=x, overlap=overlap, i=i, ref_candidate=ref_candidate)
    return jax.lax.cond(
        overlap & ~processed_mask[i],
        lambda: (
            (
                i + 1,
                processed_mask.at[i].set(True),
                merge_candidates(ref_candidate, x),
            ),
            None,
        ),
        lambda: ((i + 1, processed_mask, ref_candidate), None),
    )


# def while_body(state):
#     processed_mask, ref_candidates, candidates, rand_key = state
#     ref_candidate, processed_mask, rand_key = choose_ref_candidate(candidates, processed_mask, rand_key)
#     (_, processed_mask, ref_candidate), _ = jax.lax.scan(step_fn, (0, processed_mask, ref_candidate), candidates)
#     ref_candidates.append(ref_candidate)
#     return (processed_mask, ref_candidates, candidates, rand_key)

# def while_cond(state):
#     processed_mask, ref_candidates, candidates, rand_key = state
#     return ~processed_mask.all()

rand_key = jax.random.PRNGKey(0)
processed_mask = jnp.zeros(num_candidates, dtype=jnp.bool_)
# state, _ = jax.lax.while_loop(while_cond, while_body, (processed_mask, [], candidates, rand_key))

clusters = []
while not processed_mask.all():
    ref_candidate, processed_mask, rand_key = choose_ref_candidate(
        candidates, processed_mask, rand_key
    )
    (_, processed_mask, ref_candidate), _ = jax.lax.scan(
        step_fn, (0, processed_mask, ref_candidate), candidates
    )
    clusters.append(ref_candidate)

clusters

img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for cluster in clusters:
    row = cluster.row.astype(int)
    cols = cluster.cols.astype(int)
    img_plot[row - 2 : row + 2, cols[0] : cols[1]] = (255, 0, 0)
    img_plot[row - 2 : row + 2, cols[1] : cols[2]] = (0, 255, 0)
    img_plot[row - 2 : row + 2, cols[2] : cols[3]] = (0, 0, 255)
    img_plot[row - 2 : row + 2, cols[3] : cols[4]] = (0, 255, 0)
    img_plot[row - 2 : row + 2, cols[4] : cols[5]] = (255, 0, 0)


plt.imshow(img_plot)
plt.title("Candidate alignment patterns")
plt.show()

# %%

# next step, filling to get the corners. Let's think, what's a good algorithm. The input is a pixel value / location to use as seed. Then the output is a mask of the same shape as the image? where pixels that are connected to the seed are set to True. We maybe even just care about the boundary pixels of the mask?
# In the end, what we need, to do is to get the homography matrix that transforms image space -> qr code space. One way to do that is to get the 4 gons that define the alignment patterns as accurately as possible, and then solve an optimization problem to get the homography matrix.
# So if we have the mask defining the black ring, white ring, black square, then how do from there get the extrema?

# So yeah, we need to find the _contours_. And a pixel in the fill is part of the contour if it has a neighbor that is not part of the fill. Then we pick the two largest connceted components of the contours (for the rings), or the single largest component for the square.
# Then, knowing the contour is a 4-gon, we can optimize the location of the 4 vertices, so that the 4-gon formed by them has half the vertices of the contour inside, and half outside or something.
# but probably that's overkill, we can just find 4 corners, but finding the 4 points the furthest away from the center

# yeah, so we need to figure out the ROI flood fill algorithm in JAX. After that:
# - find the furthest corner by max distance from the center
# - find the other 3 corners by distance from the center in 90 degree increments from first corner direction
# - order by the polar angle
# - that probably already gives reasonable accuracy, but we can do better using optimization
# - For each boundary pixel (i.e. roi that borders the outside of the mask, and is in biggest connected component), we compute the midpoint to the pixel outside the ROI. These are the boundary points.
# - We then minimize the distance between boundary points and the quad defined by the 4 corners, as function of the 4 corners.

# the cool thing is that we can do the above not just for the 4 corners of each alignment pattern, but actually to define the quad of the total QR code, becuase we can still use the same boundary pixels using a complete loss function. Then once we have those points very accurately, we can compute the homography matrix, and transform the image as needed.


# %%
def get_neighbors(pixel):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = (pixel[0] + dy, pixel[1] + dx)
            if (
                0 <= neighbor[0] < img_binary.shape[0]
                and 0 <= neighbor[1] < img_binary.shape[1]
            ):
                neighbors.append(neighbor)
    return neighbors


cluster = clusters[0]
seed_pixel = (int(cluster.row), int((cluster.cols[2] + cluster.cols[3]) // 2))
print(seed_pixel)
seed_pixel_value = img_binary[seed_pixel[0], seed_pixel[1]]
region_mask = np.zeros_like(img_binary, dtype=np.bool_)
region_mask[seed_pixel[0], seed_pixel[1]] = True
queue = {seed_pixel}

while queue:
    pixel = queue.pop()
    neighbors = get_neighbors(pixel)
    for neighbor in neighbors:
        if (
            not region_mask[neighbor[0], neighbor[1]]
            and img_binary[neighbor[0], neighbor[1]] == seed_pixel_value
        ):
            region_mask[neighbor[0], neighbor[1]] = True
            queue.add(neighbor)
img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[region_mask] = (0, 255, 0)
plt.imshow(img_plot)
plt.title("Region mask")
plt.show()

# %%

# JAX-friendly flood fill: expand the wave front to 8-connected neighbors in one step (vectorized).


def expand_wave_front_neighbors(wf: jnp.ndarray) -> jnp.ndarray:
    """OR of wf shifted so each True pixel contributes all 8 neighbors (same connectivity as get_neighbors)."""
    out = jnp.zeros_like(wf)
    out = out.at[:-1, :].set(wf[1:, :])
    out = out.at[1:, :].set(out[1:, :] | wf[:-1, :])
    out = out.at[:, :-1].set(out[:, :-1] | wf[:, 1:])
    out = out.at[:, 1:].set(out[:, 1:] | wf[:, :-1])
    out = out.at[:-1, :-1].set(out[:-1, :-1] | wf[1:, 1:])
    out = out.at[:-1, 1:].set(out[:-1, 1:] | wf[1:, :-1])
    out = out.at[1:, :-1].set(out[1:, :-1] | wf[:-1, 1:])
    out = out.at[1:, 1:].set(out[1:, 1:] | wf[:-1, :-1])
    return out


@jax.jit
def region_fill_wave_front(
    img_binary: jnp.ndarray,
    seed_row: int,
    seed_col: int,
) -> jnp.ndarray:
    """
    Connected region fill (8-neighbor) matching the seed pixel value.
    Same semantics as the Python queue BFS above; iteration is jax.lax.while_loop (no Python while).
    """
    img = jnp.asarray(img_binary)
    target = img[seed_row, seed_col]
    region_mask = jnp.zeros_like(img, dtype=jnp.bool_)
    wave_front = jnp.zeros_like(img, dtype=jnp.bool_)
    wave_front = wave_front.at[seed_row, seed_col].set(True)
    region_mask = region_mask.at[seed_row, seed_col].set(True)

    def cond(carry: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        _, wf = carry
        return jnp.any(wf)

    def body(carry: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        rm, wf = carry
        expanded = expand_wave_front_neighbors(wf)
        new_pixels = expanded & (img == target) & (~rm)
        return (rm | new_pixels, new_pixels)

    region_mask_out, _ = jax.lax.while_loop(cond, body, (region_mask, wave_front))
    return region_mask_out


@jax.jit
def region_boundary_8(region_mask: jnp.ndarray) -> jnp.ndarray:
    """In-region pixels with at least one 8-neighbor not in the region."""
    return region_mask & expand_wave_front_neighbors(~region_mask)


def boundary_connected_components_networkx(
    boundary_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """
    8-connected components among True boundary pixels (NetworkX).
    One ``add_node`` per boundary pixel (isolates have no edges); each undirected
    edge is added once via ``j > i`` in row-major linear index.
    """
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    h, w = boundary_mask.shape
    g = nx.Graph()
    for y, x in zip(*np.where(boundary_mask), strict=True):
        g.add_node((int(y), int(x)))
    for y, x in zip(*np.where(boundary_mask), strict=True):
        i = y * w + x
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and boundary_mask[ny, nx_]:
                    j = ny * w + nx_
                    if j > i:
                        g.add_edge((int(y), int(x)), (int(ny), int(nx_)))
    return [sorted(c) for c in nx.connected_components(g)]


def boundary_connected_components_ndimage(
    boundary_mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """
    Same 8-connected components as ``boundary_connected_components_networkx``,
    using ``scipy.ndimage.label`` (C implementation). Pure-Python union-find often
    loses to NetworkX here: millions of ``find``/``union`` calls still execute in
    Python, while NetworkX’s graph + BFS path is heavily optimized.
    """
    from collections import defaultdict

    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, _n = ndimage.label(boundary_mask, structure=structure)
    by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for y, x in zip(*np.where(boundary_mask), strict=True):
        by_label[int(labeled[y, x])].append((int(y), int(x)))
    return [sorted(v) for _, v in sorted(by_label.items())]


cluster = clusters[1]
# seed_pixel = (int(cluster.row), int((cluster.cols[2] + cluster.cols[3]) // 2))
seed_pixel = (int(cluster.row), int((cluster.cols[0] + cluster.cols[1]) // 2))
region_mask_jax = region_fill_wave_front(
    jnp.asarray(img_binary), seed_pixel[0], seed_pixel[1]
)
img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[np.asarray(region_mask_jax)] = (0, 255, 0)
plt.imshow(img_plot)
plt.title("Region mask (JAX wave front)")
plt.show()


# %%
input_img = jnp.asarray(img_binary)
region_mask_jax = region_fill_wave_front(input_img, seed_pixel[0], seed_pixel[1])
boundary_mask = region_boundary_8(region_mask_jax)
img_plot = img_binary.copy().astype(np.uint8) * 255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
img_plot[np.asarray(boundary_mask)] = (0, 0, 255)  # BGR red
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.title("Region boundary (8-neighbor)")
plt.show()

# %%
boundary_np = np.asarray(boundary_mask)
components_nx = boundary_connected_components_networkx(boundary_np)
components_nd = boundary_connected_components_ndimage(boundary_np)
assert {frozenset(c) for c in components_nx} == {frozenset(c) for c in components_nd}

h, w = boundary_np.shape
rng = np.random.default_rng(0)
cmap = plt.cm.tab20(np.linspace(0, 1, 20))
rgb = np.stack([img_binary.astype(np.float32)] * 3, axis=-1) / 255.0
for i, comp in enumerate(components_nd):
    color = cmap[i % len(cmap)][:3]
    for y, x in comp:
        rgb[y, x] = 0.55 * rgb[y, x] + 0.45 * color
plt.imshow(np.clip(rgb, 0, 1))
plt.title(f"Boundary connected components (ndimage), n={len(components_nd)}")
plt.show()


# %%
"""
next up, we have to do corner finding for the contours. this is a two step process. the input is an ordered list of contour points.
1. find the rough corners. we find max radial distnace from centroid for the biggest contour. we want to do some kind of nms to accurately find 4 corners. using a neighborhood of e.g. 5-10, or a percentage of total, we find local maxima of angular distance from the centroid. then nms is based on angular from the centroid, i.e. maxima are supressed if they are e.g. within 10 degrees of a higher maximum. this is repeated for a top 4.
2. find the precise corners. using the rough corners, classifiy each contour point by segment of the 4 edges. then fit a line to each segment. and then the corners are the 4 intersection points
3. repeat step 2, but using the precise corners to refine further.

for optimal results, we should actually refine the edges using the original gray scale image. we need to estimate gradients from the image, and then use the gradient to nudge the contour points towards the actual edge. that's an interesting add on if we want to go for high accuracy, but really really overkill for a qr code detector.
"""


def angular_nms_top_radial_indices(
    radial_distances: np.ndarray,
    angles: np.ndarray,
    *,
    angular_nms_rad: float,
    k: int = 4,
) -> np.ndarray:
    """
    Pick ``k`` contour indices with largest radial distance (from centroid), with
    angular non-maximum suppression: after each pick, suppress candidates within
    ``angular_nms_rad`` (radians) of that pick's angle, wrapping at ±π.

    Raises ``ValueError`` if no unsuppressed candidates remain before ``k`` picks.
    """
    radial_distances = np.asarray(radial_distances, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if radial_distances.shape != angles.shape:
        raise ValueError("radial_distances and angles must have the same shape")
    if radial_distances.ndim != 1:
        raise ValueError("expected 1-D arrays")
    n = radial_distances.shape[0]
    if n == 0:
        raise ValueError("empty contour")
    supressed_mask = np.ones(n, dtype=bool)
    max_inds: list[int] = []
    neg_inf = -np.finfo(np.float64).max
    for pick in range(k):
        if not np.any(supressed_mask):
            raise ValueError(
                f"angular NMS: no candidates left before pick {pick + 1}/{k}; "
                "increase angular separation (angular_nms_rad) or reduce k."
            )
        masked_scores = np.where(supressed_mask, radial_distances, neg_inf)
        argmax = int(np.argmax(masked_scores))
        max_inds.append(argmax)
        argmax_angle = angles[argmax]
        angular_distances = np.abs(angles - argmax_angle)
        angular_distances = np.minimum(angular_distances, 2 * np.pi - angular_distances)
        supressed_mask[angular_distances < angular_nms_rad] = False
    return np.asarray(max_inds, dtype=np.intp)


# %%
comp = components_nd[0]
comp_np = np.array(comp)
centroid = comp_np.mean(axis=0)
radial_distances = np.linalg.norm(comp_np - centroid, axis=1)
angles = np.arctan2(comp_np[:, 1] - centroid[1], comp_np[:, 0] - centroid[0])
angle_order = np.argsort(angles)
angles_ordered = angles[angle_order]
radial_distances_ordered = radial_distances[angle_order]
plt.plot(angles_ordered, radial_distances_ordered)
plt.show()

# %%
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
rgb_corners = np.stack([img_binary.astype(np.float32)] * 3, axis=-1) / 255.0
for i, comp_i in enumerate(components_nd):
    color = cmap[i % len(cmap)][:3]
    for y, x in comp_i:
        rgb_corners[y, x] = 0.55 * rgb_corners[y, x] + 0.45 * color

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.clip(rgb_corners, 0, 1))
for i, comp_i in enumerate(components_nd):
    comp_arr = np.asarray(comp_i, dtype=np.float64)
    if comp_arr.shape[0] < 4:
        continue
    centroid_i = comp_arr.mean(axis=0)
    rd = np.linalg.norm(comp_arr - centroid_i, axis=1)
    ang = np.arctan2(
        comp_arr[:, 1] - centroid_i[1], comp_arr[:, 0] - centroid_i[0]
    )
    try:
        idx = angular_nms_top_radial_indices(
            rd, ang, angular_nms_rad=angular_distance_nms, k=4
        )
    except ValueError:
        continue
    corners = comp_arr[idx]
    c = cmap[i % len(cmap)]
    ax.scatter(
        corners[:, 1],
        corners[:, 0],
        s=200,
        marker="X",
        c=[c[:3]],
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        label=f"component {i} corners",
    )
ax.set_title("Boundary components + angular NMS corners (per component, if ≥4 points)")
ax.legend(loc="upper right", fontsize=8)
plt.show()

# %%
