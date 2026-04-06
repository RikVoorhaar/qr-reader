# %%
from typing import NamedTuple

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import qrcode

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

# now to make this jax compatible, we need to use a wave front mask, and process all the pixels in the wave front at once.
# so we need an update step, which for updates in place the region mask, and returns the new wave front, one neighbor dirction at a time. 

# we assume the target value is true, otherwise we invert the img_binary argument.
def region_fill_update(region_mask: jnp.ndarray, wave_front: jnp.ndarray, img_binary: jnp.ndarray, dx: int, dy: int)->tuple[jnp.ndarray, jnp.ndarray]: 
    x_start = max(-dx, 0)
    x_end = min(wave_front.shape[1] - dx, wave_front.shape[1])
    y_start = max(-dy, 0)
    y_end = min(wave_front.shape[0] - dy, wave_front.shape[0])
    wave_front_shifted = wave_front[y_start:y_end, x_start:x_end]
    new_pixels= (img_binary[y_start:y_end, x_start:x_end] & wave_front_shifted)
    # actually we need to shift the wave front, or the image, not both, and which one depends on the direction of the shift.
    # anyway, we find which pixels are newly added, then or them with the region mask, and return them as the new wave front.
    # in the full update step, we do this for all neighbors, and then or the new wave fronts together. we don't even need to update the region mask, since we can use the orred wave front for that as well. more efficient. perhaps this method instead takes the new wave front as an argument, so that it can be updated in place and we don't do memory allocation in this method. for arrays.
    wave_front_shifted 

