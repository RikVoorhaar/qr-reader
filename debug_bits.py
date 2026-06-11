"""Debug script to inspect sampled bits vs expected QR pattern."""

import numpy as np

from qr_reader.alignment import find_alignment_patterns_2d
from qr_reader.clustering import cluster_candidates
from qr_reader.corner import angular_nms_top_radial_indices
from qr_reader.decoder.decoder import decode
from qr_reader.finder_pattern import (
    extract_finder_patterns,
    find_all_associations,
    find_triplets,
)
from qr_reader.homography import (
    compute_qr_corners,
    ransac_homography,
    refine_homography_lm,
)
from qr_reader.landmarks import build_named_landmarks, canonical_grid_landmarks
from qr_reader.qr_gen import generate_test_image
from qr_reader.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.sample import (
    compute_adaptive_threshold,
    finder_pattern_known_cells,
    sample_qr_bits,
)
from qr_reader.version import build_constraints, estimate_version, filter_constraints

QR_VERSION = 3
QR_CONTENT = "https://www.rikvoorhaar.com"
N = 4 * QR_VERSION + 17  # 29

img_gray = generate_test_image(version=QR_VERSION, content=QR_CONTENT)

max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_gray > 127, max_error)
clusters = cluster_candidates(rows_valid, cols_valid_all)

all_corners = []
for ci, cluster in enumerate(clusters):
    seed_row = int(cluster.row)
    seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2)
    region_mask = region_fill_wave_front(np.asarray(img_gray > 127), seed_row, seed_col)
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
                rd, ang, angular_nms_rad=10 * 2 * np.pi / 360, k=4
            )
        except ValueError:
            continue
        all_corners.append((ci, comp_arr[idx]))

fps = extract_finder_patterns(all_corners)
associations = find_all_associations(fps)
triplets = find_triplets(fps, associations)
triplet = triplets[0]
landmarks = build_named_landmarks(triplet, fps)

constraints = build_constraints(landmarks)
usable = filter_constraints(constraints, k=4, min_span=1.0)
V_best, scores = estimate_version(usable)
assert V_best == QR_VERSION

grid_lm = canonical_grid_landmarks(N)
image_lm = build_named_landmarks(triplet, fps)


def rc_to_xy(pts):
    return pts[:, ::-1]


src_xy = []
dst_xy = []
for attr in ["A", "B", "C", "D", "E", "F"]:
    g = getattr(grid_lm, attr)
    i = getattr(image_lm, attr)
    if g is not None and i is not None:
        src_xy.append(rc_to_xy(g))
        dst_xy.append(rc_to_xy(i))
src_xy = np.vstack(src_xy)
dst_xy = np.vstack(dst_xy)

H_ransac, inliers = ransac_homography(src_xy, dst_xy, threshold=3.0, iters=2000)
H_refined = refine_homography_lm(H_ransac, src_xy, dst_xy, loss="linear")

matrix = sample_qr_bits(img_gray, H_refined, N)
modules = matrix.T  # canonical QR row/col view; True = dark

black_cells, white_cells = finder_pattern_known_cells(N)

print("=== Finder pattern cell values (sampler: True=dark, False=light) ===")
black_vals = [modules[r, c] for r, c in black_cells]
print(
    f"Black cells: True(dark)={sum(black_vals)}, False(light)={len(black_vals) - sum(black_vals)}"
)
print(f"  Sample: {black_vals[:20]}")

white_vals = [modules[r, c] for r, c in white_cells]
print(
    f"White cells: True(dark)={sum(white_vals)}, False(light)={len(white_vals) - sum(white_vals)}"
)
print(f"  Sample: {white_vals[:20]}")

threshold = compute_adaptive_threshold(img_gray, H_refined, N)
print(
    f"\nThreshold: {threshold}, img min={img_gray.min()}, max={img_gray.max()}, mean={img_gray.mean():.1f}"
)

try:
    text = decode(matrix)
    print(f"\nDecode SUCCESS: {text}")
except Exception as e:
    print(f"\nDecode FAILED: {e}")

print("\nTop-left 7x7:")
for r in range(7):
    print("  " + " ".join("1" if modules[r, c] else "0" for c in range(7)))

print("\nTop-right 7x7:")
for r in range(7):
    print("  " + " ".join("1" if modules[r, c] else "0" for c in range(N - 7, N)))

print("\nBottom-left 7x7:")
for r in range(N - 7, N):
    print("  " + " ".join("1" if modules[r, c] else "0" for c in range(7)))

print("\nRow 8, cols 0-8:")
print("  " + " ".join("1" if modules[8, c] else "0" for c in range(9)))

print("\nRow 8, cols N-8 to N-1:")
print("  " + " ".join("1" if modules[8, c] else "0" for c in range(N - 8, N)))

print("\nCol 8, rows 0-8:")
for r in range(9):
    print(f"  {modules[r, 8]}")

np.save("/home/rik/git/qr-reader/debug_bits.npy", matrix)
print("\nSaved to debug_bits.npy")
