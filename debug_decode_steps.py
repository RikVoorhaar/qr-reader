"""Debug each step of the decode pipeline."""

import numpy as np

from qr_reader.alignment import find_alignment_patterns_2d
from qr_reader.clustering import cluster_candidates
from qr_reader.corner import angular_nms_top_radial_indices
from qr_reader.decoder.codeword_extractor import extract_codewords
from qr_reader.decoder.data_block import deinterleave
from qr_reader.decoder.format_info import decode_format_info
from qr_reader.decoder.rs import rs_decode
from qr_reader.decoder.tables import ECL_NAMES
from qr_reader.finder_pattern import (
    extract_finder_patterns,
    find_all_associations,
    find_triplets,
)
from qr_reader.homography import ransac_homography, refine_homography_lm
from qr_reader.landmarks import build_named_landmarks, canonical_grid_landmarks
from qr_reader.qr_gen import generate_test_image
from qr_reader.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.sample import sample_qr_bits
from qr_reader.version import build_constraints, estimate_version, filter_constraints

QR_VERSION = 3
QR_CONTENT = "https://www.rikvoorhaar.com"
N = 29

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

matrix = sample_qr_bits(img_gray, H_refined, N)  # True = dark for decoder

print("=== Step 1: Format Info ===")
try:
    ecl_idx, mask_idx = decode_format_info(matrix, QR_VERSION)
    print(f"  ECL index: {ecl_idx} -> {ECL_NAMES[ecl_idx]}")
    print(f"  Mask index: {mask_idx}")
except Exception as e:
    print(f"  FAILED: {e}")
    ecl_idx = mask_idx = None

if ecl_idx is not None:
    print("\n=== Step 2: Extract Codewords ===")
    try:
        raw = extract_codewords(matrix, QR_VERSION, mask_idx)
        print(f"  Raw codewords: {len(raw)} bytes")
        print(f"  First 20: {raw[:20]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        raw = None

    if raw is not None:
        print("\n=== Step 3: Deinterleave ===")
        try:
            blocks = deinterleave(raw, QR_VERSION, ECL_NAMES[ecl_idx])
            print(f"  Num blocks: {len(blocks)}")
            for i, blk in enumerate(blocks):
                print(f"    Block {i}: {len(blk.data)} data + {len(blk.ec)} EC bytes")
        except Exception as e:
            print(f"  FAILED: {e}")
            blocks = None

        if blocks is not None:
            print("\n=== Step 4: Reed-Solomon ===")
            for i, blk in enumerate(blocks):
                combined = list(blk.data) + list(blk.ec)
                num_ec = len(blk.ec)
                corrected = rs_decode(combined, num_ec)
                if corrected is None:
                    print(f"  Block {i}: FAILED")
                else:
                    print(f"  Block {i}: OK")

# Compare with a perfect synthetic QR code
print("\n=== Compare with perfect QR ===")
import qrcode

qr = qrcode.QRCode(
    version=QR_VERSION,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=0,
)
qr.add_data(QR_CONTENT)
qr.make(fit=True)

# Build matrix from qrcode library (True = dark)
modules = qr.modules
perfect = np.array([[modules[r][c] for c in range(N)] for r in range(N)], dtype=bool)

# Try to determine mask from the qrcode library
print(f"qrcode version: {qr.version}")
print(f"qrcode mask: {getattr(qr, 'mask_pattern', 'unknown')}")

print("\nPerfect top-left 7x7:")
for r in range(7):
    print("  " + " ".join("1" if perfect[r, c] else "0" for c in range(7)))

print("\nSampled top-left 7x7:")
for r in range(7):
    print("  " + " ".join("1" if matrix[r, c] else "0" for c in range(7)))

# Compare format info areas
print("\nPerfect row 8, cols 0-8:")
print("  " + " ".join("1" if perfect[8, c] else "0" for c in range(9)))
print("Sampled row 8, cols 0-8:")
print("  " + " ".join("1" if matrix[8, c] else "0" for c in range(9)))

print("\nPerfect col 8, rows 0-8:")
for r in range(9):
    print(f"  {perfect[r, 8]}")
print("Sampled col 8, rows 0-8:")
for r in range(9):
    print(f"  {matrix[r, 8]}")

# Count differences
print(
    f"\nTotal differences between sampled and perfect: {np.sum(matrix != perfect)} / {N * N}"
)

# Check if perfect decodes
from qr_reader.decoder.decoder import decode

try:
    text = decode(perfect)
    print(f"Perfect decode: {text}")
except Exception as e:
    print(f"Perfect decode FAILED: {e}")
