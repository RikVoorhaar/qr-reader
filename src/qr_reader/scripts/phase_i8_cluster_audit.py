"""I8 — Cluster finder pattern audit.

For each cluster in the v12-default pipeline, this script:
1. Runs corner detection (region fill → boundary → components → angular NMS).
2. Extracts finder patterns from the corners.
3. Computes GT finder edges in the ROI.
4. Runs Hough and detects B phantoms.
5. Reports: cluster index, # finder patterns, # GT edges in ROI,
   # Hough peaks, # B phantoms.

Key question: Does C3 (the cluster with all 5 B phantoms) have any
finder patterns?  If not, the B phantoms are a test artifact.
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.corner import angular_nms_top_radial_indices
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.finder_pattern import (
    extract_finder_patterns,
    find_all_associations,
    find_triplets,
)
from qr_reader.detector.hough import hough_vote_peaks, refine_line
from qr_reader.detector.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from test_hough_harness import (
    _angular_distance_deg,
    _compute_finder_edges,
    _make_background,
    _match_peak,
)

CONFIG = AugmentationConfig(
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


def _run_corners_for_cluster(
    img_binary: np.ndarray,
    cluster_ci: int,
    clusters: list,
) -> list[tuple[int, np.ndarray]]:
    """Run corner detection for a single cluster, mimicking detector.py."""
    cluster = clusters[cluster_ci]
    seed_row = int(cluster.row)
    seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2)

    region_mask = region_fill_wave_front(
        np.asarray(img_binary), seed_row, seed_col
    )
    boundary = region_boundary_8(region_mask)
    components = boundary_connected_components_ndimage(np.asarray(boundary))

    angular_distance_nms = 10 * 2 * np.pi / 360
    corners_list: list[tuple[int, np.ndarray]] = []

    for comp in components:
        comp_arr = np.asarray(comp, dtype=np.float64)
        if comp_arr.shape[0] < 4:
            continue
        centroid_i = comp_arr.mean(axis=0)
        rd = np.linalg.norm(comp_arr - centroid_i, axis=1)
        ang = np.arctan2(
            comp_arr[:, 1] - centroid_i[1],
            comp_arr[:, 0] - centroid_i[0],
        )
        try:
            idx = angular_nms_top_radial_indices(
                rd, ang, angular_nms_rad=angular_distance_nms, k=4
            )
        except ValueError:
            continue
        corners_list.append((cluster_ci, comp_arr[idx]))

    return corners_list


def _count_phantoms(
    normals: np.ndarray,
    rhos: np.ndarray,
    scores: np.ndarray,
    nms: np.ndarray,
    angle: np.ndarray,
    gt_edges: list[dict],
) -> int:
    gt_normals = np.array(
        [e["normal"] for e in gt_edges if e["segment"] is not None]
    )
    phantom_count = 0
    for i in range(len(normals)):
        matched = any(
            gt["segment"] is not None
            and _match_peak(gt, normals[[i]], rhos[[i]]) >= 0
            for gt in gt_edges
        )
        if matched:
            continue
        if len(gt_normals) > 0:
            min_ang = min(
                _angular_distance_deg(normals[i], gn) for gn in gt_normals
            )
            if min_ang < 12.0:
                continue
        seg = refine_line(
            normals[i], float(rhos[i]), float(scores[i]), nms, angle,
            gap_tolerance=2.0, distance_thresh=1.5,
        )
        if np.all(seg.endpoints == 0):
            continue
        ys, xs = np.nonzero(np.asarray(nms))
        strengths = nms[ys, xs]
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        dists = np.abs(points @ seg.normal - seg.rho)
        mask = dists < 1.5
        mean_str = float(strengths[mask].mean()) if np.sum(mask) > 0 else 0.0
        if mean_str > 400:
            phantom_count += 1
    return phantom_count


def main() -> None:
    print("=" * 70)
    print("I8 — Cluster finder pattern audit")
    print("=" * 70)
    print()

    bg = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, CONFIG, bg)

    # Grayscale
    if image.ndim == 3:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image)

    # Binarize
    img_binary = binarize_image(gray)

    # Alignment patterns → clusters
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(
        img_binary, max_error
    )
    clusters = cluster_candidates(rows_valid, cols_valid_all)

    print(f"Total clusters: {len(clusters)}")
    print()

    # Table header
    print(f"  {'Ci':>2}  {'FP':>2}  {'Assoc':>5}  {'Trip':>4}  "
          f"{'GT_in':>5}  {'peaks':>5}  {'B_phant':>7}  "
          f"{'NMS%':>5}  ROI_shape")
    print(f"  {'-'*2}  {'-'*2}  {'-'*5}  {'-'*4}  "
          f"{'-'*5}  {'-'*5}  {'-'*7}  "
          f"{'-'*5}  {'-'*10}")

    total_phantoms = 0
    phantom_free_clusters = 0
    fp_clusters = set()
    non_fp_clusters = set()

    # Collect all corners first (like detector.py does)
    all_corners: list[tuple[int, np.ndarray]] = []
    for ci in range(len(clusters)):
        corners_for_ci = _run_corners_for_cluster(img_binary, ci, clusters)
        all_corners.extend(corners_for_ci)

    # Extract finder patterns from all corners at once
    fps = extract_finder_patterns(all_corners)
    fp_cluster_indices = {fp.cluster_idx for fp in fps}

    for ci in range(len(clusters)):
        # How many finder patterns in this cluster?
        n_fp_in_ci = sum(1 for fp in fps if fp.cluster_idx == ci)

        # Associations and triplets involving this cluster's FPs
        ci_fps = [fp for fp in fps if fp.cluster_idx == ci]
        if len(ci_fps) >= 2:
            associations = find_all_associations(ci_fps)
            triplets = find_triplets(ci_fps, associations)
            n_assoc = len(associations)
            n_trip = len(triplets)
        else:
            associations = []
            triplets = []
            n_assoc = 0
            n_trip = 0

        # ROI + Hough
        bbox = cluster_to_bbox(clusters[ci], scale=1.5)
        roi = cutout(gray, bbox)
        nms, angle = extract_thin_edges(roi, blur_sigma=1.0) if roi.size > 0 else (
            np.array([]), np.array([])
        )

        if roi.size == 0 or nms.size == 0:
            print(f"  C{ci:<2}  {n_fp_in_ci:>2}  {n_assoc:>5}  {n_trip:>4}  "
                  f"{'N/A':>5}  {'N/A':>5}  {'N/A':>7}  {'N/A':>5}  empty ROI")
            continue

        # GT edges
        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )
        n_gt = sum(1 for gt in gt_edges if gt["segment"] is not None)

        # Hough
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        n_peaks = len(normals)

        # B phantoms
        n_phantom = _count_phantoms(
            normals, rhos, scores, nms, angle, gt_edges
        )

        # NMS density
        nms_nonzero = np.count_nonzero(nms)
        nms_density = nms_nonzero / (nms.shape[0] * nms.shape[1]) * 100 if nms.size > 0 else 0

        total_phantoms += n_phantom
        if n_phantom == 0:
            phantom_free_clusters += 1
        if n_fp_in_ci > 0:
            fp_clusters.add(ci)
        else:
            non_fp_clusters.add(ci)

        print(f"  C{ci:<2}  {n_fp_in_ci:>2}  {n_assoc:>5}  {n_trip:>4}  "
              f"{n_gt:>5}  {n_peaks:>5}  {n_phantom:>7}  "
              f"{nms_density:>4.1f}%  {nms.shape}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Clusters with finder patterns: {sorted(fp_clusters)}")
    print(f"  Clusters without finder patterns: {sorted(non_fp_clusters)}")
    print(f"  Total phantoms: {total_phantoms}")
    print(f"  Phantom-free clusters: {phantom_free_clusters}/{len(clusters)}")
    print()

    # Check the C3 hypothesis
    n_phantom_in_non_fp = 0
    n_phantom_in_fp = 0
    for ci in range(len(clusters)):
        bbox = cluster_to_bbox(clusters[ci], scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0:
            continue
        nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )
        normals, rhos, scores = hough_vote_peaks(nms, angle)
        n_phant = _count_phantoms(normals, rhos, scores, nms, angle, gt_edges)
        if ci in non_fp_clusters:
            n_phantom_in_non_fp += n_phant
        else:
            n_phantom_in_fp += n_phant

    print(f"  Phantoms in non-finder clusters: {n_phantom_in_non_fp}")
    print(f"  Phantoms in finder clusters: {n_phantom_in_fp}")
    print()

    if len(non_fp_clusters) > 0 and n_phantom_in_non_fp == total_phantoms:
        print("  ✓ ALL phantoms are in non-finder clusters.")
        print("  → Phase 9 (skip non-finder clusters) would eliminate ALL B failures.")
    elif len(non_fp_clusters) > 0 and n_phantom_in_non_fp > 0:
        print(f"  ≈ {n_phantom_in_non_fp}/{total_phantoms} phantoms in non-finder clusters.")
        print("  → Phase 9 would eliminate MOST B failures.")
    elif total_phantoms == 0:
        print("  ✓ No phantoms found in any cluster.")
    else:
        print("  ✗ Phantoms are NOT confined to non-finder clusters.")
        print("  → Phase 9 won't help. Need a different B fix.")

    print()
    print("End I8 — Cluster finder pattern audit")


if __name__ == "__main__":
    main()
