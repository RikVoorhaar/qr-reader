"""I2 — Identify B phantom sources in Cluster 3.

For each phantom peak (unmatched, normal >12° from GT normals, strong support),
dump segment endpoints and map to the QR module grid to identify the QR
structure that creates each phantom.
"""

from __future__ import annotations

import sys

import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.hough import hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from test_hough_harness import (
    _angular_distance_deg,
    _compute_finder_edges,
    _match_peak,
    _make_background,
    _run_pipeline_to_rois,
)

# Approximate module-coordinate from a position in the ROI
# We project image coords into the QR coordinate system using the
# inverse of the QR corner homography (simplified: bilinear).
def _qr_grid_position(
    x: float, y: float, corners_qr: dict, N: int, roi_offset: tuple[int, int]
) -> tuple[float, float]:
    """Return (module_col, module_row) fractional position for image point (x, y)."""
    # corners_qr is in image x/y coords, roi_offset is (row0, col0)
    col0, row0 = float(roi_offset[1]), float(roi_offset[0])
    
    TL = np.array(corners_qr["TL"], dtype=np.float64)
    TR = np.array(corners_qr["TR"], dtype=np.float64)
    BR = np.array(corners_qr["BR"], dtype=np.float64)
    BL = np.array(corners_qr["BL"], dtype=np.float64)
    
    # ROI-local x,y
    p = np.array([x + col0, y + row0], dtype=np.float64)
    
    # Simple bilinear: solve for (u,v) such that:
    # p = TL + u*(TR-TL) + v*(BL-TL) + u*v*(TR+BL-TL-BR)
    # Use iterative approximation since this is a perspective mapping
    # First approximation: linear (ignoring u*v term)
    dX = TR - TL
    dY = BL - TL
    A = np.column_stack([dX, dY])
    try:
        u, v = np.linalg.lstsq(A, p - TL, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (-1, -1)
    
    # Scale to module coords
    mod_col = u * N
    mod_row = v * N
    return (mod_col, mod_row)


def main():
    print("=" * 70)
    print("I2: B phantom source identification")
    print("=" * 70)

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

    background = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, CONFIG, background)

    N = metadata["N"]
    corners_qr = metadata["corners_qr"]

    roi_results = _run_pipeline_to_rois(image)

    for roi, nms, angle, bbox, ci in roi_results:
        normals, rhos, scores = hough_vote_peaks(nms, angle)

        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        # Pre-compute GT normals
        gt_normals = np.array(
            [e["normal"] for e in gt_edges if e["segment"] is not None]
        )

        print(f"\n{'='*70}")
        print(f"Cluster {ci}: scanning for phantoms")
        print(f"{'='*70}")
        print(f"  ROI bbox: rows=[{bbox[0]},{bbox[1]}) cols=[{bbox[2]},{bbox[3]})")
        print(f"  QR version={metadata['version']}, N={N} modules")
        print(f"  corners_qr: TL={corners_qr['TL']} TR={corners_qr['TR']} "
              f"BR={corners_qr['BR']} BL={corners_qr['BL']}")
        print(f"  {len(normals)} Hough peaks")

        phantom_count = 0
        for i in range(len(normals)):
            # Check if matched to any GT edge
            matched = any(
                gt["segment"] is not None
                and _match_peak(gt, normals[[i]], rhos[[i]]) >= 0
                for gt in gt_edges
            )
            if matched:
                continue

            # Check if normal is close to any GT normal
            if len(gt_normals) > 0:
                min_ang = min(
                    _angular_distance_deg(normals[i], gn) for gn in gt_normals
                )
                if min_ang < 12.0:
                    continue

            seg = refine_line(
                normals[i],
                float(rhos[i]),
                float(scores[i]),
                nms,
                angle,
                gap_tolerance=2.0,
                distance_thresh=1.5,
            )
            if np.all(seg.endpoints == 0):
                continue

            ys, xs = np.nonzero(np.asarray(nms))
            strengths = nms[ys, xs]
            points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
            dists = np.abs(points @ seg.normal - seg.rho)
            mask = dists < 1.5
            mean_str = float(strengths[mask].mean()) if np.sum(mask) > 0 else 0.0

            if mean_str <= 400:
                continue

            phantom_count += 1
            ang = np.rad2deg(np.arctan2(normals[i][1], normals[i][0])) % 180
            n_support = int(np.sum(mask))

            print(f"\n  --- Phantom P{i} ---")
            print(f"  Peak: θ={ang:.1f}° ρ={rhos[i]:.1f} score={scores[i]:.0f}")
            print(f"  Segment: ({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→"
                  f"({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f}) "
                  f"span={np.linalg.norm(seg.endpoints[1]-seg.endpoints[0]):.1f}px")
            print(f"  Normal: ({normals[i][0]:.4f},{normals[i][1]:.4f})")
            print(f"  Support: {n_support} pixels, mean NMS={mean_str:.1f}")

            # Map segment endpoints and midpoint to QR module coordinates
            ep0 = seg.endpoints[0]
            ep1 = seg.endpoints[1]
            mid = (ep0 + ep1) / 2

            m_ep0 = _qr_grid_position(ep0[0], ep0[1], corners_qr, N, (bbox[0], bbox[2]))
            m_ep1 = _qr_grid_position(ep1[0], ep1[1], corners_qr, N, (bbox[0], bbox[2]))
            m_mid = _qr_grid_position(mid[0], mid[1], corners_qr, N, (bbox[0], bbox[2]))

            print(f"  QR module coords (col, row):")
            print(f"    endpoint 0: ({m_ep0[0]:.1f}, {m_ep0[1]:.1f})")
            print(f"    endpoint 1: ({m_ep1[0]:.1f}, {m_ep1[1]:.1f})")
            print(f"    midpoint:   ({m_mid[0]:.1f}, {m_mid[1]:.1f})")

            # Classify based on position:
            # - Timing patterns run between finder patterns (near rows/cols 0, 6, N-7, N-1)
            # - Alignment patterns at specific positions
            # - Format info at rows 0..5, cols 0..5 etc.
            c, r = m_mid
            classifications = []
            
            # Check if near a finder pattern region (corners)
            finder_zone = 8  # 7 modules + 1 border
            if r < finder_zone and c < finder_zone:
                classifications.append("TL_FINDER_INTERIOR")
            elif r < finder_zone and c > N - finder_zone:
                classifications.append("TR_FINDER_INTERIOR")
            elif r > N - finder_zone and c < finder_zone:
                classifications.append("BL_FINDER_INTERIOR")
            
            # Timing patterns: row 6 or col 6
            if abs(r - 6) < 2 or abs(c - 6) < 2:
                classifications.append("TIMING_PATTERN")
            
            # Format info: around finder patterns, rows 0-5, cols 0-5
            if (r < 6 and c < finder_zone) or (c < 6 and r < finder_zone):
                classifications.append("FORMAT_INFO")
            
            # Version info: v >= 7 has 3x6 6x3 blocks at TL-TR and TL-BL corners
            if N >= 41:  # version 7+
                if (r < 6 and c > N - 11) or (c < 6 and r > N - 11):
                    classifications.append("VERSION_INFO")
            
            # Check if close to the border
            if r < 1 or r > N - 2 or c < 1 or c > N - 2:
                classifications.append("QR_BORDER")
            
            # Alignment pattern centers (for v12, N=53):
            # Alignment pattern positions for v12: [6, 30, 50] (approx)
            align_positions = {
                6: [6],
                7: [6, 22, 38],
                8: [6, 24, 42],
                9: [6, 26, 46],
                10: [6, 28, 48],
                11: [6, 30, 50],
                12: [6, 30, 50],
                13: [6, 34, 54],
                14: [6, 26, 46, 66],
                # etc.
            }
            v = metadata["version"]
            if v in align_positions:
                for ap in align_positions[v]:
                    if abs(c - ap) < 7 and abs(r - ap) < 7:
                        classifications.append(f"ALIGNMENT_PATTERN @({ap},{ap})")
            
            # Check if this aligns with a data module boundary
            # (generally the whole QR has module boundaries, but check if it's
            # in the data region away from functional patterns)
            if not classifications:
                classifications.append("DATA_REGION")
            
            print(f"  Classification: {', '.join(classifications)}")
            
            # Print the segment line equation in QR module space
            # to understand orientation relative to QR grid
            print(f"  Segment geometry in QR module grid:")
            print(f"    ROI origin (col offset, row offset): ({bbox[2]}, {bbox[0]})")
            print(f"    Approx image position of (col=0,row=0) "
                  f"QR corner: ({corners_qr['TL'][0]:.0f},{corners_qr['TL'][1]:.0f})")

        print(f"\n  Cluster {ci}: {phantom_count} phantoms total")

    print("\n" + "=" * 70)
    print("I2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
