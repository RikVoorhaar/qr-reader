"""I3 — Profile A gap causes: structural vs dropout classification.

For each A-failing edge (TL_top in C1, BL_bottom in C2), inspect the NMS
content along the segment projection to classify each gap as:
  - Structural: NMS pixels exist at wrong angle (QR module crossing)
  - Dropout:    No NMS pixels at any angle (genuine edge suppression)
  - Mixed:      Some of each
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

# Use same gap tolerance as the fixture
GAP_TOL = 2.0
DIST_THRESH = 1.5
# Angle tolerance for "same normal direction"
ANGLE_TOL_DEG = 20.0


def _classify_gaps(
    seg, nms, angle, gt_seg, label,
) -> None:
    """Classify each gap in the segment's support as structural or dropout."""
    direction = np.array([-seg.normal[1], seg.normal[0]], dtype=np.float64)
    
    ys, xs = np.nonzero(np.asarray(nms))
    strengths = nms[ys, xs]
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    dists = np.abs(points @ seg.normal - seg.rho)
    mask = dists < DIST_THRESH
    
    support_pts = points[mask]
    support_str = strengths[mask]
    
    if len(support_pts) == 0:
        print(f"  No support pixels found")
        return
    
    # Project onto line direction
    proj = support_pts @ direction
    sort_idx = np.argsort(proj)
    proj_sorted = proj[sort_idx]
    str_sorted = support_str[sort_idx]
    pts_sorted = support_pts[sort_idx]
    ang_sorted = np.zeros(len(sort_idx))
    for j, idx in enumerate(sort_idx):
        py, px = int(support_pts[idx][1]), int(support_pts[idx][0])
        if 0 <= py < angle.shape[0] and 0 <= px < angle.shape[1]:
            ang_sorted[j] = angle[py, px]
    
    # Identify gaps ≥ GAP_TOL
    gaps = []
    for i in range(1, len(proj_sorted)):
        gap = float(proj_sorted[i] - proj_sorted[i - 1])
        if gap >= GAP_TOL:
            gaps.append({
                "gap_start": float(proj_sorted[i - 1]),
                "gap_end": float(proj_sorted[i]),
                "gap_width": gap,
            })
    
    print(f"\n  {'='*50}")
    print(f"  {label}: {len(gaps)} gaps ≥ {GAP_TOL}px")
    print(f"  {'='*50}")
    print(f"  Segment normal=({seg.normal[0]:.3f},{seg.normal[1]:.3f}) "
          f"rho={seg.rho:.1f}")
    print(f"  Endpoints: ({seg.endpoints[0][0]:.1f},{seg.endpoints[0][1]:.1f})→"
          f"({seg.endpoints[1][0]:.1f},{seg.endpoints[1][1]:.1f})")
    print(f"  GT endpoints: ({gt_seg[0][0]:.1f},{gt_seg[0][1]:.1f})→"
          f"({gt_seg[1][0]:.1f},{gt_seg[1][1]:.1f})")
    print(f"  Total support: {len(support_pts)} pixels, "
          f"mean strength={float(support_str.mean()):.1f}")
    
    # GT normal angle for comparison
    gt_normal = np.array([-direction[1], direction[0]], dtype=np.float64)  # approx
    # Get the actual GT normal from the edge (correct: perpendicular to direction)
    # direction = (-seg.normal[1], seg.normal[0])
    # So seg.normal is the line normal. The edge normal is the same if edge pixels are along the line.
    
    structural_count = 0
    dropout_count = 0
    
    for gi, gap in enumerate(gaps):
        gs = gap["gap_start"]
        ge = gap["gap_end"]
        gw = gap["gap_width"]
        
        # Sample the gap region: find all NMS pixels whose projection falls in [gs, ge]
        # and check if any have a "correct" edge-normal angle
        gap_pixels = []
        nms_gap_pixels = []
        
        # Strategy: check perpendicular cross-section at multiple sample points
        # along the gap. At each point, look ±distance_thresh perpendicular from
        # the line, and check NMS pixel values and angles.
        
        n_samples = max(3, int(gw / 2))
        sample_pts_proj = np.linspace(gs + 0.5, ge - 0.5, n_samples)
        
        nms_found = 0
        correct_angle_found = 0
        
        for sp in sample_pts_proj:
            # Point on the line at this projection
            p_on_line = seg.endpoints[0] + direction * (sp - float(proj_sorted[0]))
            
            # Check perpendicular cross-section: walk ±DIST_THRESH along the normal
            for sign in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                px = int(round(p_on_line[0] + sign * seg.normal[0] * DIST_THRESH))
                py = int(round(p_on_line[1] + sign * seg.normal[1] * DIST_THRESH))
                
                if 0 <= py < nms.shape[0] and 0 <= px < nms.shape[1]:
                    if nms[py, px] > 0:
                        nms_found += 1
                        # Check angle of this pixel vs segment normal
                        pix_ang = float(angle[py, px])
                        seg_ang = np.arctan2(seg.normal[1], seg.normal[0])
                        ang_diff = abs(np.rad2deg((pix_ang - seg_ang) % np.pi))
                        ang_diff = min(ang_diff, 180 - ang_diff)
                        if ang_diff < ANGLE_TOL_DEG:
                            correct_angle_found += 1
                        else:
                            # Wrong angle NMS pixel — check if it's from a known structure
                            pass
        
        total_checks = n_samples * 5  # 5 cross-section samples per sample point
        
        if nms_found == 0:
            classification = "DROPOUT"
            dropout_count += 1
        elif correct_angle_found == 0 and nms_found > 0:
            classification = "STRUCTURAL (wrong angle only)"
            structural_count += 1
        elif correct_angle_found > 0 and nms_found > correct_angle_found:
            classification = "MIXED"
            structural_count += 1
            dropout_count += 1
        elif correct_angle_found > 0 and nms_found == correct_angle_found:
            classification = "PARTIAL (correct angle, weak)"
            structural_count += 1
        else:
            classification = "DROPOUT"
            dropout_count += 1
        
        print(f"\n    Gap {gi}: {gw:.1f}px  [{gs:.1f} → {ge:.1f}]")
        print(f"      NMS pixels in gap cross-section: {nms_found}/{total_checks} samples")
        print(f"      Correct-angle NMS: {correct_angle_found}/{nms_found}")
        print(f"      Classification: {classification}")
    
    print(f"\n  Summary: {structural_count} structural, {dropout_count} dropout, "
          f"{len(gaps)} total gaps")


def main():
    print("=" * 70)
    print("I3: A gap profiling — structural vs dropout classification")
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

    roi_results = _run_pipeline_to_rois(image)

    for roi, nms, angle, bbox, ci in roi_results:
        normals, rhos, scores = hough_vote_peaks(nms, angle)

        gt_edges = _compute_finder_edges(
            metadata,
            roi_offset=(bbox[0], bbox[2]),
            roi_shape=roi.shape,
        )

        for gt in gt_edges:
            if gt["segment"] is None:
                continue

            match_idx = _match_peak(gt, normals, rhos)
            if match_idx < 0:
                continue  # not an A failure (D failure — skip)

            seg = refine_line(
                normals[match_idx],
                float(rhos[match_idx]),
                float(scores[match_idx]),
                nms,
                angle,
                gap_tolerance=GAP_TOL,
                distance_thresh=DIST_THRESH,
            )

            gt_seg = gt["segment"]
            direction = np.array([-gt["normal"][1], gt["normal"][0]], dtype=np.float64)
            gt_proj = gt_seg @ direction
            gt_span_dir = abs(float(gt_proj[1] - gt_proj[0]))

            degenerate = np.all(seg.endpoints == 0)
            if degenerate:
                ep_proj = np.array([0.0, 0.0])
                seg_span = 0.0
            else:
                ep_proj = seg.endpoints @ direction
                seg_span = abs(float(ep_proj[1] - ep_proj[0]))

            is_A = degenerate or seg_span < 0.8 * gt_span_dir

            if not is_A:
                continue  # not an A failure

            print(f"\n  {'='*60}")
            print(f"  Cluster {ci}, {gt['label']}: FAILURE A")
            ang = np.rad2deg(np.arctan2(gt["normal"][1], gt["normal"][0])) % 180
            print(f"  GT: θ={ang:.1f}° ρ={gt['rho']:.1f} span={gt_span_dir:.1f}px")
            print(f"  Segment: span={seg_span:.1f}px (ratio={seg_span/gt_span_dir*100:.0f}%)")

            _classify_gaps(seg, nms, angle, gt_seg, f"C{ci} {gt['label']}")

    print("\n" + "=" * 70)
    print("I3 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
