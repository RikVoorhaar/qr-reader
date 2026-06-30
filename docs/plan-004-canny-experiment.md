# Plan 004 — Canny-based corner finding experiment

> Derived from [issue #4](https://github.com/RikVoorhaar/qr-reader/issues/4).

## Goal

Test an edge-based alternative to the flood-fill corner-finding path. The current
pipeline uses `region_fill_wave_front` → `region_boundary_8` →
`boundary_connected_components_ndimage` → `angular_nms_top_radial_indices`.
Flood fill is the dominant cost. Edge extraction + Hough voting may be faster.

## Implementation Plan

### Phase 1 — ROI module (`detector/roi.py`)

Two functions:

1. **`cluster_to_bbox(cluster, scale=1.5)`** → `(r0, r1, c0, c1)`
2. **`cutout(image, bbox)`** → `np.ndarray`

**Status: ✅ Done**

### Phase 2 — Edge extraction (`detector/edges.py`) + diagnostic script

Pipeline: Gaussian blur → Sobel (L2 magnitude) → atan2 → interpolated NMS.

- `extract_thin_edges(roi, blur_sigma=1.0)` → `(nms, angle)` in `detector/edges.py`
- Diagnostic script `scripts/full-pipeline-canny.py` displays 4 subplots per cluster:
  grayscale cutout, raw L2 magnitude, NMS-thinned edges, edge angle (mod π).

**Status: ✅ Done**

### Phase 3 — Hough line extraction (`detector/hough.py`) + diagnostic visualization

Two independent functions, tested individually:

1. **`hough_vote_peaks(nms, angle, …)`** → `(normals, rhos, scores)`
   - One-theta-per-edge-pixel gradient-guided Hough voting (pixel coords).
   - Smear-suppression peak NMS in accumulator, thresholded.
   - Returns peaks already converted to geometric form (no binning leakage).

2. **`refine_line(normal, rho, nms, angle, …)`** → `LineSegment`
   - Collects edge pixels near candidate line, fits weighted TLS.
   - Finds longest contiguous support run with gap bridging (`gap_tolerance` px).
   - Returns `endpoints` as projected segment extent, `vote_score` from Hough peak.

Diagnostic script `scripts/full-pipeline-canny.py` extended to overlay detected
line segments on the grayscale cutout per cluster.

**Status: 🔨 In Progress**

### Phase 4 (future) — Hough corner extraction

- Select 4-line quadrilateral from `LineSegment` list
- Intersect segments → finder corner points → feed into existing finder-pattern pipeline

## Design Decisions (Hough)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Module | `detector/hough.py` | Separate concern from edges and ROI |
| Voting input | `(nms, angle)` from `edges.py` | `angle = atan2(gy, gx)` already computed; no need for raw gx/gy |
| Voting algorithm | One-theta-per-edge-pixel | Simple and fast; NMS edges are clean and straight |
| Coordinate convention | Pixel coords (origin top-left of ROI) | Simplest for visualization and implementation |
| Peak extraction | Geometric `(normals, rhos, scores)` | No binning knowledge leaks into refinement |
| Accumulator return | Not returned from `hough_vote_peaks` | Only needed for debugging; caller can re-run if needed |
| Line refinement | Weighted TLS (edge-strength weights) | More robust than unweighted |
| Segment endpoints | Projected extent of longest contiguous run | Gap-bridging tolerance parameterized in px |
| `LineSegment` location | In `hough.py` | Only consumer for now |
| Edge re-extraction | `refine_line` extracts its own edge pixels | Cleaner interface; ROI is small so cost is negligible |
| NMS radii in Hough | `nms_radius_theta=3`, `nms_radius_rho=6` | Conservative suppression to avoid duplicate line registration |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Padding method | Proportional to cluster size | Adapts to varying QR versions/scales |
| Default scale | 1.5 | Wide enough to capture the full alignment pattern region |
| ROI types | Bounding box + separate cutout function | Separation of concerns; cutout handles clamping |
| Module location | `detector/roi.py` | Part of detection pipeline, not just a script utility |
| Canny input | Grayscale | Much better edge response than binary |
| Visualization layout | One figure per cluster, 4 subplots | Shows raw→NMS→angle pipeline stages |
| Edge clamping | In `cutout()`, not `cluster_to_bbox()` | Keeps bbox computation pure |
| Edge extraction | `detector/edges.py` | Separate from ROI; reusable by future Hough module |
| Blur | `scipy.ndimage.gaussian_filter(σ=1.0)` | No OpenCV dependency; consistent with detector module deps |
| Gradients | `ndimage.sobel`, L2 magnitude | L2 for accurate angle; atan2 on whole image for interpolation NMS |
| NMS type | Interpolated along exact gradient direction | Better thinning for off-axis edges than 4-sector quantization |
| Thresholding | Deferred to downstream consumer | NMS output is a dense float image; caller thresholds as needed |
| Border pixels | Suppressed in NMS | Gradient-direction neighbors fall outside image; border edges are noise |
| Hough voting method | One-theta gradient-guided (not full angular sweep) | Clean edges; design allows switching to windowed voting later |

## Success Criteria (from issue #4)

- Edge-based path is measurably faster per cluster than flood fill
- No meaningful regression in corner accuracy or end-to-end recall
