# Plan 004 — Canny-based corner finding experiment

> Derived from [issue #4](https://github.com/RikVoorhaar/qr-reader/issues/4).

## Goal

Test an edge-based alternative to the flood-fill corner-finding path. The current
pipeline uses `region_fill_wave_front` → `region_boundary_8` →
`boundary_connected_components_ndimage` → `angular_nms_top_radial_indices`.
Flood fill is the dominant cost. Canny + contours may be faster.

## Implementation Plan

### Phase 1 — ROI module (`detector/roi.py`)

Two functions:

1. **`cluster_to_bbox(cluster, scale=1.5)`** → `(r0, r1, c0, c1)`

   - Center: `(cluster.row, (cols[2] + cols[3]) / 2)`
   - Half-extent: `max((cols[5] - cols[0]) / 2, height / 2)` where height = `cols[3] - cols[2]`
   - Bbox: center ± `scale * half_extent` in both directions
   - Returns integer coordinates (not clamped)

2. **`cutout(image, bbox)`** → `np.ndarray`

   - Clamps bbox to image bounds
   - Returns sub-image slice

### Phase 2 — Diagnostic script (`scripts/full-pipeline-canny.py`)

- Pipeline up to and including `cluster_candidates`
- For each cluster: compute bbox → cutout (grayscale) → OpenCV Canny
- One figure per cluster, two subplots: cutout (left) + Canny edges (right)

### Phase 3 (future) — Contour corner extraction

- `cv2.findContours` on the Canny edge image
- Fit quadrilateral or run angular NMS on the best contour
- Compare against flood-fill corner placement

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Padding method | Proportional to cluster size | Adapts to varying QR versions/scales |
| Default scale | 1.5 | Wide enough to capture the full alignment pattern region |
| ROI types | Bounding box + separate cutout function | Separation of concerns; cutout handles clamping |
| Module location | `detector/roi.py` | Part of detection pipeline, not just a script utility |
| Canny input | Grayscale | Much better edge response than binary |
| Visualization layout | One figure per cluster, 2 subplots | Clean comparison; avoids tall figures |
| Edge clamping | In `cutout()`, not `cluster_to_bbox()` | Keeps bbox computation pure; clamping is a display concern |

## Success Criteria (from issue #4)

- Edge-based path is measurably faster per cluster than flood fill
- No meaningful regression in corner accuracy or end-to-end recall
