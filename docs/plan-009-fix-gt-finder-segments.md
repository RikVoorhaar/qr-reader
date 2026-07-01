# Plan: Fix GT Finder-Pattern Segments (12 per finder × 3 finders = 36 total)

## Problem

`_compute_finder_edges` in the harness, test harness, and pipeline scripts draws
**8 incorrect** GT segments:
- Only 2 edges per finder (top + left), should be 12.
- `TR_right` = diagonal `TR → BR` instead of true right edge `TR → BR`
  parallel to left edge.
- `BL_bottom` = diagonal `BL → BR` instead of true bottom edge `BL → BR`
  parallel to top edge.
- Module-offset boundaries (outer/inner rings) are missing entirely.

## Required output

Each finder pattern (TL, TR, BL) has a 7×7 module grid.  The four sides each
have 3 module-boundary lines:

| Side     | Module offsets | Direction vector        |
|----------|---------------|-------------------------|
| Top      | 0, 1, 2       | `TR - TL` for TL-finder |
| Bottom   | 5, 6, 7       | `TR - TL`               |
| Left     | 0, 1, 2       | `BL - TL`               |
| Right    | 5, 6, 7       | `BL - TL`               |

For TL-finder, `direction_top_bottom = TR - TL`, `direction_left_right = BL - TL`.
For TR-finder, `direction_top_bottom = TL - TR`, `direction_left_right = BR - TR`.
For BL-finder, `direction_top_bottom = BR - BL`, `direction_left_right = TL - BL`.

Line at offset `k`: passes through `origin + k/N * direction_vector`, with
normal perpendicular to direction_vector.  Canonicalise rho ≥ 0.

Boundary semantics:
- `k=0`: outer dark boundary (outside of finder)
- `k=1`: light ring boundary (dark→light transition)
- `k=2`: center square boundary (light→dark transition)

## Implementation steps

### Step 0 — Visual verification script (HARD GATE)

**File:** `src/qr_reader/scripts/verify_finder_segments.py`

Notebook-style (`# %%` cells), standalone, no dependencies on the rest of the
pipeline except `generate_sample` and `cluster_to_bbox`/`cutout`.

Cells:
1. Generate v12 image with `generate_sample` (same config as harness)
2. Compute 36 segments from `metadata["corners_qr"]` in global coordinates
3. Full-image plot: all 36 segments overlaid on grayscale, colour-coded by
   finder (TL=blue, TR=green, BL=orange), labeled
4. Run clustering pipeline (`binarize → alignment → cluster`), then per-ROI:
   a. Show ROI grayscale
   b. Overlay only the GT segments that intersect this ROI
   c. Label each segment
5. Print a summary for each finder: `{finder} top[0,1,2]: θ=...° bottom[5,6,7] ...`

**GATE:** Do NOT proceed past this step until the user has run the script and
confirmed all 36 segments look correct.  No tests, no harness changes, no
re-baselining until visual verification passes.

### Step 1 — Fix `_compute_finder_edges` in the harness

**File:** `src/qr_reader/scripts/run_hough_ablation.py`

Replace the current 8-edge `_compute_finder_edges` with the correct 36-segment
version.  Same for the `_edge_coverage_fraction` and `_edge_intersects_roi`
callers — they should now iterate over 12 edges per finder.

The function signature stays the same: `(metadata, roi_offset, roi_shape) →
list[dict]` where each dict has `label`, `normal`, `rho`, `segment`.

Labels should follow the pattern `{finder}_{side}{offset}`.  Example:
`TL_top0`, `TL_top1`, `TL_top2`, `TL_left0`, ..., `TL_right7`.

### Step 2 — Fix `_compute_finder_edges` in the test harness

**File:** `src/qr_reader/tests/detector/test_hough_harness.py`

Same function duplicated there.  Replace with the identical implementation.

### Step 3 — Fix `_compute_finder_edges` in the pipeline script

**File:** `src/qr_reader/scripts/full-pipeline-canny.py`

Replace the 6-edge helper with the 36-segment version.

### Step 4 — Update failure classification

The `_classify_failures` function in the harness uses the number of GT edges
to compute peak hit rates.  With 36 GT edges instead of 8, the hit-rate
denominator changes but the logic stays the same.

The per-finder ROI intersection logic (`_assign_clusters_to_finders` in E1)
still works — each cluster maps to one finder.  But now each finder ROI may
contain up to 12 GT edges instead of 2–4.

### Step 5 — Re-baseline

Run `--mode baseline` with the new GT set.  Expected: much higher total counts
(because we now track 12 edges per finder instead of 2), but the A/B/C/D
*categories* should remain meaningful.  The v12-clean baseline must still show
zero failures.

### Step 6 — Re-run E3–E6 (or at minimum re-evaluate)

The E3–E6 best configs were chosen based on 8-edge GT data.  With 36 edges the
relative improvements may differ.  Priority: re-run E6 (support sweep) since
gap_tolerance=3 was the key A=0 finder — with 36 edges there may be new A
failures.

## Files affected

| File | Change |
|------|--------|
| `src/qr_reader/scripts/verify_finder_segments.py` | **NEW** — visual verification |
| `src/qr_reader/scripts/run_hough_ablation.py` | Replace `_compute_finder_edges` |
| `src/qr_reader/tests/detector/test_hough_harness.py` | Replace `_compute_finder_edges` |
| `src/qr_reader/scripts/full-pipeline-canny.py` | Replace `_compute_gt_edges` |
| `docs/plan-008-hough-ablation-sweeps.md` | Update prerequisites after re-baseline |
