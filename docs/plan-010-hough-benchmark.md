# Plan 010 — Hough Benchmark Script

> **Goal:** A standalone benchmark script that generates several QR images, runs
> the Hough pipeline on GT-derived ROI cutouts, and reports granular line- and
> segment-level TP/FN/FP metrics plus overlap quality at every stage. Replaces
> ad-hoc eyeballing with a reproducible before/after comparison harness.

## 1. Script layout

**File:** `src/qr_reader/scripts/hough_benchmark.py`
**Style:** notebook-style (`# %%` cells), default matplotlib backend, `plt.show()`
— human-inspectable, no files saved to disk unless `--csv` is passed.

### CLI

```
python -m qr_reader.scripts.hough_benchmark \
    [--cases v12-default,v12-clean,v5-default]  \
    [--seeds 42,43,44]                           \
    [--n-images 5]                               \
    [--csv out/bench.csv]                        \
    [--no-plots]                                 \
    [--hough-config default|e6best]
```

- `--cases`: comma-separated fixture case names (same `FIXTURE_SPECS` as
  `run_hough_ablation.py`).
- `--seeds`: override the per-case default seed; for each case × seed combo, a
  new image is generated (seed is fed to `np.random.default_rng`).
- `--n-images`: if `--seeds` not given, generate N seeds starting from the
  case's default seed (seed, seed+1, … seed+N-1).
- `--csv`: write a machine-readable summary CSV to the given path.
- `--no-plots`: skip all `plt.show()` calls (for CI / batch runs).
- `--hough-config`: which preset Hough parameter set to use (`default` =
  library defaults, `e6best` = the E6 best config from Plan 008).

## 2. Pipeline per image

For each `(case, seed)` pair:

1. **Generate image + metadata.**
   `rng = np.random.default_rng(seed)`
   `image, metadata = generate_sample(rng, config, background)`
   — reuses `FIXTURE_SPECS`, `_make_background` from `run_hough_ablation.py`.

2. **Extract clusters + ROIs.**
   Reuse `_run_pipeline_to_rois(image)` → `list[(roi, nms, angle, bbox, ci)]`.
   This gives us the same ROIs the real detector produces.

3. **Assign clusters to finders (GT-based).**
   For each cluster `ci`:
   - Compute cluster centre `(cx, cy) = (cluster.row, mid_cols)` in (row, col)
     → convert to (x, y).
   - Compute the three GT finder centres in image space via the metadata
     homography: project grid points `(3.5, 3.5)` for TL, `(N-3.5, 3.5)` for
     TR, `(3.5, N-3.5)` for BL through `H = estimate_homography_dlt(grid_4,
     corners_qr_4)`.
   - Greedy-assign each cluster to the nearest GT finder centre if within
     `2 * half_extent` px; else mark `NON_FINDER`.
   This is the same approach as `_assign_clusters_to_finders` in the audit
   mode, but cleaner and used for every run.

4. **Per-cluster Hough + refine.**
   For each cluster:
   - `gt_edges = _compute_finder_edges(metadata, roi_offset, roi_shape)` —
     12 edges for finder clusters, 0 for NON_FINDER (the homography projects
     outside the ROI, so all segments clip to `None`).
   - `normals, rhos, scores, acc_data = hough_vote_peaks(nms, angle,
     return_acc=True, **hough_kwargs)`.
   - For each peak: `seg = refine_line(normal, rho, score, nms, angle,
     **refine_kwargs)` → `LineSegment`.
   - Store all per-cluster arrays.

5. **Compute metrics** (§3 below).

6. **Plot** (§4 below), unless `--no-plots`.

## 3. Metrics

All metrics computed per cluster, then aggregated per image, then across all
images.  Reported both as console table and CSV row(s).

### 3.1 Hough-space (peak-level) — "is there a vote peak near each GT line?"

**Matching rule:** a GT edge `(n_gt, ρ_gt)` matches peak `i` if
`angular_distance(n_gt, n_i) ≤ angle_tol_deg` **and**
`|ρ_gt − ρ_i| ≤ ρ_tol`.  Tolerances default `5° / 5 px` (same as existing
`_match_peak`).  Among matches, pick the one with smallest
`ang_dist + ρ_dist`.

**Per GT edge (line):**
- `peak_hit: bool` — matched?
- `ang_err_deg: float` — angular distance to matched peak (NaN if miss).
- `rho_err_px: float` — |ρ_gt − ρ_match| (NaN if miss).
- `gt_bin_value: float` — accumulator value at the GT (θ, ρ) bin.
- `gt_bin_snrr: float` — `gt_bin_value / mean(acc[θ-window, :])` (signal-to-
  background ratio; window = ±5°).

**Peak-level counts:**
- `n_peak_TP` — peaks that matched a GT edge.
- `n_peak_FN` — GT edges with no matching peak.
- `n_peak_FP` — peaks that matched no GT edge **and** are not near any GT
  normal (min angular distance ≥ 12°).  (Peaks near a GT normal but outside
  the ρ tolerance are "near-miss FP" — reported separately as
  `n_peak_FP_near`.)

**Aggregate:**
- `peak_precision = TP / (TP + FP)`
- `peak_recall    = TP / (TP + FN)`
- `peak_F1`
- `mean_ang_err` (over TP only)
- `mean_rho_err` (over TP only)
- `mean_gt_bin_snrr` (over all GT edges with a segment)

### 3.2 Segment-level — "how well does the refined segment overlap the GT?"

Only for GT edges with `segment is not None` (visible in ROI).

**Matching rule:** first require a peak match (§3.1).  Then refine the peak to
a `LineSegment`.  A GT segment is a **segment TP** if:
- Refined segment is non-degenerate (`endpoints ≠ 0`), **and**
- 1-D IoU (see below) ≥ `0.3`.

**1-D IoU computation:**
1. `direction = gt_segment[1] − gt_segment[0]`, normalised.
2. Project both GT endpoints and both refined endpoints onto `direction`:
   `t_gt = [gt · dir]`, `t_seg = [seg · dir]` — two scalars each.
3. Sort each pair → two 1-D intervals `[t_gt_lo, t_gt_hi]` and
   `[t_seg_lo, t_seg_hi]`.
4. `inter = max(0, min(hi) − max(lo))`
5. `union = max(hi) − min(lo)`
6. `iou_1d = inter / union` (0 if union ≤ 0).

**Per TP segment:**
- `iou_1d: float` — 1-D interval IoU.
- `coverage_gt: float` — `inter / gt_span` (recall: how much of GT is
  covered by the detected segment).
- `coverage_seg: float` — `inter / seg_span` (precision: how much of the
  detected segment lies within GT span).
- `endpoint_err_px: float` — max over both refined endpoints of the distance
  to the nearest GT endpoint.
- `lateral_err_px: float` — mean perpendicular distance of the refined
  endpoints to the GT line (= |(endpoint − gt_start) · n_gt|).

**Segment-level counts:**
- `n_seg_TP` — GT segments with a matching detected segment (IoU ≥ 0.3).
- `n_seg_FN` — GT segments with no match (either no peak, or degenerate, or
  IoU < 0.3).
- `n_seg_FP` — detected segments (refined, non-degenerate) with no matching
  GT segment **and** angle ≥ 12° from all GT normals.  (Near-miss FPs
  reported separately as `n_seg_FP_near`.)

**Aggregate:**
- `seg_precision`, `seg_recall`, `seg_F1`
- `mean_iou_1d` (over TP only)
- `mean_coverage_gt` (over TP only)
- `mean_endpoint_err` (over TP only)
- `mean_lateral_err` (over TP only)
- `p05_iou_1d` (5th percentile — worst-case quality)

### 3.3 FP characterization

For every FP segment (peak or segment level):
- `is_duplicate: bool` — is there a TP segment on the same GT edge (i.e. the
  GT edge already had a TP match)?  Duplicates are "double-detection" noise,
  not truly phantoms.
- `is_on_non_finder: bool` — cluster is `NON_FINDER`.
- `support_length_px: float` — `||endpoints[1] − endpoints[0]||`.
- `mean_nms_strength: float` — mean NMS magnitude of support pixels.

Reported as a small table in the console output.

### 3.4 Per-finder and per-side breakdown

Metrics (TP/FN/FP, IoU, errors) are also broken down by:
- **Finder:** TL / TR / BL / NON_FINDER.
- **Side:** top / bot / left / right.
- **k (boundary):** 0 / 1 / 2 / 5 / 6 / 7.

This lets us see, e.g., "TL_left2 always misses" or "inner edges (k=1,6)
have lower IoU than outer edges (k=0,7)".

## 4. Visualizations

All plots use the default matplotlib backend + `plt.show()` (notebook-style).
One figure-set per cluster, per image.  Controlled by `--no-plots`.

### 4.1 ROI overlay (`fig_roi`)

Single axes, aspect-equal, y-inverted:
- Grayscale ROI image (light).
- GT segments: coloured solid lines (colour-coded by side: top=blue,
  bot=green, left=red, right=orange; alpha varies by k: outer=1.0,
  inner=0.5).
- Detected segments: cyan solid lines with endpoint markers.
- FP segments: magenta dashed lines.
- Legend + title with cluster ID, finder assignment, TP/FN/FP counts.

### 4.2 Hough accumulator (`fig_hough`)

Single axes:
- Accumulator heatmap (`acc.T`, "inferno" cmap, `vmax = 0.3 * acc.max()`).
- GT edges: coloured circles at `(θ_gt, ρ_gt)` — green if peak hit, red if
  miss.
- Detected peaks: cyan crosses at `(θ_peak, ρ_peak)`, sized by score.
- Title with peak_precision / peak_recall.

### 4.3 Per-edge support strip (`fig_support`, optional)

One subplot per GT edge (6×2 grid = 12 subplots for a finder cluster):
- NMS pixels (lime = inlier within `distance_thresh`, gray = outlier).
- Refined segment (red solid) + GT segment (blue dashed).
- Title: `{label} IoU={iou:.2f} hit={Y/N}`.

Only shown for finder clusters; skipped for NON_FINDER.

### 4.4 Summary bar chart (`fig_summary`, at end)

Across all images × clusters:
- Grouped bar chart: TP / FN / FP for peaks vs segments.
- A second panel: IoU distribution (histogram) for TP segments.
- A third panel: angle error vs rho error scatter (TP peaks).

## 5. CSV output format

One row per `(case, seed, cluster)`.  Columns:

```
case, seed, cluster_idx, finder, n_gt_edges,
peak_TP, peak_FN, peak_FP, peak_FP_near,
peak_precision, peak_recall, peak_F1,
mean_ang_err, mean_rho_err, mean_gt_bin_snrr,
seg_TP, seg_FN, seg_FP, seg_FP_near,
seg_precision, seg_recall, seg_F1,
mean_iou_1d, p05_iou_1d, mean_coverage_gt, mean_endpoint_err, mean_lateral_err,
n_fp_duplicate, n_fp_non_finder, runtime_ms
```

Plus an aggregate row per `(case, seed)` with the same columns (averaged
across clusters, `cluster_idx = -1`).

## 6. Implementation details

### 6.1 Reused code

- `FIXTURE_SPECS`, `_make_background`, `_config_v*` — import from
  `run_hough_ablation.py` (or copy the small case-definition dict; these are
  stable).
- `_run_pipeline_to_rois(image)` — reuse directly.
- `_compute_finder_edges(metadata, roi_offset, roi_shape)` — reuse directly
  (returns 36-edge GT with `segment` field).
- `extract_thin_edges`, `hough_vote_peaks`, `refine_line` — import from
  `qr_reader.detector.*`.
- `estimate_homography_dlt`, `project_points` — import from
  `qr_reader.detector.homography`.
- `_match_peak`, `_angular_distance_deg` — import or copy (small helpers).
- Cohen-Sutherland `_clip_segment` — import or copy (needed only if we add
  extra clipping; `_compute_finder_edges` already clips).

### 6.2 New code

- `_assign_cluster_to_finder(cluster, H, N) -> str` — greedy nearest-finder
  assignment.
- `_compute_gt_finder_centres(metadata) -> dict[str, np.ndarray]` — project
  grid `(3.5, 3.5)` etc. through H.  (Already exists as
  `_compute_finder_centres` in the audit mode; promote/reuse.)
- `_line_match(gt_edge, normals, rhos, angle_tol, rho_tol) -> int` — same as
  `_match_peak` but returns matched index + distances.
- `_segment_iou_1d(gt_seg, det_seg) -> dict` — returns
  `{iou, coverage_gt, coverage_seg, endpoint_err, lateral_err}`.
- `_classify_peaks(gt_edges, normals, rhos, scores, acc_data) -> dict` —
  peak-level metrics (§3.1).
- `_classify_segments(gt_edges, normals, rhos, segments, nms, angle,
  refine_kwargs) -> dict` — segment-level metrics (§3.2).
- `_characterize_fp(fp_segments, tp_gt_labels, cluster_finder) -> list[dict]`
  — §3.3.
- `_plot_roi_overlay(...)`, `_plot_hough_accumulator(...)`,
  `_plot_support_strips(...)`, `_plot_summary(...)`.
- `run_benchmark(args) -> pd.DataFrame` — top-level orchestrator.

### 6.3 Hough parameter presets

```python
HOUGH_PRESETS = {
    "default": {},  # library defaults
    "e6best": {
        "theta_step_deg": 0.5,
        "rho_step": 1.0,
        "nms_radius_rho": 2,
        "nms_radius_theta": 2,
        "gap_tolerance": 3,
        "distance_thresh": 1.5,
        "support_dilate": 0,
        "vote_scheme": "onebin",
        "theta_window_deg": 0,
        "acc_smooth": None,
    },
}
```

Split into `hough_kwargs` (passed to `hough_vote_peaks`) and `refine_kwargs`
(passed to `refine_line`) at call time — same split as the ablation harness.

### 6.4 Dependencies

- No new third-party deps.  Uses `numpy`, `matplotlib`, `cv2`, `scipy` (all
  already in the project).  `pandas` only if already available (check
  `pyproject.toml`); else fall back to `csv` module for output.

## 7. Console output format

After each image:

```
=== v12-default seed=42 ===
  Cluster 0 (TL):  12 GT edges
    Peaks:   TP=10  FN=2   FP=1   (prec=0.91, rec=0.83, F1=0.87)
    Segments: TP=8   FN=4   FP=1   (prec=0.89, rec=0.67, F1=0.76)
    IoU:     mean=0.72  p05=0.41   endpoint_err=3.2px  lateral=0.8px
  Cluster 1 (TR):  12 GT edges
    ...
  Cluster 3 (NON_FINDER):  0 GT edges
    Peaks:   TP=0  FN=0  FP=5  (all phantoms, 3 with support > 5px)
```

After all images:

```
=== SUMMARY (v12-default, 5 seeds) ===
  Peaks:     mean TP=9.8  FN=2.2  FP=1.4   F1=0.85
  Segments:  mean TP=7.6  FN=4.4  FP=1.2   F1=0.74
  IoU:       mean=0.70  p05=0.38
  By finder: TL F1=0.88  TR F1=0.72  BL F1=0.79
  By side:   top=0.82  bot=0.80  left=0.55  right=0.61
  By k:      k0=0.91  k1=0.74  k2=0.58  k5=0.60  k6=0.73  k7=0.90
```

## 8. What this script is NOT

- Not a sweep harness (no parameter grid).  One config per run.
- Not a regression gate (no pass/fail thresholds).  Pure measurement.
- Not a replacement for `run_hough_ablation.py` (which does sweeps + D/A/C/B
  classification).  This script is for **understanding** — it shows *how
  well* segments overlap, not just *whether* a peak exists.
- Does not save images to disk (notebook-style `plt.show()` only), unless
  `--csv` is given for the numeric summary.

## 9. Verification

- Run on `v12-clean` (no noise/rotation): expect peak recall ≈ 1.0, segment
  IoU ≈ 0.95+ for all 36 edges.  If not, the GT or matching logic is wrong.
- Run on `v12-default` with 5 seeds: expect the summary F1 to be in the same
  ballpark as the E6-best baseline from Plan 008.
- Visually inspect `fig_roi` for one `v12-default` image: GT segments should
  align with the finder pattern edges in the grayscale ROI.
