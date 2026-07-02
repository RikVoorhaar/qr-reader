# Plan 012 — Finder-Profile Edge Fitting (Idea 1 + Idea 4)

> **Goal:** Replace the Hough edge-detection stage with a QR-specific model
> fitting approach that avoids accumulator vote quantization entirely. Build
> a visualization script mirroring `full-pipeline-canny.py` for interactive
> diagnosis. Implement Idea 1 (orientation histogram + 1D profile fitting)
> first, then extend to Idea 4 (direct template fitting with polarity +
> 1:1:3:1:1 contrast scoring).

## Motivation

The Hough-based edge detector hit a hard 0.928 recall ceiling (Plan 011).
Root cause: edge-pixel position jitter scatters votes across adjacent ρ bins,
leaving the GT bin empty. No Hough parameter tuning can fix this — it's a
fundamental limitation of discrete accumulator voting.

ChatGPT's recommendation (see `docs/report-hough-bottleneck.md`): stop trying
to make Hough "perfect" and instead exploit the **known structure** of the
finder pattern. A finder is a 7×7 module square with 1:1:3:1:1 concentric
rings. The RLE cluster already gives us an approximate center and size. We
need only:

1. Estimate the finder's local orientation from the gradient distribution
2. Project edge pixels into finder-local coordinates
3. Fit the 6 expected edge-transition positions (±3.5m, ±2.5m, ±1.5m)
4. Extract the 4 outer corners from the fitted outer lines

This never asks individual edge pixels to vote into a global (θ, ρ) grid.

**Product metric:** The 4 outer corners of each finder pattern, accurate
enough to feed into the existing `extract_finder_patterns` → `find_triplets`
→ `build_named_landmarks` → homography pipeline. We do NOT need all 36 edges
— the outer quadrilateral is the target. Inner edges are supporting evidence.

## Out of scope

- LSD / ELSED line-segment detectors (Idea 2 — deferred)
- Soft-voting / clustered-peak Hough (Idea 3 — deferred)
- Wiring the new approach into `detector.py` (production pipeline) — this
  plan builds the module + visualization script only. Integration is a
  follow-up once recall/accuracy is validated.

## Architecture

### New module: `src/qr_reader/detector/finder_fit.py`

Pure NumPy/SciPy, no matplotlib. All geometry in **(x, y) = (col, row)**
pixel coordinates (consistent with `edges.py` and `hough.py`).

```
estimate_orientation(nms, angle, center_xy) → phi, e1, e2
build_projection_profile(nms, angle, center_xy, axis, m_est, angle_gate_deg) → (positions, profile)
fit_finder_1d(profile, m_est, expected_offsets) → (center_offset, m_fitted, peak_positions)
refine_outer_line(nms, angle, center_xy, axis, position, distance_thresh) → (normal, rho)
intersect_lines(n1, rho1, n2, rho2) → (x, y)
fit_finder_template(roi_gray, nms, angle, center_xy, e1, e2, m_est) → FinderFit
```

**`FinderFit` dataclass:**
```
center: ndarray (2,)      # fitted finder center (x, y)
e1: ndarray (2,)          # first local axis (unit vector)
e2: ndarray (2,)          # second local axis (unit, perp to e1)
m: float                  # fitted module pitch (px)
outer_lines: dict         # {"u+": (n, rho), "u-": (n, rho), "v+": (n, rho), "v-": (n, rho)}
corners: ndarray (4, 2)   # 4 outer corners (x, y): [(-,-), (+,-), (+,+), (-,+)]
score: float              # template fit score (Phase 4)
```

### New script: `src/qr_reader/scripts/full-pipeline-profile.py`

Notebook-style (`# %%` cells), `plt.show()`, no file saves. Mirrors
`full-pipeline-canny.py` structure: same image generation, binarization,
alignment scanning, clustering, ROI extraction, and GT computation
(`_compute_gt_edges` reused). Replaces the Hough stage with `finder_fit`.

Per cluster, produces figures showing each phase's intermediate results.

## Coordinate conventions

- `edges.py` returns `angle = atan2(gy, gx)` in **(x, y) = (col, row)** space.
- `finder_fit.py` works in **(x, y)** to be consistent with `angle`.
- Cluster center is `(row, col)` from `CandidateCluster`; convert to `(x, y)`
  via `center_xy = (center_col, center_row)`.
- GT geometry uses the module-grid homography (reuse `_compute_gt_edges`
  from `full-pipeline-canny.py`, which already works in ROI-local (x, y)).

## GT computation

Reuse `_compute_gt_edges` from `full-pipeline-canny.py` (already 36-edge
homography-based). Additionally compute:

- **GT finder center**: project grid center `(3.5, 3.5)` (for TL) through H.
- **GT module pitch**: `m_gt = |project(H, [1,0]) - project(H, [0,0])|` (distance
  per module along the grid x-axis; average with y-axis for robustness).
- **GT orientation**: angle of the normal to the GT top edge (k=0), mod π/2.
- **GT outer corners**: project grid corners `(0,0), (7,0), (7,7), (0,7)` per
  finder through H.

## Phases

### Phase 1: Orientation estimation + visualization

**Algorithm:**
```
ys, xs = nonzero(nms)
w = nms[ys, xs]
alpha = angle[ys, xs] mod π        # edge-normal in [0, π)
z = sum_i w_i * exp(4j * alpha_i)  # 4-fold symmetric complex sum
phi = (arg(z) / 4) mod (π/2)       # finder orientation mod π/2
e1 = (cos phi, sin phi)
e2 = (-sin phi, cos phi)
```

The factor 4 collapses the square's 4-fold symmetry: edges at α, α+π/2,
α+π, α+3π/2 all contribute coherently to `z`. This gives a robust
orientation estimate without needing to identify which edges belong to which
side.

**Visualization (Figure 1, per cluster):**
1. Grayscale cutout
2. NMS edges
3. Gradient-normal angle histogram **mod π/2** (fold α into [0, π/2) by
   `alpha_mod = alpha mod (π/2)`), weighted by strength, with GT φ and
   estimated φ marked as vertical lines
4. ROI with estimated e1/e2 axes (arrows from cluster center) and GT axes
   overlaid

**Print:** estimated φ, GT φ, axis error in degrees.

**Success criterion:** axis error < 2° on v12-clean, < 5° on v12-default.

### Phase 2: 1D projection profiles + transition fitting (Idea 1)

**Algorithm — profile construction:**
For each axis `e ∈ {e1, e2}`:
```
# Gate: only edge pixels whose gradient normal is close to e
e_angle = atan2(e[1], e[0]) mod π
alpha = angle[ys, xs] mod π
angle_diff = min(|alpha - e_angle|, π - |alpha - e_angle|)
gate = angle_diff < angle_gate_deg  (default 22.5°)

# Project onto axis (relative to center)
proj = (points - center_xy) @ e

# 1D histogram, quarter-module resolution
bin_width = m_est / 4
bins from -4*m_est to +4*m_est
profile = weighted_histogram(proj[gate], weights=w[gate])
```

**Algorithm — transition fitting:**
The profile should have 6 peaks at offsets `{-3.5, -2.5, -1.5, +1.5, +2.5, +3.5} * m`
relative to the finder center. Search over `(center_offset, m)`:

```
for du in linspace(-m_est/2, m_est/2, 17):
    for m in linspace(m_est*0.8, m_est*1.2, 21):
        expected = [-3.5, -2.5, -1.5, 1.5, 2.5, 3.5] * m + du
        hit_score = sum(interpolate(profile, pos) for pos in expected)
        miss_penalty = mean(profile values away from expected positions)
        score = hit_score - lambda * miss_penalty
    keep best (du, m)

fitted_peaks = expected offsets * m_best + du_best
outer_positions = ±3.5 * m_best + du_best
```

Repeat for both axes → `(du_best, m_u)` and `(dv_best, m_v)`. Average
`m_u` and `m_v` for the final module pitch.

**Visualization (Figure 2, per cluster):**
1. 1D profile P_u (along e1) with GT transition markers (±3.5m_gt, ±2.5m_gt,
   ±1.5m_gt) and fitted markers (±3.5m_fit, etc.)
2. 1D profile P_v (along e2) with same
3. ROI with fitted outer lines overlaid: lines at `u = ±3.5m_fit` and
   `v = ±3.5m_fit` (drawn as infinite lines through the ROI), GT outer lines
   for comparison

**Print:** fitted m, GT m, m error (%), fitted center offset, GT center,
outer line position error (px).

**Success criterion:** outer line position error < 1.5 px on v12-clean,
< 3 px on v12-default.

### Phase 3: TLS line refinement + corner extraction

The 1D fit gives approximate outer line positions in finder-local coords.
Refine each of the 4 outer lines to sub-pixel accuracy using the actual NMS
edge pixels:

**Algorithm — per outer line:**
```
# The outer line is at position t along axis e (t = ±3.5*m_fit)
# Its normal is e, and it passes through center_xy + t * e
approx_normal = e
approx_rho = e @ (center_xy + t * e)

# Collect nearby edge pixels (within distance_thresh, angle-gated)
dists = |points @ approx_normal - approx_rho|
mask = (dists < distance_thresh) & angle_gate

# Weighted TLS fit (same as refine_line in hough.py)
w = strengths[mask] / sum(strengths[mask])
c = weighted_centroid(points[mask])
SVD of weighted-centered points → direction, normal
refined_rho = normal @ c
```

**Corner extraction:**
Intersect the 4 refined outer lines:
```
corner_00 = intersect(u- line, v- line)
corner_10 = intersect(u+ line, v- line)
corner_11 = intersect(u+ line, v+ line)
corner_01 = intersect(u- line, v+ line)
```

Line intersection: solve the 2×2 system `n1·p = ρ1`, `n2·p = ρ2`.

**Visualization (Figure 3, per cluster):**
1. ROI with refined outer quadrilateral (4 corners connected, solid green)
   + GT quadrilateral (dashed red) + 1D-fit quadrilateral (dotted blue, for
   comparison)
2. Refined lines with supporting NMS pixels highlighted

**Print:** per-corner error (Euclidean px), mean corner error, max corner
error.

**Success criterion:** mean corner error < 1.0 px on v12-clean, < 2.5 px on
v12-default.

### Phase 4: Template fitting with polarity + contrast (Idea 4)

Extend the Phase 2 scoring to exploit the **intensity structure** of the
finder pattern, not just edge magnitude.

**Polarity scoring:**
The 1:1:3:1:1 cross-section (center → outward) is:
```
black(3) → white(1) → black(1) → white(1) → [quiet: white]
```

Expected gradient directions along axis e (outward = +e direction):
```
+1.5m: black→white  → gradient = +e  (sign = +1)
+2.5m: white→black  → gradient = -e  (sign = -1)
+3.5m: black→white  → gradient = +e  (sign = +1)
-1.5m: black→white  → gradient = -e  (sign = -1)
-2.5m: white→black  → gradient = +e  (sign = +1)
-3.5m: black→white  → gradient = -e  (sign = -1)
```

General formula at position `t * k * m` (t=±1, k∈{1,2,3}):
`expected_sign = t * (-1)^(k+1)`

**Polarity score:**
```
for each expected position p_k:
    grad_sign = sign(dot(gradient_at_p, e))
    polarity_match = (grad_sign == expected_sign)
    score += polarity_match * edge_magnitude_at_p
```

**Cross-section contrast scoring:**
Sample the image intensity along a 1D cross-section through the finder
center (along e1 and e2). The ideal template (normalized to [0,1]):
```
[white, black, white, black×3, white, black, white]
widths: 1:1:3:1:1 (relative to m)
```

Score = normalized cross-correlation between sampled profile and ideal
template (shifted/scaled by fitted center + m).

**Quiet-zone scoring:**
Mean intensity outside ±3.5m should be high (bright). Score =
`mean(intensity in quiet zone) / 255`.

**Combined template score:**
```
score = w_edge * edge_response
      + w_polarity * polarity_consistency
      + w_contrast * contrast_ncc
      + w_quiet * quiet_zone_brightness
```

Search over `(du, dv, m)` in a local neighborhood around the Phase 2
estimate. The Phase 2 result seeds the search, so the grid is fine and small.

**Visualization (Figure 4, per cluster):**
1. Cross-section intensity profile along e1 (through fitted center), with
   the ideal 1:1:3:1:1 template overlaid (scaled to fitted m)
2. Same along e2
3. Polarity arrows at each expected edge position (green = correct, red =
   wrong)
4. ROI with Phase 4 fitted quadrilateral + Phase 3 quadrilateral + GT
   quadrilateral

**Print:** template score breakdown (edge, polarity, contrast, quiet),
corner error comparison Phase 3 vs Phase 4.

**Success criterion:** Phase 4 corner error ≤ Phase 3 corner error on
v12-default (template fitting should be at least as good, ideally better on
noisy images).

## Implementation order

1. **Phase 1** — `estimate_orientation` in `finder_fit.py` + Figure 1 in
   script. Validate axis error before proceeding.
2. **Phase 2** — `build_projection_profile` + `fit_finder_1d` + Figure 2.
   Validate outer line positions.
3. **Phase 3** — `refine_outer_line` + `intersect_lines` + Figure 3.
   Validate corner errors. This is the minimum viable result.
4. **Phase 4** — `fit_finder_template` + Figure 4. Compare to Phase 3.

Each phase is independently testable: the script shows the intermediate
result and prints the error metric before moving on.

## Reused components

| Component | Source | Usage |
|-----------|--------|-------|
| `extract_thin_edges` | `edges.py` | NMS edges + gradient angles (unchanged) |
| `cluster_to_bbox`, `cutout` | `roi.py` | ROI extraction (unchanged) |
| `find_alignment_patterns_2d` | `alignment.py` | Candidate scanning (unchanged) |
| `cluster_candidates` | `clustering.py` | Clustering (unchanged) |
| `estimate_homography_dlt`, `project_points` | `homography.py` | GT computation |
| `_compute_gt_edges` | `full-pipeline-canny.py` | GT edges (copy into new script) |
| Image generation (AugmentationConfig, generate_sample) | `synth/` | Same test image setup |

## Risks and mitigations

1. **Orientation ambiguity**: The 4-fold symmetric sum gives φ mod π/2, which
   is correct for a square — we can't distinguish "top" from "left" but don't
   need to (all 4 outer lines are found symmetrically). **Mitigation:** none
   needed; the symmetry is a feature.

2. **Cluster center offset**: The RLE cluster center may be off by 1–2
   modules from the true finder center. **Mitigation:** the Phase 2 search
   over `center_offset` covers ±m/2, which is ±half a module — sufficient.

3. **Module pitch estimate**: The cluster width may not exactly equal 7
   modules. **Mitigation:** search over m in [0.8m_est, 1.2m_est].

4. **Skewed finders**: Perspective warp can make e1/e2 non-orthogonal.
   **Mitigation:** Phase 1 assumes orthogonality (e2 = perp(e1)). For strong
   perspective, Phase 4 could add an optional skew parameter. Defer until
   corner errors are evaluated.

5. **Edge gating threshold**: The `angle_gate_deg` for profile construction
   filters edge pixels by gradient-normal alignment with the axis. Too narrow
   → few pixels; too wide → noise. **Mitigation:** start at 22.5°, tune
   visually in the script.
