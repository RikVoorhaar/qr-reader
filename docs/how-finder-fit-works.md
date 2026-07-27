# How the finder-pattern fitter works

The core routine is `fit_finder_full()` in `finder_fit.py`.  It receives a grayscale
ROI cutout, an NMS edge map + edge-normal angles from `extract_thin_edges()`,
an approximate centre guess from the `CandidateCluster`, and an initial module-pitch
estimate.

It runs through **four phases**:

---

## Phase 1 — Orientation estimation (`estimate_orientation`)

### The problem
A QR finder pattern is a 7×7-module square.  The edge map contains pixels at the
boundaries between the dark rings and the light rings.  Because the pattern is
square, each edge-normal points in one of four directions: 0°, 90°, 180°, or 270°
(relative to the finder's local axes).  But the algorithm doesn't know which
edge pixel belongs to which of the four sides.

### The 4-fold symmetry trick
Instead of trying to separate the four edge families, the algorithm exploits
symmetry.  Each edge-normal direction `α` is taken modulo `π` (because a 180°
flip gives the same line orientation), giving a direction in `[0°, 180°)`.

Then it multiplies the angle by **4**: `4α mod 2π`.

| Actual edge direction | α mod π | 4α mod 2π |
|-----------------------|---------|-----------|
| 0°   (rightward)      | 0°      | 0°        |
| 90°  (upward)         | 90°      | 360° = 0° |
| 180° (leftward)       | 0°      | 0°        |
| 270° (downward)       | 90°     | 360° = 0° |

All four families collapse into the **same** direction in the 4-fold space.
A background edge at some random angle (say 37°) would land at 148° — not 0°.

### Weighted circular mean
Each edge pixel contributes a complex number `w_i · exp(4jα_i)` where `w_i` is
the NMS edge magnitude.  The sum `z = Σ w_i exp(4jα_i)` points in the dominant
direction.  The true orientation `φ` is then:

```
φ = angle(z) / 4   (mod π/2)
```

This gives the first local axis `e1 = (cos φ, sin φ)`.  The second axis is the
perpendicular: `e2 = (-sin φ, cos φ)`.

### Why this can fail
- If the ROI is too large and includes edges from non-finder structures
  (text inside the QR, background texture, other QR elements), those edges
  pollute the histogram.
- The weight is the NMS magnitude — strong background edges can overwhelm weak
  finder edges.
- The method gives `φ mod π/2`, which is always correct for a square, but
  there's a 4-way ambiguity: the axes could be swapped or flipped.  The 1-D
  profile fitting later resolves which is e1 and which is e2.

### Natural consequence: noise creates systematic *bias* towards integer
multiples of *π/2*, because **all** directions get multiplied by 4, so even random
edges get 'folded' into the tiny 90°-wide bin, when averaged into a histogram.
If there's a prevalence of a particular orientation in the edges (e.g. lines from
the background) this will bias the estimate.

---

## Phase 1b — Module-pitch refinement (`estimate_m_from_edges`)

An initial `m_est` came from `cluster_width / 7.0`, which is often an
**underestimate** (the cluster is defined by the 1:1:3:1:1 ratio scanner, which
typically captures the inner dark ring, not the full 7-module outer square).

`estimate_m_from_edges` corrects this:

1. Gating: only edge pixels whose normal is within 22.5° of e1 or e2 are kept.
2. Project each pixel onto e1 and e2 relative to the centre.
3. Compute the 5%–95% percentile span along each axis.
4. Divide by **7** (the finder's width in modules).

Take the **maximum** of the two axis estimates (the narrower span is potentially
cut-off by the ROI boundary).  Use `max(m_est, m_edge)` as the working pitch.

---

## Phase 2 — Affine 1-D profile fitting (`build_projection_profile` + `fit_finder_1d`)

### Building the profile
For each axis (e1, e2 separately):

1. Gate edge pixels to within 22.5° of the axis normal.
2. Project each pixel: `t = (p − centre) · axis`.  This gives a 1-D scalar.
3. Accumulate weighted edge strengths into bins of width `m/4` (quarter-module
   resolution), spanning `±4m`.

A perfect finder pattern viewed down an axis should show 6 edge peaks at
positions `{-3.5, -2.5, -1.5, +1.5, +2.5, +3.5} × m` relative to the true
centre (the inner dark/light transitions at ±0.5m and the centre gap ±0.5m are
invisible to a gradient-based edge detector — only the strong dark↔light
transitions show up).

### Grid-search fit
`fit_finder_1d` grid-searches over `(center_offset, m)` to maximise the sum of
interpolated profile values at those 6 expected positions, with a penalty for
the mean profile floor.  The search range:
- Centre offset: `±½m`
- Module pitch: `0.5×` to `1.8×` the initial estimate

### Why this can fail
- If the gated edge pixels include noise that creates phantom peaks at
  positions that happen to align with the expected pattern, the grid search
  locks onto the wrong spacing.
- The 6-peak grid at `{±1.5, ±2.5, ±3.5}` is a strong prior — a profile with
  only 2–3 real peaks might still get a decent score at the wrong offset.
- Large ROI → many background edges pass the angle gate → profile floor is
  high, diluting the real peaks.

---

## Phase 3 — Projective scanline fitting (`fit_scanline_projective`)

Phase 2 assumes **uniform spacing** (affine model — equal module pitch
everywhere).  But if the finder is perspectively distorted, the spacing is
non-uniform.  Phase 3 corrects this.

### Peak detection
`detect_profile_peaks()` finds local maxima in the profile using a prominence
threshold of 10% of the max profile value, with a minimum peak separation of
`½m`.

### Matching to canonical positions
The 6 canonical inner transition positions are `{-2.5, -1.5, -0.5, +0.5, +1.5, +2.5}`
in module units (note: **not** ±3.5 — the outer boundary at ±3.5 is white-on-white
and invisible to edge detection).

The algorithm predicts where these 6 positions would fall using the Phase 2
affine estimate, then matches the detected peaks to the nearest predicted
position using **nearest-neighbour with order preservation** (a peak at -3 must
match the -2.5 canonical before -1.5, etc.).  At least 4 matches are required.

### 1-D homography fit
With ≥4 correspondences `(u_canonical, t_observed)`, the algorithm fits a 1-D
projective map:

```
t = (a·u + b) / (c·u + d)
```

This is solved by SVD on the homogeneous linear system.  The outer positions
`±3.5` are then **extrapolated** through this map — they can't be observed
directly.

### Effective pitch
`m_effective = (t(+3.5) − t(−3.5)) / 7` — the average module pitch accounting
for the projective stretch.

### Fallback
If <4 peaks are detected, or <4 matches found, or the projective fit produces
non-finite values, Phase 3 falls back to the Phase 2 affine result.

### Why this can fail
- Peak detection misses real peaks (profile too noisy, peaks not prominent
  enough).
- Peak detection finds phantom peaks from background texture.
- The nearest-neighbour matcher can pair peaks with the wrong canonical
  positions, especially if the affine seed is poor.
- The 1-D homography can be degenerate (denominator → 0) for near-orthographic
  views.

---

## Phase 3b — Outer-line refinement (`refine_outer_line`)

After the centre and pitch are estimated, each of the 4 outer edges (at
±3.5m along e1 and e2) is refined independently:

1. Collect NMS edge pixels within 3.0 px perpendicular distance of the line
   `axis·p = axis·(center + position·axis)`.
2. Also gate by gradient-angle consistency (±22.5° of the axis normal).
3. **Default: fix the normal** to `axis` and refine only `rho` (the signed
   distance from origin) via a weighted mean.  This is robust when support
   pixels are sparse.
4. **Optional: full TLS fit** — weighted total least squares via SVD, refining
   both direction and position.  Used by Phase 4's template fitter.

### Corner extraction
The 4 refined lines are intersected pairwise to produce the 4 outer corners:

| Corner | Line 1 (e1) | Line 2 (e2) |
|--------|-----------|-----------|
| (-,-)  | u- (left) | v- (bottom) |
| (+,-)  | u+ (right)| v- (bottom) |
| (+,+)  | u+ (right)| v+ (top)   |
| (-,+)  | u- (left) | v+ (top)   |

The corners are in the local order `[(-,-), (+,-), (+,+), (-,+)]` relative to
the e1/e2 axes.  The e1,e2 axes form the basis: `corner = rho_u·e1 + rho_v·e2`
(because e1⊥e2, the axis basis is orthonormal).

---

## Phase 4 — Template fitting (optional, `use_template=True`)

A grid search over `(du, dv, m)` in a local neighbourhood around the Phase 3
estimate, with a **4-component score**:

| Component  | Weight | What it measures |
|-----------|--------|-----------------|
| Edge response | 0.25 | Sum of NMS magnitudes at the 12 expected edge positions (±{1.5, 2.5, 3.5} × m along each axis) |
| Polarity     | 0.35 | Gradient sign consistency with the 1:1:3:1:1 intensity profile (dark→light→dark→light→dark) |
| Cross-section NCC | 0.25 | Normalised cross-correlation between sampled intensity along each axis and the ideal 9-segment template |
| Quiet-zone   | 0.15 | Brightness of pixels at ±4m (should be white, i.e. high intensity) |

### Disabled by default
The main pipeline in `detector.py` calls `fit_finder_full()` with
`use_template=False`.  Phase 4 is only used when explicitly requested.

---

## Where things go wrong: known failure modes

### 1. ROI too large / too small

| Problem | Effect |
|---------|--------|
| ROI too large (`scale=1.5` in `cluster_to_bbox`) | Background edges pass the angle gate, polluting the orientation histogram and projection profiles |
| ROI too small | The outer edges (±3.5m) are cropped, so `refine_outer_line` has no support pixels → corners drift |

### 2. Orientation histogram bias

The 4-fold method averages over *all* edge pixels in the ROI.  If a strong
diagonal edge from background texture falls in the ROI, its contribution to
`z = Σ w_i exp(4jα_i)` can rotate the estimated `φ` by several degrees.  The
`angle_gate_deg=22.5°` parameter in later phases helps, but Phase 1 itself has
no spatial gating.

### 3. Peak detection failures

- In low-contrast images the profile has a high floor → peaks don't meet the
  10% prominence threshold.
- When multiple weak peaks exist near a strong one, the dedup (½m separation)
  might keep the wrong one.

### 4. Outer-line refinement on sparse support

If the ROI is cropped or the edge map is thin, `refine_outer_line` might have
fewer than 2 support pixels → falls back to the unrefined line position (which
may be off by 1–3 px).

### 5. Corner intersection with near-parallel lines

If e1 and e2 are not perfectly perpendicular (which can happen when the two
axes are estimated independently and `refine_outer_line` with `fix_normal=False`
modifies the normal), the intersection geometry degrades.  The default
`fix_normal=True` avoids this.

---

## Diagram: the 7×7 finder pattern cross-section

```
Axis cross-section (intensity → dark/light):

   -3.5   -2.5  -1.5  -0.5  0  +0.5  +1.5  +2.5  +3.5  (module units)
    |       |     |     |   |   |     |     |     |
    ██████░░░░░░███████████░░░░░░███████████░░░░░░██████
    black  white black    white  black    white black
    (outer (inner (center        (center (inner (outer
     ring)  ring)  square)        square) ring)  ring)
           ^visible^             ^visible^
           edges                 edges

The edge detector sees strong gradients at the █↔░ transitions
(±2.5 and ±1.5).  The outer boundary at ±3.5 is white-on-white
and invisible.  The centre gap at ±0.5 is also invisible.
```
