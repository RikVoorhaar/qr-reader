# Plan 014 — Perspective-Aware Finder-Fit

**A research plan for handling perspective distortion in the finder-pattern
fitting pipeline.**

---

## 1. Current Approach

`fit_finder_full` (`finder_fit.py:711`) models each finder pattern as an
orthogonal, unit-aspect-ratio **square** in a local coordinate system
`(e1, e2)`. It runs four phases:

| Phase | Function | What it does |
|-------|----------|--------------|
| 1 | `estimate_orientation` | 4-fold gradient histogram → `φ` (mod π/2), axes `e1 ⟂ e2` |
| 2 | `build_projection_profile` → `fit_finder_1d` | Project NMS edge pixels onto `e1` and `e2` independently; search for 6 edge-transition peaks at `{±1.5, ±2.5, ±3.5} × m` (equal spacing) |
| 3 | `refine_outer_line` → `_corners_from_rho` | For each outer edge (±3.5m), collect nearby NMS pixels with `fix_normal=True`; compute `ρ = axis·pos` as a weighted mean; intersect the 4 lines to get corners |
| 4 | `fit_finder_template` | NCC + polarity + quiet-zone scoring (not used in production) |

The core assumption across all phases: the finder is a **square in the
image plane** — `e1 ⟂ e2`, and `m` is the same along both axes.

The downstream pipeline in `detector.py` fits a **Procrustes similarity**
(rotation + uniform scale + translation) from the three finder centres,
then refines via Levenberg-Marquardt on 12 corners. The similarity
initialiser provides no perspective handles `(h20, h21)`, so LM must
discover perspective entirely from point correspondences — many of which
are already corrupted by the intra-finder bias.

---

## 2. Failure Mode Under Perspective

Under perspective projection the finder appears as a **general
quadrilateral**, not a square or even a parallelogram. Every assumption in
the current pipeline is violated.

### 2.1 Orientation: bisector, not directions

The 4-fold histogram (`Σ w·exp(4iα)`) collapses two true image directions
into one bisector angle `φ` and forces orthogonality:

```
world:       ∟ = 90°   (square on a plane)
image:       ∠ = 90° + δ   (two line families, not orthogonal)
histogram:   φ ≈ bisector of the two families
axes:        e1, e2 forced orthogonal → neither points along the real edges
```

This is the **root failure**: every subsequent step projects onto axes that
are misaligned by `±δ/2`. Literature on planar Manhattan rectification
explicitly treats two line families as distinct vanishing directions
rather than collapsing them into one orthogonal pair [Shemiakina et al.,
2019; Abbas and Zisserman, 2019].

### 2.2 1D profiles: equal spacing is lost

Projecting edge pixels from a perspective-warped finder onto an axis
yields transition positions that are **not equally spaced**.  A planar
homography restricted to a line is a 1D projective map `t = (au + b)/(cu + d)`;
equal spacing in the canonical frame becomes a rational function in the
image.  The single invariant is the **cross-ratio** of colinear quadruples
— not the spacing `m` [Hartley & Zisserman, 2003].

Current code assumes `peak_k = k·m + offset` for `k ∈ {±1.5, ±2.5, ±3.5}`.
Under perspective this model drifts: the 6 peaks fit poorly, and the
estimated `m` becomes a biased average that corrupts the downstream
`_corners_from_rho`.

### 2.3 Outer line refinement: fixed normals

`refine_outer_line` with `fix_normal=True` sets the normal to the
bisector-derived `e1`/`e2`.  The true outer-edge normal differs by `±δ/2`.
With `fix_normal=False` it runs an unconstrained TLS fit, but the four
edges are fit **independently** with no coupling constraint, so they drift
apart.

Furthermore, the original Strategy B constraint — opposite sides parallel —
is wrong for perspective.  Parallelism is an **affine** property.
Under projective geometry, opposite sides of a planar rectangle converge
to a vanishing point; the correct constraint is that edges within each
family are consistent with **one vanishing direction**, not zero relative
angle.

### 2.4 Corner computation: rectangle formula

`_corners_from_rho` synthesises corners as:

```python
c00 = um·e1 + vm·e2
c10 = up·e1 + vm·e2
c11 = up·e1 + vp·e2
c01 = um·e1 + vp·e2
```

This uses `vm` (the `ρ` of the left outer edge) identically for both
`c00` and `c10` — i.e. it assumes the left edge is a straight line
parallel to `e2`.  Under perspective the left edge slants, and its `ρ`
varies along its length.  The same error affects all four edges.

The structural error is that **corners are synthesised from intermediate
variables** (`um, up, vm, vp`, `e1, e2`) rather than from a geometrically
consistent warp.  Strategy E already noted this as a key question: derive
corners from a homography rather than from `_corners_from_rho`.

### 2.5 Global homography: similarity init

The Procrustes similarity from 3 finder centres carries zero perspective
degrees of freedom.  LM must simultaneously correct perspective and absorb
intra-finder corner errors.  With 12 biased corner correspondences, it
frequently converges to a local minimum.

The marker-detection literature consistently places **global geometric
constraints** (line families, vanishing points, multi-finder consistency)
before local refinement, rather than hoping LS will fix everything
post-hoc [Muñoz-Salinas et al., 2016; Yu et al., 2019].

---

## 3. Anatomy of the Problem Space

| Dimension | Description |
|-----------|-------------|
| **Intra-finder** | Each finder's local warp: orientation (two family directions), anisotropic scale (`m_u ≠ m_v`), projective shear (non-parallel edges), corner positions |
| **Inter-finder** | Three finders at different apparent scales (depth ratio), feeding into version estimation and the global homography |

The ordering matters: **projective correction must happen before corner
synthesis**.  Once corners are produced by the current `_corners_from_rho`
formula, the downstream homography fights already-biased inputs.  The fix
is to produce corners from a per-finder projective warp, then fit a global
homography from clean correspondences.

---

## 4. Strategy Space

### 4.1 Strategy A — Per-Axis Module Pitch

**Adopt `m_u` and `m_v` independently.**  Store both from the 1D profile
fits and use them in the finder state.  The finder becomes an axis-aligned
rectangle in (u,v) space instead of a square.

```
m_u ≠ m_v   →   first-order anisotropy captured
```

**Pros:** One-line change.  Immediate diagnostic: the `m_u/m_v` ratio
quantifies the perspective depth gradient across the finder.

**Cons:** Still forces edges parallel to `e1`/`e2` (no skew).  The 1D
profile blur from axis misalignment remains unchanged.

**Verdict:** Adopt immediately as necessary bookkeeping, but it fixes only
the first-order symptom.  Not sufficient for perspective robustness.

### 4.2 Strategy B — Two Line Families

**Replace the 4-fold histogram with a two-cluster direction estimate.**

The finder pattern has two dominant edge families separated by ≈90° in
world space but not in image space.  Cluster edge normals modulo π into
two modes, output two unit normals `n₁, n₂` (not forced orthogonal), plus
a confidence score per family.

These two directions are approximations of the vanishing lines of the
planar rectangle — they are the image-plane manifestation of the two
parallel edge families that converge at finite vanishing points under
perspective.

**Constraint structure:** Edges within one family share the same direction
`n_family` but have different offsets `ρ`.  The correct projective
constraint is **vanishing-direction consistency**, not parallelism.

**Pros:** Eliminates the bisector bias.  `n₁`, `n₂` become the correct
reference directions for all downstream steps.

**Cons:** 2-mode clustering is more fragile with sparse NMS pixels than
the 4-fold summation.  Might need a fallback to the current method for
near-orthogonal (weak perspective) cases.

**Key question:** Measure the angle `|∠(n₁, n₂) - 90°|` as a function of
perspective angle and compare it against the current `φ` bisector error.
This tells us how much bias we are removing.

### 4.3 Strategy C — Projective-Aware Profile Fitting

**Keep the scanline idea but replace equal-spacing with 1D projective
fit.**

On a scanline across the finder along a family direction:

1. Extract a 1D intensity/edge profile
2. Detect the 6 transition peaks (3 dark→light, 3 light→dark)
3. Their **order** is known (outer→inner→center)
4. Their canonical positions in module units are known:
   `{-3.5, -2.5, -1.5, +1.5, +2.5, +3.5}`
5. Fit a 1D projective map `t = (au + b)/(cu + d)` that maps canonical
   positions to detected peaks
6. Score with polarity consistency and quiet-zone checks

The **cross-ratio** of any 4 colinear points is preserved under planar
homography, so the canonical transitions `(-3.5, -1.5, +1.5, +3.5)` must
have the same cross-ratio in the image.  This is a stronger constraint
than equal spacing.

**Implementation:** RANSAC over ordered 4-point subsets of the 6 peaks to
fit a 1D homography, then score the remaining peaks for inlier agreement.

**Pros:** Removes the equal-spacing assumption.  Uses the nested structure
of the finder (1:1:3:1:1) as a known prior.  Cross-ratio is a projective
invariant and checks out automatically on a true finder.

**Cons:** Adds a RANSAC stage per finder.  Needs reliable peak detection
on each scanline (which might be noisy under strong blur/noise).

**Key insight:** A 1D projective fit on a central scanline provides a
compact initialiser for a 2D per-finder homography (Strategy D).  It
directly answers: "are the transitions on this scanline consistent with a
1:1:3:1:1 finder pattern, and what is the local projective warp?"

### 4.4 Strategy D — Per-Finder Homography Refinement

**Fit an 8-DOF homography per finder, refined on edge residuals.**

The finder pattern is a planar square `[-3.5, 3.5]²` in module units.  Its
image is a general quadrilateral — an 8-DOF projective warp of the
canonical square.  Fit it explicitly.

**Two-stage approach:**

1. **Initialise** from Strategy C output (1D projective fit on a few
   scanlines → 2×2 affine approximation → 3×3 homography with `h20=h21=0`)
   plus the two family directions from Strategy B and per-axis `m` from A.

2. **Refine** the 8-DOF homography via Gauss-Newton or LM against an
   edge-domain objective: project the known edge lines of the canonical
   finder pattern through the trial `H`, measure perpendicular distance to
   nearby NMS pixels with consistent gradient orientation, and minimise
   squared distance with a robust loss (soft-L₁ or Huber).

**Parameterisation:** 8-parameter homography with `H[2,2] = 1`.  A good
starting point is the affine initialiser from the current pipeline
(Strategy A + B), which provides a decent translation + linear part.

**Pros:** Directly optimises the geometric object (projected finder
edges), not intermediate proxy variables.  Corners are a natural byproduct:
project the canonical corners through the refined `H`.  The optimisation
is small (1 ROI, ~8 params, few hundred support pixels per edge).

**Cons:** Non-convex — needs a good initialiser (hence A+B+C as
prerequisites).  Inner-ring NMS pixels (from the 3×3 dark module) can
pollute the edge sample; needs an explicit edge-label gate.

**Key implementation note:** Existing direct-alignment literature [Baker
& Matthews, 2002] provides an efficient framework for this — inverse
compositional Gauss-Newton can run very fast on a single finder ROI when
provided a reasonable initial warp.  The expensive part is the search, not
the model.

### 4.5 Strategy E — Global Homography From Refined Finder Geometry

**Fit one global homography from all finder corner correspondences, then
run LM on reprojection error.**

Once Strategies A–D produce reliable per-finder homographies (and thus
reliable corner positions), the global step is straightforward:

1. From each of the 3 finders, extract 4 canonical corner positions in
   grid coordinates and their refined image positions → 12 point
   correspondences
2. Fit `H_global` via standard DLT on all 12 correspondences
3. Optionally refine with LM on reprojection error across all 12 points

This replaces the current "similarity from centres + LM" pipeline with a
much stronger initialiser that already carries perspective DOF.

**Pros:** Geometrically correct.  Uses the same DLT + LM machinery already
in `homography.py`.  Cross-check: project centre points through `H_global`
and compare against the directly-measured centres — this validates the
per-finder fits.

**Cons:** Depends entirely on the quality of per-finder corner estimates.
If Strategies A–D are not enough, this inherits their errors.
Additionally, the 12 correspondences are clustered in three groups of 4
each, which can still produce a degenerate DLT for near-frontoparallel
views (see earlier diagnostic of condition number > 10⁴).  A wider
distribution of points (e.g. including grid-edge points between finders)
would help.

---

## 5. Recommended Research Order

Strategies build on each other — each provides the initialiser for the next.

```
A (diagnostic: m_u, m_v separation)
  │
  ▼
B (two family directions n₁, n₂)
  │
  ▼
C (1D projective scanline fit + cross-ratio check)
  │
  ▼
D (per-finder homography refinement)
  │
  ▼
E (global homography from refined corners)
```

### Phase 1 — Diagnostic benchmarks

1. **Synthetic single-finder sweep.**  Generate isolated finder-pattern
   ROIs with known homography, varying pitch/yaw independently from 0° to
   45°, no noise/blur.  Measure:
   - True `φ₁, φ₂` directions vs. the 4-fold histogram bisector `φ`
   - True `m_u, m_v` vs. the fitted single `m`
   - Corner RMSE of `_corners_from_rho` vs. ground truth
   - Plot these against perspective angle to find the inflection point

2. **`m_u/m_v` ratio diagnostic.**  For a range of perspectives, measure
   how much the per-axis ratio deviates from 1.0.  This quantifies the
   first-order anisotropy.

3. **Normal misalignment diagnostic.**  Run `refine_outer_line` with
   `fix_normal=False` and measure `∠(refined_normal, e1)` for each edge.
   Plot against perspective angle.

### Phase 2 — Strategy A (per-axis `m`)

4. Store `m_u`, `m_v` in `FinderFit`.  Use them in downstream corner
   computation and version estimation.  Benchmark corner error vs. current
   shared `m`.  **Expected: modest improvement, primarily diagnostic.**

### Phase 3 — Strategy B (two line families)

5. Replace `estimate_orientation` with a 2-cluster estimator:
   - Compute edge normals modulo π for all NMS pixels
   - Cluster into two modes using weighted angular k-means
   - Selection heuristic: pick the two orthogonal-adjacent modes (≈90°
     apart in world space, whatever apart in image space)
   - Falling back to the current 4-fold histogram when the modes are
     ambiguous

6. **Benchmark:** Single-finder sweep comparing true family angles to
   estimated `n₁, n₂` vs. the old bisector `φ`.  Measure downstream
   projection profile quality.

### Phase 4 — Strategy C (projective scanlines)

7. Implement 1D projective fitting on two central scanlines (one per
   family direction):
   - Detect 6 transition peaks via derivative zero-crossings on the
     scanline intensity/edge profile
   - RANSAC: sample 4 ordered peaks, fit 1D homography, score remaining 2
     peaks + polarity + quiet-zone brightness
   - Best fit → 1D projective parameters for this scanline

8. **Benchmark:** Transition matching under yaw/pitch sweeps.  Compare
   three models: equal spacing, affine spacing, 1D projective fit.  Report
   peak-assignment accuracy and corner seed error for each.

### Phase 5 — Strategy D (per-finder homography)

9. Build affine initialiser from A+B output, parameterise as 8-DOF
   homography, refine on NMS edge residuals with soft-L₁ loss.  Derive
   corners from the refined homography.

10. **Benchmarks:**
    - Single-finder homography RMSE vs. perspective angle
    - Convergence basin size: start from perturbed initialiser (±5px
      translation, ±5° family angle, ±20% scale), report convergence rate
    - Corner RMSE comparison: `_corners_from_rho` vs. homography corners

### Phase 6 — Strategy E (global homography)

11. Feed refined finder corners into DLT + LM from all 12 points.
    Replace the similarity-from-centres initialiser.

12. **Benchmarks:**
    - Global homography reprojection error vs. current pipeline
    - Decode rate on default augmentation config
    - Corner error at the QR-code level (project `(0,0)`, `(N,0)`, etc.
      through H)

---

## 6. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Single-finder corner RMSE (0° perspective) | ~1 px | < 1 px (no regression) |
| Single-finder corner RMSE (30° perspective) | ~15-30 px | < 5 px |
| v12-clean detection rate | 5/5 seeds | 5/5 |
| v12-default detection rate | 2/5 seeds | ≥ 4/5 |
| v12-default corner error (when detected) | 70–260 px | < 15 px |
| DLT condition number (12 point correspondences) | > 10⁴ | < 500 |
| Existing test regression | — | 0 regressions |

---

## 7. Scope Boundaries

**In scope:**
- Perspective distortion of the finder pattern due to non-frontoparallel
  camera pose (pitch/yaw up to ~45°)
- Anisotropic module pitch and projective shear within a single finder
- Improved initialisation of the QR-level homography
- Deterministic synthetic benchmarks with known ground truth

**Out of scope:**
- Lens distortion (radial/tangential) — assumed corrected upstream
- Multi-QR detection — single QR code assumption
- Extreme perspective (>60°) — vanishing QR codes are a separate problem
- Real-world image pipeline (lighting, specular highlights, sensor noise
  beyond the augmentation config)
- Decoder robustness (this plan is about geometry, not Reed-Solomon)
