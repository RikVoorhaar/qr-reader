# Plan 015 — Perspective-Aware Finder-Fit Implementation

**A concrete, step-by-step implementation plan with control gates.**
Each step is designed to be autonomously evaluable and reversible.

---

## 0. Branch & Baseline

### 0.1 Branch

```
git checkout -b plan-015-perspective-finder
```

### 0.2 Record baseline

Run the existing test suite and the standard benchmark. Save the outputs
into `docs/plan-015-baseline/`.  Do not start implementation until these
numbers are recorded.

```bash
pytest src/qr_reader/tests/test_detector.py -q > docs/plan-015-baseline/test_detector_baseline.log
python src/qr_reader/scripts/qr_benchmark.py > docs/plan-015-baseline/qr_benchmark_baseline.log
```

### 0.3 Baseline deliverables

- `docs/plan-015-baseline/test_detector_baseline.log`
- `docs/plan-015-baseline/qr_benchmark_baseline.log`
- A one-line summary in this file:
  - Tests: `PASS / FAIL / SKIP`
  - v12-default detection rate from the benchmark

---

## Step 1 — Add a Single-Finder Perspective Benchmark

### Goal

Create an isolated, ground-truth benchmark for one finder pattern under
controlled perspective (yaw and pitch).  This benchmark is the evaluation
engine for every later step.  It does **not** change production code.

### Implementation

1. Create `src/qr_reader/tests/detector/test_finder_perspective.py`.
2. Add a helper `synthesize_finder_homography(yaw, pitch, image_size)` that
   warps a canonical `[-3.5, 3.5]²` finder pattern into an image ROI.
3. Add a helper `evaluate_fit(fit, H_true)` that computes:
   - corner RMSE between `fit.corners` and the 4 true projected corners
   - family-angle error (after Step 3)
   - `m_u / m_v` ratio (after Step 2)
4. Parametrise a test matrix:
   - yaw: `0°, 10°, 20°, 30°, 40°`
   - pitch: `0°, 10°, 20°, 30°, 40°`
   - no noise, no blur, no occlusion

### Deliverable

- File: `src/qr_reader/tests/detector/test_finder_perspective.py`
- Running `pytest src/qr_reader/tests/detector/test_finder_perspective.py -v`
  prints a table of corner RMSE vs yaw/pitch for the **current**
  `fit_finder_full`.

### Evaluation

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -v
```

### Control gate

- **PASS:** The benchmark runs to completion and produces a CSV/table.
- **FAIL:** Benchmark crashes, hangs, or produces non-finite values.
- **Action on fail:** Fix the benchmark harness before touching production code.

---

## Step 2 — Per-Axis Module Pitch (`m_u`, `m_v`)

### Goal

Allow `fit_finder_1d` to estimate a separate module pitch for each axis.
Keep the old behaviour as the default; expose the new bookkeeping behind
an opt-in flag.

### Implementation

1. Extend the `FinderFit` dataclass in `finder_fit.py` with two new
   optional fields:
   ```python
   m_u: Optional[float] = None
   m_v: Optional[float] = None
   ```
2. Modify `fit_finder_1d` to accept an `axis: Literal["u", "v"]` argument
   and return `m_axis`.
3. Modify `build_projection_profile` / its caller to run the fit twice
   (once per axis) when `estimate_anisotropic_pitch=True`.
4. Store both values in `FinderFit`; leave `_corners_from_rho` using the
   original shared `m` unless explicitly told otherwise.

### Deliverable

- `FinderFit` instances produced with `estimate_anisotropic_pitch=True`
  contain non-None `m_u` and `m_v`.
- A new diagnostic test asserts `m_u / m_v` grows monotonically with
  perspective angle on the Step-1 benchmark.

### Evaluation

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -v -k anisotropic_pitch
pytest src/qr_reader/tests/test_detector.py -q
```

### Metrics

- Existing `test_detector.py` must not regress.
- On the Step-1 benchmark, the `m_u / m_v` ratio must differ from 1.0 by
  > 5% at 30° perspective and must trend monotonically with angle.

### Control gate

- **PASS:** All existing tests pass; diagnostic ratio behaves as expected.
- **FAIL:** Any regression in `test_detector.py`, or `m_u/m_v` stays near
  1.0 under perspective.
- **Action on fail:** Revert the anisotropic flag to default-off, fix the
  bug, and re-run.

---

## Step 3 — Two Line-Family Orientation Estimation

### Goal

Replace the single `φ` bisector with two independently estimated edge
family directions `n_u`, `n_v`.  Provide a robust fallback to the old
4-fold histogram.

### Implementation

1. Implement `estimate_orientation_two_families(nms_pixels, weights)` in
   `finder_fit.py`:
   - Compute edge normals modulo π for each NMS pixel.
   - Run weighted 2-mode angular clustering (e.g. expectation-maximisation
     for two von-Mises distributions).
   - Return `(n_u, n_v, score_u, score_v)`.
2. Add a heuristic fallback: if the two modes are ambiguous (score ratio
   < 0.3 or angle separation outside `[30°, 150°]`), fall back to the
   existing `estimate_orientation`.
3. Gate the new estimator behind `use_two_families=True`.

### Deliverable

- Function `estimate_orientation_two_families` in `finder_fit.py`.
- `FinderFit` gains optional fields `n_u`, `n_v`.
- Diagnostic test in `test_finder_perspective.py` reports the angle error
  between estimated `(n_u, n_v)` and the true image edge families.

### Evaluation

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -v -k two_families
pytest src/qr_reader/tests/test_detector.py -q
```

### Metrics

- Frontoparallel cases (`yaw < 5°, pitch < 5°`) must still pass all
  existing tests.
- At 30° perspective, the mean angle error of `n_u` and `n_v` must be
  < 5° (vs. ground truth).
- The old bisector `φ` error at 30° must be shown to be > 8° in the same
  diagnostic (to confirm we are removing bias).

### Control gate

- **PASS:** Existing tests pass; family-angle error meets target.
- **FAIL:** Clustering is unstable on frontoparallel inputs or degrades
  `test_detector.py`.
- **Action on fail:** Make the fallback the default and keep the new
  estimator opt-in until clustering is stabilised.

---

## Step 4 — 1D Projective Scanline Fitting

### Goal

Implement a projective-aware scanline fit that maps canonical transition
positions to observed peaks via a 1D homography, using cross-ratio as the
projective invariant.

### Implementation

1. Implement `fit_scanline_projective(scanline, family_direction, n_u, n_v)`:
   - Extract a 1D edge-strength profile along a line through the finder
     centre in the given family direction.
   - Detect up to 6 ordered transition peaks (3 dark→light, 3
     light→dark).
   - Canonical positions: `[-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]`.
   - RANSAC over ordered 4-point subsets to fit a 1D projective map
     `t = (au + b)/(cu + d)`.
   - Score inliers by polarity and quiet-zone brightness consistency.
2. Add `use_projective_scanlines=True` flag to `fit_finder_full`.
3. When enabled, use the projective fit to initialise `m_u`, `m_v` and
   the family-direction offsets.

### Deliverable

- Function `fit_scanline_projective` in `finder_fit.py`.
- Diagnostic test comparing three models on the Step-1 benchmark:
  - equal spacing (current)
  - affine spacing
  - 1D projective fit
- A report of peak-assignment accuracy and corner-seed RMSE per model.

### Evaluation

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -v -k projective_scanline
pytest src/qr_reader/tests/test_detector.py -q
```

### Metrics

- Peak-assignment accuracy must be ≥ 95% on the noise-free sweep.
- Corner-seed RMSE at 30° perspective must be lower than the equal-spacing
  seed by at least 30%.
- No regression in `test_detector.py`.

### Control gate

- **PASS:** Projective seed beats equal-spacing seed on perspective; no
  regressions.
- **FAIL:** RANSAC is unstable, peak detection misses transitions, or
  `test_detector.py` degrades.
- **Action on fail:** Keep `use_projective_scanlines=False` by default;
  debug the scanline extractor on failing cases.

---

## Step 5 — Per-Finder Homography Refinement

### Goal

Fit an 8-DOF homography per finder, initialised from Steps 2–4, and
derive corners from the warp instead of `_corners_from_rho`.

### Implementation

1. Implement `refine_finder_homography(image_roi, H_init, canonical_square)`:
   - Canonical square corners: `[-3.5, 3.5]²`.
   - Objective: sum of robust squared perpendicular distances from NMS
     edge pixels to the projected canonical edges, with gradient-orientation
     consistency.
   - Optimiser: Gauss-Newton or Levenberg-Marquardt with soft-L₁ loss.
   - Parameterise `H` with `H[2,2] = 1`.
2. Implement `corners_from_homography(H)`:
   - Project the 4 canonical corners through `H`.
3. Add `use_finder_homography=True` flag to `fit_finder_full`.
   - When enabled, run the affine initialiser (Step 2/3/4) →
     `refine_finder_homography` → `corners_from_homography`.
   - When disabled, keep the existing `_corners_from_rho` path.

### Deliverable

- Functions `refine_finder_homography` and `corners_from_homography` in
  `finder_fit.py`.
- Diagnostic test reporting single-finder homography RMSE vs. perspective
  angle.
- A convergence-basin test: perturb `H_init` by ±5 px translation, ±5°
  family angle, ±20% scale; report convergence rate.

### Evaluation

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -v -k finder_homography
pytest src/qr_reader/tests/test_detector.py -q
```

### Metrics

- Single-finder corner RMSE at 30° perspective must be < 5 px.
- `_corners_from_rho` RMSE at 30° must be shown to be > 10 px in the same
  diagnostic (to confirm improvement).
- Convergence rate from the perturbed initialiser must be ≥ 90%.
- No regression in `test_detector.py`.

### Control gate

- **PASS:** Homography corners beat `_corners_from_rho` on perspective
  and do not regress frontoparallel cases.
- **FAIL:** Optimiser diverges, is slow (> 500 ms per finder), or degrades
  existing tests.
- **Action on fail:** Revert to `_corners_from_rho` default and tighten
  the initialiser / robust loss.

---

## Step 6 — Global Homography from Refined Finder Corners

### Goal

Replace the similarity-from-centres initialisation in `detector.py` with a
DLT homography fitted to all 12 refined finder corners.

### Implementation

1. In `detector.py::_run_detection`, after finder-pattern extraction:
   - Collect 12 correspondences: 4 canonical corners per finder × 3
     finders.
   - Fit `H_global` with DLT using `homography.dlt`.
   - Refine with LM on reprojection error using `homography.refine_homography_lm`.
2. Keep the old path behind `use_global_dlt_from_corners=False`.
3. Add a sanity check: condition number of the DLT design matrix.  If
   `cond > 500`, log a warning and fall back to the old similarity init.

### Deliverable

- Modified `_run_detection` with a flag `use_global_dlt_from_corners`.
- Diagnostic log line: `global_dlt_cond = ...`.
- Full-pipeline benchmark numbers for v12-default config.

### Evaluation

```bash
pytest src/qr_reader/tests/test_detector.py -q
python src/qr_reader/scripts/qr_benchmark.py
```

### Metrics

- v12-default detection rate must be ≥ the baseline recorded in §0.2.
- Global reprojection error on detected v12 samples must be < 15 px.
- No regressions in `test_detector.py`.
- DLT condition number must be < 500 for typical samples.

### Control gate

- **PASS:** Detection rate improves or stays the same; reprojection error
  meets target.
- **FAIL:** v12-default rate drops, condition numbers blow up, or tests
  regress.
- **Action on fail:** Re-enable the old similarity initialiser by default
  and investigate DLT degeneracy (likely need to add grid-edge points or
  weight finders by confidence).

---

## Step 7 — Integration, Cleanup, and Documentation

### Goal

Make the new pipeline the default, remove temporary flags, and update
project documentation.

### Implementation

1. Set defaults:
   - `estimate_anisotropic_pitch=True`
   - `use_two_families=True`
   - `use_projective_scanlines=True`
   - `use_finder_homography=True`
   - `use_global_dlt_from_corners=True`
2. Remove dead code paths only if all control gates in Steps 2–6 passed.
3. Update `AGENTS.md`:
   - Data Flow: reflect that per-finder homography refinement now happens
     before global homography estimation.
   - Module Map: add `refine_finder_homography`, `corners_from_homography`,
     `fit_scanline_projective`, `estimate_orientation_two_families`.
4. Update `README.md` Architecture section if it duplicates the Data Flow.
5. Run the `doc-maintenance` skill to audit for drift.

### Deliverable

- Clean branch with no feature flags in production code.
- Updated `AGENTS.md` and `README.md`.
- Final benchmark report saved to `docs/plan-015-baseline/final_report.md`.

### Evaluation

```bash
pytest src/qr_reader/tests -q
python src/qr_reader/scripts/qr_benchmark.py
python src/qr_reader/scripts/full-pipeline.py
```

### Metrics

- All tests pass (same or better than baseline).
- v12-default detection rate ≥ baseline.
- Single-finder corner RMSE at 30° perspective < 5 px.
- Global reprojection error < 15 px on detected v12 samples.

### Control gate

- **PASS:** All metrics meet targets and documentation is consistent.
- **FAIL:** Any metric misses target or docs drift.
- **Action on fail:** Do **not** merge.  Return to the failing step,
  keep feature flags on, and fix the issue.

---

## 8. Revert Policy

Every step has a feature flag.  The branch must remain in a working state
after each step.  If a control gate fails:

1. Set the new feature flag to `False` (reverting to the previous working
   path).
2. Fix the issue in the next commit.
3. Re-run the evaluation before moving to the next step.

Do not proceed to Step `N+1` until Step `N` passes its control gate with
the new path enabled.

---

## 9. Definition of Done

- [ ] Baseline recorded in `docs/plan-015-baseline/`.
- [ ] Single-finder perspective benchmark exists and runs.
- [ ] `m_u`/`m_v` bookkeeping implemented and passing.
- [ ] Two-family orientation estimator implemented and passing.
- [ ] 1D projective scanline fit implemented and passing.
- [ ] Per-finder homography refinement implemented and passing.
- [ ] Global DLT-from-corners path implemented and passing.
- [ ] New paths are default; old paths removed or cleanly deprecated.
- [ ] `AGENTS.md` and `README.md` updated; `doc-maintenance` skill run.
- [ ] Final test suite and benchmark pass with no regressions.

---

## 10. Risk Register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Two-family clustering unstable on sparse NMS | Medium | Robust fallback to 4-fold histogram; score threshold |
| Projective scanline RANSAC slow per finder | Medium | Limit iterations; early exit on high inlier count |
| Per-finder LM diverges from poor initialiser | Medium | Affine initialiser from Steps 2–4; bounds on perspective params |
| DLT degenerate for frontoparallel QR | Low | Condition-number guard; fallback to similarity init |
| Test suite regressions on v1 small codes | High | Run `test_detector.py` after every step; gate strictly |
