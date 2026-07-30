# Ray-Profile Finder Fitter — Integration Plan

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 0 | Benchmark current pipeline | ✅ done |
| 1 | Homography-based version estimation | ✅ done |
| 2 | New `detector/ray_fit.py` module | ✅ done |
| 3 | Unit tests for `ray_fit` | ⬜ pending |
| 4 | Replace `fit_finder_full` in `_run_detection` | ⬜ pending |
| 5 | Simplify `find_valid_triplets` | ⬜ pending |
| 6 | Full benchmark of new pipeline | ⬜ pending |
| 7 | Joint refinement benchmark | ⬜ pending |
| 8 | Remove deprecated code | ⬜ pending |

## Implementation instructions

Each phase is self-contained.  Follow this procedure:

1. **Read the phase description** below — understand the deliverable and quality gate.
2. **Implement** the changes.
3. **Run the quality gate** (tests, benchmark, or manual check as specified).
4. **Commit** with message `phase-N: <name>` (e.g. `phase-0: benchmark current pipeline`).
5. **Update the status table** above: change the phase's status from `⬜ pending` to
   `✅ done`.  If the phase encountered surprises, add a brief note in the
   **Phase Notes** section at the bottom of this document — anything that a future
   agent implementing a later phase needs to know (e.g. "benchmark harness uses
   synch config presets; path `data/images/train` must exist", "decoder test
   fixtures assume V≥2; V=1 fixtures were added in Phase 1").

Do NOT skip the commit step.  Do NOT start a phase before committing the previous one.

---

Replace the NMS-edge-based `finder_fit.fit_finder_full` path in `detector._run_detection`
with the ray-profile-based edge-fitting pipeline (`ray-profile.py` + `edge_fitting.py`),
which estimates finder corners from radial intensity profiles.

## Glossary

**RayFitResult** — New minimal dataclass returned by `fit_finder_ray`.
Fields: `corners` (4×2 float64, x,y), `score` (float), `valid` (bool).

**Cluster Concentration** — Fraction of valid boundary points assigned to the top-4
edge clusters.  Used as a false-positive filter: if the top 4 clusters contain less
than `min_concentration_ratio` of all valid points, the candidate is rejected.

## Parameters of `fit_finder_ray`

| Parameter | Default | Role |
|-----------|---------|------|
| `num_rays` | 36 | Number of equally-spaced ray directions |
| `num_samples` | 120 | Samples per ray |
| `roi_scale` | 1.5 | Scale factor for `cluster_to_bbox` |
| `sigma` | 1.0 | Edge softness for template fitting |
| `max_gap` | 1 | Max cyclic gap in Phase 0 clustering |
| `distance_threshold` | 0.1 | Single-linkage distance threshold |
| `min_concentration_ratio` | 0.5 | False-positive filter threshold |
| `refine_joint` | False | Run optional Step 4 joint refinement |

Constants (not tunable): `ray_length=1.0`, `mask_boundary=4.5`, `pitch_constant=3.5`.

---

## Phase 0 — Benchmark current pipeline

**Deliverable**: Script `scripts/qr_benchmark.py` (rewritten).

Sweep: 3 presets (easy, medium, hard) × {1, 3, 5, 8, 12, 20} versions × 10 seeds =
180 images.  Each config: `generate_sample` with background from `data/images/train`.

Metrics per image:
- Detection success (`detect_corners` did not raise)
- Corner error (mean px distance detected vs GT, when both succeed)
- Decode success (`decode()` matched payload)
- Wall time (detection only, `perf_counter`)

Output: JSON file `benchmark_current.json` with per-config metrics, and a Markdown
summary table.

**Quality gate**: All 180 images benchmarked; results saved.

---

## Phase 1 — Replace version estimation in `_run_detection`

Remove per-finder module-pitch (`m`) from the pipeline entirely.  Instead, estimate
version directly from the homography by scoring the timing pattern at candidate
N values.

**Changes to `detector.py:_run_detection`**:

1. Drop `m_avg`, `dx`, `dy`, `dh`, `s_hat`, `N_est`, `N_legal` computation.
2. Instead, iterate over candidate `N ∈ {21, 25, 29, ..., 177}`, for each:
   - Build 12-point DLT correspondences from the three finder corners (existing logic,
     scaled to `N`).
   - Fit homography.
   - Score via `_score_timing_pattern` (already exists).
   - Keep best `(H, N)` by timing score.
3. Remove `fit_map[idx].m` references from the dedup loop (use `.score` only).

`find_valid_triplets` still takes `fit_map` but only for `.score` and `.m` in the
module-size compatibility check.  For now, keep the existing call; the version
estimate is done *after* triplet finding.

**Quality gate**: `test_detector.py` tests pass.  Re-benchmark (Phase 0 script), compare
detection rate — should be at least as good.

---

## Phase 2 — New module `detector/ray_fit.py`

**Deliverable**: New file `detector/ray_fit.py` with single public function:

```python
@dataclass
class RayFitResult:
    corners: np.ndarray   # (4, 2) float64, (x, y) corners of the finder pattern
    score: float          # quality score (higher = better)
    valid: bool           # False if concentration check failed

def fit_finder_ray(
    roi: np.ndarray,        # Grayscale ROI (H, W), uint8
    center_xy: np.ndarray,  # (2,) float64, (x=col, y=row) estimated centre
    m_est: float,           # Estimated module pitch from cluster width
    num_rays: int = 36,
    num_samples: int = 120,
    roi_scale: float = 1.5,
    sigma: float = 1.0,
    max_gap: int = 1,
    distance_threshold: float = 0.1,
    min_concentration_ratio: float = 0.5,
    refine_joint: bool = False,
) -> RayFitResult:
    ...
```

**Internal pipeline**:

1. `normalize_roi_intensities(roi, center_xy, m_est)` → normalized ROI [0, 1]
2. `sample_ray_profiles(norm_roi, center_xy, num_rays, num_samples, ray_length=1.0)` → `profiles`
3. `fit_all_rays(profiles, m_est, max_dist, ...)` → per-ray `m` estimates
4. `compute_boundary_points(center_xy, m, theta, pitch_constant=3.5)` → boundary points
5. `fit_finder_edges(boundary_points, max_gap, distance_threshold, k=4)` → `EdgeFitResult`
6. **False-positive filter**: `sum(len(clusters[i].support) for i in range(4)) / len(points) < min_concentration_ratio` → `RayFitResult(valid=False)`
7. `assign_points(clusters, len(points))` → `assignment`
8. (If `refine_joint`) → `refine_finder_edges_joint(...)` → refined clusters
9. Compute corners from the 4 edge lines (intersect adjacent edges via `compute_corners`)
10. Score: fraction of rays with valid m fits × (1 − mean σ₂/σ₁ of top-4 clusters)

Move helper functions from `ray-profile.py` into `ray_fit.py`:
- `sample_ray_profiles`
- `normalize_roi_intensities`
- `fit_m_half_ray`, `fit_all_rays`
- `_masked_mse`, `finder_soft_template`

**Quality gate**: Module imports cleanly; no runtime errors on synthetic input.

---

## Phase 3 — Unit tests

**Deliverable**: New file `tests/detector/test_ray_fit.py`.

Tests:

1. **`sample_ray_profiles`** — Synthetic 50×50 ROI with known gradient; verify
   sampled values at known (angle, distance) match bilinear interpolation.
2. **`normalize_roi_intensities`** — Two-tone ROI (dark center, bright border);
   verify dark ≈ 0, bright ≈ 1.
3. **`fit_all_rays`** — Synthetic 1D profiles from `finder_soft_template` at known
   `m` with Gaussian noise; verify median fitted `m` within 5% of ground truth.
4. **Integration: synthetic square** — 72×72 ROI with a perfect dark/light finder
   pattern (3:1:1:1 ratio), known corners; `fit_finder_ray` → corner error < 1 px.
5. **Integration: clean QR image** — `make_qr_image(version=1); detect_corners` →
   version == 1, decode succeeds.
6. **Concentration filter** — Boundary points forming a circle (no straight edges) →
   `fit_finder_ray` returns `valid=False`.

**Quality gate**: All tests pass with `pytest tests/detector/test_ray_fit.py`.

---

## Phase 4 — Replace finder fitting in `_run_detection`

Replace per-cluster block in `detector.py:_run_detection` (lines 110–137):

**Before**:
```python
nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
fit = fit_finder_full(nms, angle, roi, center_xy, m_est)
corners_xy_global = fit.corners + offset
fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
fit_map[ci] = fit
```

**After**:
```python
result = fit_finder_ray(roi, center_xy, m_est, ...)
if not result.valid:
    continue
corners_xy_global = result.corners + offset
fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
score_map[ci] = result.score
```

Remove imports of `extract_thin_edges`, `fit_finder_full`, `FinderFit`.
Add import of `fit_finder_ray`, `RayFitResult`.

Propagate `num_rays`, `num_samples`, `roi_scale`, `sigma`, `max_gap`,
`distance_threshold`, `min_concentration_ratio`, `refine_joint` as optional
parameters of `_run_detection` (and `detect_corners` etc.), with the defaults
from Phase 2.

Dedup loop: replace `fit_map[fp.cluster_idx].score` with `score_map.get(fp.cluster_idx, 0.0)`.

**Quality gate**: `test_detector.py` tests pass (these use clean `make_qr_image` images).

---

## Phase 5 — Simplify `find_valid_triplets`

**Changes to `finder_pattern.py:find_valid_triplets`**:

Signature change:
```python
def find_valid_triplets(
    fps: list[FinderPattern],
    score_map: dict[int, float],
    ...
) -> list[Triplet]:
```

Replace:
- `fit_map[idx].m` → compute `m` from `fp.outer_corners` side length / 7
- `fit_map[idx].center` → `fp.outer_corners.mean(axis=0)` (already used)
- `fit_map[idx].e1`, `fit_map[idx].e2` axis-alignment check → simpler check:
  inter-center vector dot product with corner-edge vectors (prefer alignment
  within 20° of one of the four corner-edge directions)
- Module-size compatibility: `abs(ma - mb) / max(ma, mb)` → corner-area ratio
  `abs(area_i - area_j) / max(area_i, area_j) < 0.5`

Update caller in `_run_detection` to pass `score_map` instead of `fit_map`.

**Quality gate**: Existing `test_finder_pattern.py` tests pass or are adjusted.
Phase 0 benchmark re-run; detection rate matches Phase 1.

---

## Phase 6 — Full benchmark of new pipeline

Run Phase 0 benchmark script against the new pipeline.  Produce:

- `benchmark_ray.json` (raw results)
- `benchmark_comparison.md` — side-by-side table:

  | Preset | Version | Seeds | Old Detect | New Detect | Old Corn Err | New Corn Err | Old Decode | New Decode | Old Time | New Time |
  |--------|---------|-------|------------|------------|--------------|--------------|------------|------------|----------|----------|

Aggregate summaries by preset and version.

**Quality gate**: New pipeline meets or exceeds old pipeline on:
- Detection rate: within 5% (absolute)
- Corner error: within 1 px mean
- Decode success: within 5% (absolute)

If it does not, stop and diagnose before proceeding.

---

## Phase 7 — Optional: Joint refinement benchmark

Run the benchmark script with `refine_joint=True` vs `refine_joint=False`.
Measure:

- Corner accuracy improvement (mean px)
- Wall-time overhead (ms per cluster)

**Deliverable**: Markdown table with the comparison.  Decision on whether
to make `refine_joint=True` the default (or remove the option entirely)
recorded as an ADR.

**Quality gate**: Clear recommendation written.

---

## Phase 8 — Remove deprecated code

Files to remove:
- `detector/finder_fit.py`
- `scripts/full-pipeline.py` (old)
- `scripts/full-pipeline-profile.py` (old diagnostic)

Files to update:
- `detector/detector.py` — remove `FinderFit`, `fit_finder_full`, `extract_thin_edges` imports
- `detector/__init__.py` — remove `finder_fit` exports if any
- `scripts/full-pipeline-current.py` — update to use ray-based pipeline
- `AGENTS.md` — update module map, data flow, key data structures
- `CONTEXT.md` — add `RayFitResult`, remove `FinderFit` glossary entry
- `README.md` — update architecture section

**Quality gate**: Full test suite passes.  `git grep finder_fit` returns zero results
in `src/`.  `AGENTS.md` audit via `doc-maintenance` skill.

---

## Summary of files changed/created

| File | Action |
|------|--------|
| `scripts/qr_benchmark.py` | Rewrite |
| `detector/detector.py` | Modify `_run_detection` (Phases 1, 4) |
| `detector/finder_pattern.py` | Modify `find_valid_triplets` (Phase 5) |
| `detector/ray_fit.py` | **New** (Phase 2) |
| `tests/detector/test_ray_fit.py` | **New** (Phase 3) |
| `tests/detector/test_finder_pattern.py` | Adjust for new signature (Phase 5) |
| `scripts/full-pipeline-current.py` | Update (Phase 8) |
| `detector/finder_fit.py` | **Remove** (Phase 8) |
| `scripts/full-pipeline.py` | **Remove** (Phase 8) |
| `scripts/full-pipeline-profile.py` | **Remove** (Phase 8) |
| `AGENTS.md` | Update (Phase 8) |
| `CONTEXT.md` | Update (Phase 8) |

---

## Phase Notes

*Populated by implementers as each phase completes.  Notes here inform subsequent phases.*

### Phase 0
- No `generate_sample` function exists; used `generate_test_image` with preset parameter values instead.
- GT corner error computed by re-running deterministic RNG chain (rotation + perspective) from same seed.
- Benchmark JSON at `benchmark_current.json` (74KB, 180 results).
- Detection rate: ~38-47% depending on preset. When it works, corner error typically ~0.6-1.5px.
- V=1 often misdetected as V=2 (a known finder_fit issue); many high-version detections have large corner errors (>100px).

### Phase 1
- Removed `m_avg`, `dx`, `dy`, `dh`, `s_hat`, `N_est`, `N_legal` computation from `_run_detection`.
- Full-range N search (21-177 step 4) with `combined_err = err - timing` scoring.
- Fixed `_score_timing_pattern`: changed `np.median` → `np.mean` threshold; median failed when dark/light count was unbalanced (e.g., 5 dark/4 light in V=2 timing pattern), causing zero alternation score on clean images.
- `fit_map[idx].m` no longer used in `_run_detection`; `fit_map` still passed to `find_valid_triplets` (Phase 5 will change that).

### Phase 2
- Created `detector/ray_fit.py` with `RayFitResult` dataclass and `fit_finder_ray` public function.
- Moved helpers from `ray-profile.py`: `sample_ray_profiles`, `normalize_roi_intensities`, `finder_soft_template`, `_masked_mse`, `fit_half_ray`, `fit_all_rays`.
- Wired into `edge_fitting.py` Phases 0–4 (clustering, TLS, assignment, joint refinement).
- `roi_scale` parameter accepted but ignored (ROI scaling done by caller via `cluster_to_bbox`).
- Score formula: `frac_valid * (1 - mean_sigma_ratio)`.
- Detection rate: ~38-47% depending on preset. When it works, corner error typically ~0.6-1.5px.
- V=1 often misdetected as V=2 (a known finder_fit issue); many high-version detections have large corner errors (>100px).

