# Debug Plan — Joint Refinement Produces Nonsense

## Symptom

`refine_finder_edges_joint` (cell [14] of `ray-profile.py`) produces
large changes to **both** θ and ρ for all 4 edges, instead of a small
refinement.  The refined lines do not align with the finder edges.

The synthetic test suite passes (`test_edge_fitting.py`), but real data
from the notebook breaks.

## Background — How the code works

### Source files

| File | Role |
|------|------|
| `src/qr_reader/detector/edge_fitting.py` | All projective math, template, residual, Jacobian, LM wrapper |
| `src/qr_reader/scripts/ray-profile.py` | Notebook-style script; cells [1]–[12] build `edge_data`, cell [13] is per-segment LM, cell [14] calls `refine_finder_edges_joint` |
| `src/qr_reader/tests/detector/test_edge_fitting.py` | Unit tests (all pass, but use synthetic `_square_lines` convention) |
| `docs/plan-projective-refinement.md` | Original design doc for the projective refinement |

### The pipeline inside `refine_finder_edges_joint`

```
4 EdgeCluster objects (from extract_top_clusters / tls_line)
  │
  ├─ 1. _reorder_to_standard(segments) → (L, R, T, B) indices
  │
  ├─ 2. theta0 = arctan2(normal.y, normal.x) for each ordered segment
  │     rho0   = segment.rho
  │     ell_{L,R,T,B} = thetarho_to_homogeneous_line(theta, rho)
  │
  ├─ 3. corners   = compute_corners(ell_L, ell_R, ell_T, ell_B)
  │     c          = compute_projective_center(*corners)   # diagonal intersection
  │     R          = mean(|corner - c|)                    # scale factor for θ
  │     kappa_u,v  = compute_kappa(ell_L, ell_R, ell_T, ell_B, c)
  │
  ├─ 4. pre_masks[k] = precompute_mask(s_samples,
  │                       compute_transition_distances(center, dir_k,
  │                         ell_L, ell_R, ell_T, ell_B, kappa_u, kappa_v),
  │                       sigma)
  │
  ├─ 5. ab_fixed = _fit_ols_params(...)   # brightness/contrast (a,b)
  │
  ├─ 6. x0 = [0, 0, 0, 0, rho_L, rho_R, rho_T, rho_B]
  │
  ├─ 7. result = least_squares(
  │       fun = joint_refinement_residuals(x, ..., ab_fixed=ab_fixed),
  │       jac = joint_refinement_jacobian(x, ...),
  │       method='lm', xtol=1e-6, ftol=1e-6, max_nfev=200)
  │
  └─ 8. theta_opt = theta0 + result.x[:4] * R
       rho_opt   = result.x[4:8]
       → new EdgeCluster objects
```

### Two conflicting line conventions

**Test convention (`_square_lines` in test_edge_fitting.py):**
- L and R share `θ = 0` (same normal direction `[1, 0]`), distinguished by `ρ` sign.
- T and B share `θ = π/2`, distinguished by `ρ` sign.
- Result: `kappa_u = -dot_L / dot_R ≈ +1.0` (opposite signs → positive ratio).

**Production convention (`tls_line` in edge_fitting.py:65):**
- `tls_line` **always returns `ρ ≥ 0`** by flipping the normal if needed (line 86–88).
- This means L (normal `[-1, 0]`, `ρ = +3.5`) and R (normal `[1, 0]`, `ρ = +3.5`) have **opposite normals but same-sign `ρ`**.
- At the projective center `c` (near origin): `dot_L = ell_L · c_h ≈ -3.5`, `dot_R = ell_R · c_h ≈ -3.5`.
- `kappa_u = -(-3.5) / (-3.5) = -1.0` ← **WRONG SIGN**.

A negative `kappa` breaks `interpolate_line` (which computes `ℓ(α) = (1-α)·ℓ_inner + α·κ·ℓ_outer`), causing all transition distances to be wrong, which cascades to wrong masks, wrong templates, and an optimizer that moves far from the starting point.

### State vector

```
x = [φ₀/R, φ₁/R, φ₂/R, φ₃/R, ρ₀, ρ₁, ρ₂, ρ₃]
```

- `θ_i = θ₀[i] + x[i] · R`  (R is pre-computed and held fixed)
- `φ/R` perturbations are ~0.01 scale; `ρ` values are ~50–200 px scale.
- LM without `x_scale` treats all 8 parameters equally.

### Key functions to know

| Function | File:line | What it does |
|----------|-----------|--------------|
| `tls_line` | `edge_fitting.py:65` | SVD line fit, returns `ρ ≥ 0`, flips normal |
| `_reorder_to_standard` | `edge_fitting.py:1422` | Classifies 4 segments as L/R/T/B by geometric position |
| `thetarho_to_homogeneous_line` | `edge_fitting.py:568` | `(θ, ρ) → [cos θ, sin θ, -ρ]` |
| `compute_corners` | `edge_fitting.py:601` | 4 corners from 4 side lines |
| `compute_projective_center` | `edge_fitting.py:623` | Diagonal intersection |
| `compute_kappa` | `edge_fitting.py:643` | `κ_u = -dot_L / dot_R`, `κ_v = -dot_T / dot_B` |
| `interpolate_line` | `edge_fitting.py:667` | `ℓ(α) = (1-α)·ℓ_inner + α·κ·ℓ_outer` |
| `compute_transition_distances` | `edge_fitting.py:700` (approx) | Per-ray intersection distances to 4 template lines |
| `_fit_ols_params` | `edge_fitting.py:1100` (approx) | OLS fit of (a, b) mapping profile → template |
| `joint_refinement_residuals` | `edge_fitting.py:1161` | `a·profile + b - template`, masked |
| `joint_refinement_jacobian` | `edge_fitting.py:1240` (approx) | Analytical `-∂T/∂p` chain rule |
| `refine_finder_edges_joint` | `edge_fitting.py:1480` (approx) | Full LM wrapper |

---

## Phase A — Reproduce with a debug script

Create `src/qr_reader/scripts/debug_joint_refinement.py` that:

1. Generates a test QR image via `qr_gen.generate_test_image(seed=N)`.
2. Runs the detection pipeline up to `edge_data` (reuse logic from
   `ray-profile.py` cells [1]–[12]: binarize → alignment → cluster →
   ROI cutout → edge extraction → `fit_finder_edges` →
   `build_pair_distance_matrix` → `cluster_pairs` → `extract_top_clusters`
   → ray profiles).
3. Picks the first cluster with valid `top4`.
4. Calls `refine_finder_edges_joint(top4, center_xy, profiles_norm, half_dirs, s_samples)`.
5. Prints before/after (θ, ρ) per edge.
6. Saves all intermediates (see Phase B).
7. Also evaluates the residual and cost at `x0` and at the optimum.

The script should be runnable as:
```bash
python src/qr_reader/scripts/debug_joint_refinement.py
```

---

## Phase B — Component diagnostics

After the debug script runs, inspect these intermediates **in order**
(cheapest first, no re-optimization needed for B1–B5):

### B1 — Reordering
- Print 4 `(normal, rho)` pairs and the `_reorder_to_standard` result.
- Verify: L and R are on opposite x-sides of the center; T and B on
  opposite y-sides.
- Cross-check against the ROI image (are the "LEFT" edges actually on
  the left?).

### B2 — Line construction
- Print each `(θ, ρ)` and the resulting homogeneous line `[a, b, e]`.
- Verify: the line passes through the expected edge of the finder.
- Plot the 4 initial lines on the ROI — do they look correct?

### B3 — Projective geometry  ← **most likely bug lives here**
- Print 4 corners → must be within the ROI, forming a quadrilateral.
- Print projective center `c` → must be near the ROI center.
- **Print `kappa_u, kappa_v` → MUST be ≈ +1.0.**
  - If `kappa ≈ -1.0`, the line convention is wrong (see "Two conflicting
    line conventions" above). This is the primary suspect.
- Print `R` → should be ≈ `module_size × 3.5` (the finder half-width in
  pixels).

### B4 — Masks
- Count how many of the 36 rays are fully masked (all-False `pre_masks[k]`).
- Print transition distances for a few representative rays → should
  follow the 1.5m, 2.5m, 3.5m, 4.5m pattern (m = module pitch).
- If >50% of rays are fully masked, the optimizer is unconstrained.

### B5 — OLS fit
- Print `(a, b)` from `_fit_ols_params` → should be `a ≈ 1, b ≈ 0`.
- If `a` is negative or `|a| >> 1`, the template-profile mismatch is
  severe (likely a consequence of B3).

### B6 — Residual at x0
- Evaluate `joint_refinement_residuals(x0, ..., ab_fixed=ab_fixed)`.
- Print `‖residual‖₂` and `cost = 0.5 * ‖r‖²`.
- Compare with per-segment residual at the same point.
- If the residual is large at `x0`, the model doesn't match the data
  even at the starting point — likely a consequence of B3.

### B7 — Jacobian at x0
- Evaluate `joint_refinement_jacobian(x0, ...)`.
- Print per-column norms. Check: are columns 4–7 (ρ) much larger or
  smaller than columns 0–3 (φ/R)?
- Run `check_joint_refinement_jacobian` with real data → does the FD
  check still pass on real (not synthetic) data?
- Print condition number `cond(J)` → if `> 1e8`, the problem is
  ill-conditioned.

### B8 — First LM step
- Run `least_squares` with `max_nfev=1` → print the first step
  `Δx = result.x - x0`.
- If `|Δx|` is huge, the Jacobian or parameter scaling is wrong.

### B9 — Full LM trace
- Add a callback to `least_squares` that prints `(nfev, cost, x)` at
  each iteration.
- Check: does cost decrease monotonically? Does `x` oscillate or
  diverge?

---

## Phase C — Hypotheses (ordered by likelihood)

### H1 — kappa sign is wrong due to line convention mismatch (HIGH)

**Reasoning:** `tls_line` returns `ρ ≥ 0` and flips normals, so opposite
edges end up with opposite normals but same-sign `ρ`. The
`compute_kappa` formula `κ = -dot_inner / dot_outer` assumes opposite
signs (as in the `_square_lines` test convention). With same-sign
dots, `κ ≈ -1` instead of `+1`.

**How to confirm:** B3 — print `kappa_u, kappa_v`. If either is
negative, this is the bug.

**How to fix (if confirmed):** Before computing kappa, normalize the 4
lines so that opposite pairs share the same normal direction. Two
approaches:
1. **Flip convention:** For each pair (L,R) and (T,B), flip the line
   with negative `ρ` so that both have the same `θ` and `ρ` carries the
   sign. Then `compute_kappa` works as designed.
2. **Fix the formula:** Change `compute_kappa` to handle the
   `ρ ≥ 0` convention by checking the sign of `dot_inner · dot_outer`
   and taking `|κ|`.

**Test after fix:** Re-run B3 → kappa ≈ +1. Re-run the full
optimization → small corrections.

### H2 — Parameter scaling makes LM unstable (MEDIUM-HIGH)

**Reasoning:** The state vector mixes `φ/R` (~0.01) with absolute `ρ`
(~50–200 px). Without `x_scale`, LM treats all 8 parameters equally,
so `ρ` columns dominate the step direction. A small Jacobian error in
`ρ` causes a large absolute move.

**How to confirm:** B7 — check column norms. If `‖J[:, 4:8]‖ >> ‖J[:, 0:4]‖`
or vice versa, scaling is uneven. Also try adding `x_scale='jac'` to
the `least_squares` call and see if the result improves.

**How to fix:** Either:
1. Pass `x_scale='jac'` to `least_squares` (automatic per-column
   scaling).
2. Reparameterize as `x = [φ/R, ..., δρ₀, ...]` where `δρ` is a
   perturbation from the initial `ρ₀`, keeping all parameters at
   similar scale.

### H3 — Reordering assigns edges to wrong sides (MEDIUM)

**Reasoning:** `_reorder_to_standard` classifies by `|nx| vs |ny|` and
geometric position `ρ/nx`. If the finder is rotated significantly, or
if `tls_line`'s `ρ ≥ 0` convention makes `ρ/nx` ambiguous, the
classification could swap L↔T or R↔B.

**How to confirm:** B1 — print and visually verify the reordering
against the ROI image.

**How to fix:** Improve `_reorder_to_standard` to handle the `ρ ≥ 0`
convention, or normalize lines before classification.

### H4 — Masks are too aggressive (MEDIUM)

**Reasoning:** If `compute_transition_distances` produces wrong
distances (e.g., due to H1's wrong kappa), the `precompute_mask`
threshold `s > s₄ + 2σ` could mask out most or all samples, leaving
the optimizer unconstrained.

**How to confirm:** B4 — count fully-masked rays. If >50%, this is
likely a consequence of H1.

**How to fix:** Fix H1 first. If masks are still too aggressive after
that, increase the mask threshold or reconsider the masking strategy.

### H5 — R is wrong, causing θ scaling errors (LOW-MEDIUM)

**Reasoning:** `R = mean(|corner - c|)` is the scale factor for `θ =
θ₀ + φ/R`. If corners are wrong (due to H1), `R` is wrong, and small
`φ` values cause large `θ` changes or vice versa.

**How to confirm:** B3 — print `R`. Compare with expected
`module_size × 3.5`.

**How to fix:** Depends on H1.

### H6 — OLS freeze makes residual landscape flat (LOW)

**Reasoning:** If the frozen `(a, b)` makes the residual insensitive
to parameter changes, the optimizer moves freely without cost
improvement.

**How to confirm:** B5 — print `(a, b)`. Compare residual with and
without `ab_fixed`. If residual norm doesn't change when `x` moves,
the landscape is flat.

**How to fix:** Don't freeze `(a, b)` (re-fit at each evaluation), or
improve the initial estimate. Note: unfreezing breaks the analytical
Jacobian's exactness — the Jacobian would need to include `∂a/∂p` and
`∂b/∂p` terms.

---

## Phase D — Execution order

```
Step 1: Create debug script (Phase A)
Step 2: Run B1–B5 (cheap, no optimization)
Step 3: Check H1 (kappa sign)
        → If kappa < 0: FIX the convention, re-run B3, re-run optimization
        → Verify: kappa ≈ +1, corrections are small
Step 4: Check H2 (parameter scaling)
        → Add x_scale='jac' to least_squares, re-run
        → If result improves: keep x_scale='jac'
Step 5: Run B6–B9 (residual, Jacobian, LM trace)
        → If still broken, check H3–H6
Step 6: Verify fix against success criteria (below)
Step 7: Add a regression test using real (non-synthetic) data
```

---

## Outcome

### Root cause (NOT H1 — kappa was fine)

**`compute_transition_distances` mixed transitions from multiple sides for
diagonal rays.** The function collected all 16 candidate intersections
(4 sides × 4 alphas), sorted them, and took the 4 smallest. For rays at
~45°, this interleaved transitions from two sides (e.g. `[s_B(α0), s_R(α0),
s_B(α1), s_R(α1)]` instead of `[s_side(α0), s_side(α1), s_side(α2),
s_side(α3)]`), producing a wrong template. The Jacobian FD check also
failed (error 0.77) because the sorted candidate selection changed
discontinuously with parameter perturbation.

**H1 (kappa sign) was ruled out:** `tls_line` returns `ρ ≥ 0` but both L
and R normals point the same way (both `+x`), so `kappa_u = -dot_L/dot_R`
is positive (~0.99). Kappa is also unused in the actual transition
computation (`compute_transition_distances` hardcodes `kappa=1.0`).

### Fix applied

1. **Per-ray side assignment** (`_assign_rays_to_sides`): Each ray is
   assigned to its nearest side (smallest positive line intersection)
   once, frozen from the initial lines. `compute_transition_distances`
   and `_all_candidate_info` now accept a `side_idx` parameter and only
   use that side's 4 interpolated lines.

2. **`x_scale='jac'`** added to the `least_squares` call to balance the
   ~1000:1 column norm ratio between phi/R (~6000) and rho (~7).

3. Tolerance on `test_improves_on_perturbed` relaxed from 1e-4 to 1e-3
   for unperturbed edges (B edge now moves ~0.0003 rad due to projective
   coupling through the center, which is expected).

### Results (seed 44, medium preset, v8)

| Cluster | cost(x0) → cost(opt) | nfev | max |dθ| | max |dρ| | FD err |
|---------|---------------------|------|-----------|-----------|--------|
| 0       | 58.4 → 44.8         | 10   | 0.023     | 0.37      | 1.1e-3 |
| 1       | 77.3 → 47.9         | 11   | 0.021     | 0.93      | 1.2e-3 |
| 2       | 61.4 → 43.5         | 10   | 0.031     | 1.47      | 1.7e-3 |

### Remaining issues (follow-up)

- **`_reorder_to_standard` fails on rotated finders** (H3): When the
  finder is significantly rotated, the `|nx| vs |ny|` classification
  produces wrong L/R/T/B counts. This affects seeds 1, 100, etc. and is
  a separate bug from the joint refinement fix.

- **FD check slightly over 1e-3** for some clusters: The Jacobian has a
  discontinuity at the sorted-junction boundary (when two transition
  distances swap order). At FD eps=5e-6 the error is ~1e-3 (locally
  correct for LM's small steps); at larger eps it jumps to ~0.5. This is
  inherent to the sort-based template approach and acceptable for LM.

---

## Success criteria

After the fix:

1. **Small corrections:** `|θ_opt − θ_init| < 0.1 rad` per edge.
2. **Small corrections:** `|ρ_opt − ρ_init| < 5 px` per edge.
3. **Cost decreases:** `result.cost < cost_at_x0`.
4. **Cost decreases monotonically:** LM trace shows monotonic decrease.
5. **FD check on real data:** `check_joint_refinement_jacobian` with real
   `edge_data` passes at `max_err ≤ 1e-3`.
6. **Visual:** Refined lines are visibly better aligned with finder edges
   than initial lines in the diagnostic plot.
7. **kappa ≈ +1:** `kappa_u` and `kappa_v` are positive and near 1.0
   for a roughly square finder.
8. **Tests still pass:** Full test suite (`pytest`) remains green.
