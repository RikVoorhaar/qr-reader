# Updated Plan — Literature-Backed Fix for Per-Finder Homography

## Diagnosis (confirmed by literature)

The research report confirms two architectural mistakes:

1. **Per-finder homography as final output** — Three independent 8-DOF projective warps destroy inter-finder colinearity. No production QR detector (ZXing, quirc, OpenCV) allows per-finder homographies to define the final corners used for association. Even quirc, which computes a local 7×7 capstone transform, uses it only to measure center and corner order, then forces everything into a common global frame.

2. **Solving π/2 ambiguity at the single-finder level** — A finder pattern has 4-fold rotational symmetry; no local measurement can disambiguate the quadrant without global context. All production detectors resolve orientation at the triplet/global stage (e.g. ZXing's `orderBestPatterns()`, quirc's `rotate_capstone()`).

3. **Pairwise colinearity as the association mechanism** — Every production detector uses triplet-level geometric validation (isosceles right triangle, module-size compatibility, grid fitness) rather than testing whether two independently warped segments are colinear.

The architectural pattern from all three detectors is: **local measurements → global QR hypothesis → single global projective transform**.

---

## Amendment Plan

### Phase 1 — Strip per-finder homography from final corners

**File:** `finder_fit.py`

Keep steps 1–7 of `fit_finder_full` (orientation, profiles, projective scanlines, outer line refinement) as measurement extraction. Stop using per-finder homography corners as the output. The output corners revert to `extract_finder_corners_from_rho` (rho-based line intersections).

Remove from `FinderFit`: per-finder homography refinement call, `_align_quad_order`, `corners_from_finder_homography` from the hot path. Optionally preserve `refine_finder_homography` as dead code or a utility function for later global refinement.

The `_align_quad_order` and per-finder homography corner are not needed — they were the proximate cause of the association failure.

**Expected result:** Association works again on clean images. Rho corners are axis-aligned in the (e1,e2) frame, forming exact squares. The exhaustive axis pairing in `check_association` handles the π/2 rotation between finders' frames.

### Phase 2 — Replace pairwise colinearity association with triplet-center validation

**File:** `finder_pattern.py` (new), `detector.py`

Instead of `check_association` testing segment colinearity, build association based on finder centers and local module pitches:

1. For every pair of finder clusters, accept as "associated" if their centers are within a loose bounding box (e.g. share approximate row or column within 3× module pitch tolerance of each other).

2. For every pair, compute the normalized inter-center vector and check that the two finders' e1/e2 frames are approximately aligned with it (one axis should be nearly parallel to the inter-center line, the other nearly perpendicular). Don't resolve which is which yet.

3. Find triplets by checking if center A is connected to B and C, and the angle ∠BAC is approximately 90° (isosceles right triangle test). Also check module-size compatibility: `|m_a - m_b| / max(m_a, m_b) < 0.3` for all three pairs.

4. Once a triplet is identified, resolve the π/2 ambiguity by picking a consistent e1/e2 assignment per finder using the inter-center vectors (analogous to quirc's `rotate_capstone`). The top-left finder's e1 should point toward the top-right finder, etc.

This replaces the `local_offset` / `angular_distance` segment-pair tests entirely. The `check_association` function can be removed or repurposed.

### Phase 3 — Use per-finder measurement to seed a single global homography

**File:** `detector.py`, `homography.py`

The global homography is already computed from 12 finder corners (3 finders × 4 rho corners) via DLT + LM. This stays. The improvement: use per-finder measurements (center, e1, e2, m) to validate the version estimate more robustly, and optionally use per-finder homography as an **initialization** for the global homography (project centers through a 4-point DLT, not independent per-finder warps).

If we want to retain the per-finder homography's refinement benefit, do so in a global context: after the global DLT+LM, optionally refine by comparing projected corners against local edge pixels on all three finders simultaneously — a single global LM over a single homography, weighted by all three finder ROIs. This is analogous to quirc's global grid fitness.

### Phase 4 — Fix or remove two-family von-Mises EM

**File:** `finder_fit.py`

The EM is unstable on symmetric data. Options:
- **Remove it** and use only the 4-fold histogram (simplest, consistent with Phase 2's global disambiguation).
- **Replace with a deterministic axial estimator** (Bingham or simple peak-finding on the doubled-angle histogram after subtracting the dominant mode).
- **Fix the EM** by debugging initialization and convergence on synthetic data.

Given that Phase 2 resolves quadrant ambiguity globally, the 4-fold histogram alone is sufficient. The two-family EM's benefit (independent n_u, n_v for gating) has not been demonstrated to improve the downstream pipeline on benchmarks.

**Recommendation:** Remove the EM, use only `estimate_orientation` for per-finder frames. If perspective gating proves valuable later, the EM can be replaced with a simpler deterministic method (find the two peaks in the doubled-angle histogram and assign them based on proximity to φ).

### Phase 5 — Optionally constrain per-finder homography for global refinement

**File:** `finder_fit.py`, `homography.py`

If we want per-finder homography to contribute to the global fit (rather than being dropped entirely), restrict it to a measurement role:

- Compute per-finder homography from rho corners + center (this is already the `H_init` in the current code).
- Do NOT use LM-refined corners for association.
- For the global homography, optionally include per-finder homography as a prior or use its fit error as a quality score for the global fit.
- OR: constrain the per-finder homography to a 4-DOF model (similarity: scale + rotation + translation) which cannot introduce perspective shear that breaks colinearity.

This is lower priority — the rho-based corners + global DLT+LM may already achieve good accuracy. The perspective finder test suite showed per-finder homography improves local RMSE by 37%, but this may be recoverable in the global DLT+LM.

---

## File Change Map

| Phase | File | Change |
|-------|------|--------|
| 1 | `finder_fit.py` | Remove per-finder homography from `fit_finder_full`; output rho corners |
| 1 | `finder_fit.py` | Update `FinderFit` docstring (corners explanation) |
| 2 | `finder_pattern.py` | Add `find_valid_triplets(fps)` using center geometry + module-size compatibility |
| 2 | `detector.py` | Replace `find_all_associations` + `find_triplets` with new triplet finder |
| 2 | `finder_pattern.py` | Add `resolve_finder_orientation(triplet, fps, fits)` to disambiguate π/2 |
| 3 | `detector.py` | Global DLT+LM stays; optionally refine with multi-finder edge pixels |
| 4 | `finder_fit.py` | Remove `estimate_orientation_two_families` from hot path |
| 4 | `finder_fit.py` | Remove `n_u, n_v` from `FinderFit` fields (or deprecate) |
| 5 | `finder_fit.py` | Optional: add constrained 4-DOF or measurement-only role for homography |
| — | `test_finder_perspective.py` | Update tests for removed functionality |
| — | `test_detector.py` | Re-enable passing tests |
| — | `AGENTS.md`, `README.md` | Update Data Flow and module descriptions |

---

## Expected Results

| Test suite | Before | After |
|-----------|--------|-------|
| `test_finder_perspective.py` | 25 passed, 4 skipped | Slight reduction (EM + homography tests removed) |
| `test_detector.py` | 16 failed, 2 passed | 16+ passed |
| `qr_benchmark.py` | Crash (no triplet) | Runs, decodes correctly |
| Full `pytest` | 724 passed, 18 failed | 735+ passed, ~7 failed |

---

## Deferrals

- **Global multi-finder edge-pixel refinement** (Phase 3 optional): skip for now; the existing 12-point DLT+LM is likely sufficient for v1-v12 benchmarks.
- **Per-finder homography as global prior** (Phase 5): lower priority; can be added if global RMSE needs improvement.
- **Timing-pattern and alignment-pattern scoring** (quirc-style global fitness): out of scope for this fix; existing version estimation already works.
