# Plan 007 — Hough + Refine Phased Fixes

> **Goal:** Improve `hough_vote_peaks` + `refine_line` one atomic change at a time.
> Each phase targets a single failure mode, has a binary accept/revert gate,
> and leaves the codebase in a working state whether kept or discarded.

---

## Gate rules

Every phase follows the same pattern:

1. **Implement the fix** in `src/qr_reader/detector/hough.py`
2. **Flip one isolation test** — the "assert the bug exists" assertion inverts
   to "assert the fix works."  That isolation test must now **pass** (the bug
   is gone) for the phase to proceed.
3. **Run the fixture tests** — count failure-mode tallies before and after.
   The targeted mode must show a measurable reduction (≥1 fewer failure).
4. **Run the full test suite** — `pytest src/qr_reader/tests/ -q`.  Must be
   regresssion-free (same pass count ± tolerant).
5. **Gate decision:**
   - **ACCEPT** — if the isolation test flipped + targeted mode improved +
     no regressions → commit and proceed.
   - **REVERT** — if the isolation test didn't flip or regressions appeared
     → `git checkout src/qr_reader/detector/hough.py`, revert the isolation
     test change, document the negative result, move to next phase.

   When documenting a revert, always:
   - State the objective evidence (isolation test result, fixture tallies, regressions).
   - Propose a hypothesis for why the fix didn't work (or caused regressions).
     Word it as a *hypothesis*, not a conclusion — use "may", "one possibility",
     "unconfirmed without dedicated tests", etc.
   - Suggest specific tests or measurements that would confirm or reject the
     hypothesis, so the next attempt at the same failure mode has a stronger
     evidence base.

Isolation tests live in `test_hough.py::TestRefineLineRealistic`.
Fixture tests live in `test_hough_harness.py::TestFixtureReal`.

---

## Pre-fix baseline (v12-default, seed=42)

```
Failure A (span too short):  2
Failure B (phantom):         5
Failure C (span too long):   4
Failure D (edge missing):    2
Total failures:             13  (6 GT edges in ROIs, 4 matched → 67%)
```

---

## Phase 1 — Rho-bin smoothing (targets D, partial A)

### Rationale

Failure D is caused by vote dilution: fragmented true-edge votes spread
across 3–5 rho bins, strong parallel edges in the *same theta bin* dominate,
and the relative threshold (`0.25 * max`) filters the diluted peak.
A 1-D Gaussian kernel (σ ≈ 1–2 bins) along the rho axis merges
fragmented votes into a single peak without changing the threshold semantics.

### Change

In `hough_vote_peaks`, after `acc = acc_flat.reshape(...)`, apply:

```python
from scipy.ndimage import uniform_filter1d  # or convolve1d for Gaussian
# 1-D Gaussian along rho axis (axis=1) with σ=1.5 bins
sigma = 1.5
from scipy.ndimage import gaussian_filter1d
acc = gaussian_filter1d(acc, sigma=sigma, axis=1, mode="nearest")
```

**No other changes.**  All existing parameters and NMS logic unchanged.

### Isolation test flip

`test_isolation_D_hough_quantization_misses_peak` — currently asserts
`not found` (bug confirmed).  After the fix:

```python
# OLD: assert not found, "... BUG CONFIRMED ..."
# NEW: assert found, (
#     f"FIX WORKS: peak matches fragmented true edge within 5°+5px "
#     f"(closest={best_info[0]:.1f}°/{best_info[1]:.1f}px)"
# )
```

### Expected fixture impact

- **D** (edge missing): ≥1 fewer failure (target: 2 → 0 or 1)
- **A** (span too short): possible marginal improvement from stronger peaks
- B, C: no change expected

### Accept if

- Isolation test flipped (now asserts `found`)
- ≥1 D failure eliminated in fixture tests
- Zero regressions in full suite (≥715 passes)

### Revert if

- Isolation test still asserts `not found` (D not fixed)
- New fixture failures appear
- Full-suite pass count drops

### Risk

- **Over-smoothing** — σ too large may merge distinct parallel lines 3–5 px
  apart into one peak.  Mitigation: keep σ ≤ 2.0, test on v5-default
  (has parallel finder+internal edges).
- **scipy dependency** — `gaussian_filter1d` requires scipy (already a
  dependency per module map).  If avoiding scipy, use a 3-tap boxcar via
  `np.convolve`.

---

## Phase 2 — Adaptive gap tolerance (targets A)

### Rationale

Failure A occurs when `gap_tolerance=2.0` can't bridge 4–7 px NMS gaps.
The fixed tolerance is too small for real finder boundaries after the
augmentation pipeline (blur, feather, perspective).  Scaling `gap_tolerance`
by the local median inter-pixel spacing in the support set adapts to the
edge's actual fragmentation level.

### Change

In `refine_line`, after collecting `support_pts` and computing the sorted
projection, compute the median gap between consecutive projections:

```python
# After proj_sorted is computed
if len(proj_sorted) > 1:
    gaps = np.diff(proj_sorted)
    median_gap = float(np.median(gaps))
else:
    median_gap = 1.0

# Adaptive tolerance: base_tolerance * median_gap, clamped to [2, 10]
adaptive_tol = max(2.0, min(10.0, gap_tolerance * median_gap))
```

Then use `adaptive_tol` in the contiguous-run loop instead of `gap_tolerance`.

### Isolation test flip

`test_isolation_A_gap_tolerance_insufficient` — currently asserts
`span < 20.0` (bug confirmed).  After the fix:

```python
# OLD: assert span < 20.0, "BUG CONFIRMED ..."
# NEW: assert span >= 20.0, (
#     f"FIX WORKS: adaptive gap_tolerance bridges 4+ px gaps → span={span:.1f}"
# )
```

### Expected fixture impact

- **A** (span too short): ≥1 fewer failure (target: 2 → 0 or 1)
- D: unchanged (Phase 1 already addressed the peak-detection side)
- B, C: no change expected

### Accept if

- Isolation test flipped (now asserts `span >= 20.0`)
- ≥1 A failure eliminated
- Zero regressions

### Revert if

- Isolation test still asserts `span < 20.0`
- Regressions appear (C failures increase — adaptive tolerance may worsen
  bleed-through by bridging gaps to parallel edges)
- Full-suite pass count drops

### Risk

- **C worsening** — higher tolerance may bridge across to parallel edges,
  increasing C failures.  The accept/revert gate catches this.  If this
  happens, defer to Phase 4 (angle-gated collection) before re-applying.

---

## Phase 3 — Minimum contiguous-run gate (targets B)

### Rationale

Failure B occurs when coincidentally-aligned sparse pixels produce a
non-degenerate segment.  `refine_line` has no quality floor.  A minimum
contiguous-run length requirement prevents phantoms from returning
meaningful segments.

### Change

In `refine_line`, after the contiguous-run loop finds `best_a, best_b`:

```python
# After the contiguous-run loop, before converting to endpoints.
# Require the longest contiguous run to contain ≥ min_contiguous pixels.
min_contiguous = 5

if best_len < 1e-6:
    return LineSegment(
        normal=refined_normal,
        rho=refined_rho,
        endpoints=np.zeros((2, 2), dtype=np.float64),
        vote_score=vote_score,
    )

# Count pixels in the best run (projection of support onto the line)
run_mask = (proj_sorted >= (best_a - 1e-6)) & (proj_sorted <= (best_b + 1e-6))
n_run_pixels = int(np.sum(run_mask))

if n_run_pixels < min_contiguous:
    return LineSegment(
        normal=refined_normal,
        rho=refined_rho,
        endpoints=np.zeros((2, 2), dtype=np.float64),
        vote_score=vote_score,
    )
```

### Isolation test flip

`test_isolation_B_sparse_noise_creates_phantom` — currently asserts
`span > 20.0` (bug confirmed).  After the fix:

```python
# OLD: assert not degenerate and span > 20.0, "BUG CONFIRMED ..."
# NEW: assert degenerate or span <= 20.0, (
#     f"FIX WORKS: phantom suppressed — span={span:.1f}px, degenerate={degenerate}"
# )
```

### Expected fixture impact

- **B** (phantom): ≥1 fewer failure (target: 5 → ≤3)
- A, C, D: no change expected

### Accept if

- Isolation test flipped (phantom segment is degenerate or short)
- ≥1 B failure eliminated
- Zero regressions

### Revert if

- Isolation test still produces non-degenerate phantom
- Real finder edges start becoming degenerate (A/D failures increase)
- Full-suite pass count drops

### Risk

- **False-negative on real edges** — if a real finder edge has fewer than 5
  contiguous pixels (e.g. small ROI, version ≤3), the gate incorrectly
  rejects it.  Mitigation: keep `min_contiguous` low (3–5).  If this
  triggers, consider scaling the gate by ROI size.

---

## Phase 4 — Angle-gated support collection (targets C, partial B)

### Rationale

Failure C occurs when weighted-TLS refinement drifts the line normal ~1°
from the Hough peak, causing the support-set distance gate to capture
pixels from a parallel edge 3–5 px away.  `refine_line` currently ignores
the `angle` array entirely.  Filtering support pixels by gradient-angle
consistency with the Hough peak normal prevents parallel-edge bleed-through.

### Change

In `refine_line`, add an angle-consistency filter when collecting support:

```python
# After computing dists from the Hough normal:
dists = np.abs(points @ normal - rho)
mask = dists < distance_thresh

# Angle-gate: only keep pixels whose normal angle is within ±tol of
# the Hough peak normal (modulo π, since lines are undirected).
if angle_gate_deg is not None:
    hough_theta = np.arctan2(normal[1], normal[0])
    edge_thetas = np.fmod(np.abs(angle[ys, xs]), np.pi)
    theta_diff = np.abs(edge_thetas - (hough_theta % np.pi))
    theta_diff = np.minimum(theta_diff, np.pi - theta_diff)
    mask &= theta_diff < np.deg2rad(angle_gate_deg)

support_pts = points[mask]
support_strengths = strengths[mask]
```

Add a new parameter `angle_gate_deg: float | None = None` to `refine_line`.
Keep the default `None` (disabled) so all existing callers are unaffected
until we validate the parameter value.

For the isolation test, call with `angle_gate_deg=10.0`.

Once the gate is validated, set the default to `10.0`.

### Isolation test flip

`test_isolation_C_tls_drift_bridges_parallel_edges` — currently asserts
`span <= 30.0`, which *passes* because the bug is present (the span IS
≤ 30 because distance_thresh=2.0 doesn't capture the parallel edge at
distance 3 px).  Wait — let me re-read the test...

Actually, the test calls `refine_line` with `distance_thresh=2.0` and the
parallel edge is at rho=28 (3 px from rho=25).  The test asserts
`span <= 30.0` which currently passes because `2.0 < 3.0` so the parallel
edge isn't captured.  That test needs to be fixed first — it doesn't
actually reproduce C with the current parameters.

**Fix the isolation test first:** change `distance_thresh` from 2.0 to
something that DOES capture the parallel edge (e.g. 4.0), so the test
currently fails with `span > 30.0`.  Then the angle gate should bring it
back to `span <= 30.0`.

```python
# Updated isolation test:
# distance_thresh=4.0 → captures parallel edge → span ~60 px (BUG)
# Add angle_gate_deg=10.0 → filters parallel edge → span ~20 px (FIX)
```

### Expected fixture impact

- **C** (span too long): ≥1 fewer failure (target: 4 → ≤2)
- **B** (phantom): possible improvement (angle-consistency also rejects
  coincidental alignment)
- A, D: no change expected

### Accept if

- Updated isolation test reproduces C → angle gate brings span ≤ 30.0
- ≥1 C failure eliminated in fixtures
- Zero regressions

### Revert if

- Angle gate degrades A/D (real finder pixels have inconsistent angles)
- No C improvement
- Full-suite pass count drops

### Risk

- **Angle aliasing at diagonal edges** — gradient angles at 45° are on a
  theta-bin boundary (±2° = 1 bin), so a ±10° gate is fine.  But very weak
  edge pixels may have noisy angles.  Mitigation: keep angle_gate_deg
  generous (10–15°).
- **Changes `refine_line` signature** — the new parameter has a default
  of `None`, so all existing callers are source-compatible.  The test
  harness will pass the parameter explicitly.

---

## Execution order and dependencies

```
Phase 1 (rho-bin smoothing) — REVERTED
  │
  └─ Phase 2 (adaptive gap_tolerance)
          │
          ├─ ACCEPTED → Phase 3 (min-contiguous gate)
          │               │
          │               ├─ ACCEPTED → Phase 4 (angle-gated collection)
          │               └─ REVERTED → Phase 4
          │
          └─ REVERTED → Phase 3 (skip 2)
```

Phases are conceptually independent, but the recommended order
de-risks later phases:

- **Phase 1 first** because rho-bin smoothing merges fragmented votes,
  giving stronger peaks for Phases 2–4 to refine.  This is the
  highest-impact, lowest-risk change.
- **Phase 2 second** because adaptive gap tolerance depends on the
  support-set structure (which Phase 1 doesn't change), and bridging
  larger gaps is a prerequisite for C fixes (you don't want to trim
  endpoints to the wrong fragment).
- **Phase 3 after 1+2** because the minimum-contiguous gate should see
  the merged support sets from Phases 1–2; otherwise real edges might
  be falsely rejected.
- **Phase 4 last** because it changes the `refine_line` signature and
  requires the most careful validation; the other phases establish a
  stable baseline.

---

## Validation commands

```bash
# Before starting any phase
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough_harness.py -v
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough.py::TestRefineLineRealistic -v
.venv/bin/python -m pytest src/qr_reader/tests/ -q

# After each phase change
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough.py::TestRefineLineRealistic -v -x
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough_harness.py -v
.venv/bin/python -m pytest src/qr_reader/tests/ -q

# If a phase is reverted
git checkout src/qr_reader/detector/hough.py
git checkout src/qr_reader/tests/detector/test_hough.py
# Re-run to confirm baseline restored
.venv/bin/python -m pytest src/qr_reader/tests/ -q
```

---

## Target end state

After all accepted phases, v12-default should show:

```
Failure A:  0 or 1
Failure B:  ≤ 2
Failure C:  ≤ 2
Failure D:  0 or 1
Total:      ≤ 6  (from 13 → ≥50% reduction)
Match rate: ≥ 83% (5/6 GT edges matched, from 4/6 = 67%)
```

---

## Revert log

### Phase 1 — REVERTED (2026-06-30)

**Change:** `gaussian_filter1d(acc, sigma=1.5, axis=1, mode="nearest")` in `hough_vote_peaks`.

**Isolation test D:** ✅ Flipped to `assert found` — passed.

**Fixture impact:** ❌ D failures unchanged (still 2 in v12-default). No measurable improvement in any failure mode.

**Regressions:**
- `TestHoughVotePeaks::test_horizontal_edges` — σ=1.5 merges parallel lines 3 px apart (y=20,23,26) into one peak.
- `TestHoughVotePeaks::test_vertical_edges` — same cause for vertical lines.
- v5-default: 6→7 failures (new A failure on TL_top).

**Full suite:** 713 passed (down from 715).

**Hypothesis for why D failures persisted:** The real D failures in v12-default (TL_left, BL_left) may not be caused by votes spread across adjacent rho bins. One possibility is that the true edge's votes are concentrated in a single rho bin but the total weight is below `0.25 * max_score` — rho-bin smoothing cannot help when there are no adjacent fragmented votes to merge. However, this is unconfirmed without dedicated tests that measure the vote distribution per bin for each failing GT edge.

## Future work

- **Confirm or rule out vote-scarcity hypothesis** — Add a test that, for each D-failing GT edge in the fixture data, dumps the per-bin vote count for its theta bin to confirm whether votes are concentrated (hypothesis) or spread (dilution). Only then can the right fix be chosen (absolute threshold floor vs. a different smoothing kernel / sigma).
- **Explore σ < 1.0 or boxcar filter** — A weaker rho-axis smoothing (e.g. 3-tap uniform `[1,1,1]/3`) may avoid the parallel-line regression while still helping if vote dilution is present in other failure modes.

## Files affected

| File | Phases touching it |
|------|-------------------|
| `src/qr_reader/detector/hough.py` | All (production code) |
| `src/qr_reader/tests/detector/test_hough.py` | All (isolation test flips) |
| `src/qr_reader/tests/detector/test_hough_harness.py` | None (fixtures unchanged, just re-run) |
| `docs/plan-007-hough-phased-fixes.md` | This document |
| `docs/hough-failure-analysis.md` | Updated with fix results per phase |
