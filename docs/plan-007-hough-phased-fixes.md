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

## Phase 2 — Adaptive gap tolerance (targets A)

### Result: REVERTED

Date: 2026-06-30

**Change:** In `refine_line`, replaced fixed `gap_tolerance` with
`adaptive_tol = max(2.0, min(10.0, gap_tolerance * max_gap))` where
`max_gap` is the maximum gap between consecutive support-point projections.
Initial plan used `median_gap` but that was dominated by within-cluster 1 px
gaps, so it never increased tolerance. Switched to `max_gap` to detect
structural between-cluster gaps.

**Isolation test:** `test_isolation_A_gap_tolerance_insufficient` flipped
from `assert span < 20.0` → `assert span >= 20.0`. **Passed.** (Side-effect:
`test_isolation_D2_few_pixels_become_degenerate` also flipped from
`assert degenerate` → `assert not degenerate` — adaptive tolerance bridges
4 px gaps between 3 sparse pixels.)

**Fixture impact:** ✅ **A failures eliminated** (v12-default: 2→0, all
span-too-short cases fixed). v12-default total dropped from 13 to 10.

**Regressions:**
- `test_fixture_version12_clean`: **0→4 C failures** (new C failures on
  BL_bottom, TL_top, TL_left, TR_right). The adaptive tolerance bridged
  7–9 px gaps separating finder edges from QR-internal edges in clean
  axis-aligned images, causing span-excessive failures.
- v5-default: 6→5 (marginal A improvement on BL_left, but still failing).

**Full suite:** fixture regression alone violates gate's zero-regression
requirement.

**Hypothesis for C regressions:** `max_gap`-based scaling is too aggressive
for clean images — a single large NMS gap (7–9 px) between the finder edge
and an adjacent internal edge inflates `adaptive_tol` to ~10 px, merging
distinct structures. A better approach may use a more robust statistic
(e.g. 75th percentile of gaps, or gap count-weighted tolerance) to
distinguish structural fragmentation (many moderate gaps) from a single
large gap crossing into a different edge.

## Phase 3 — Minimum contiguous-run gate (targets B)

### Result: REVERTED

Date: 2026-06-30

**Change:** In `refine_line`, after the contiguous-run loop, added an
average-gap density check: if `best_len / (n_run_pixels - 1) > 2.0`, return
a degenerate segment (no phantom).  The initial plan specified a simple pixel
count (`min_contiguous = 5`), but that was insufficient because with
`gap_tolerance=5.0` in the test, all 15 noise pixels bridge into one run
(15 > 5).  Switched to the average-gap density gate, which correctly rejects
the isolation test's phantom (avg gap ≈ 3.8 px).

**Isolation test:** `test_isolation_B_sparse_noise_creates_phantom` flipped
from `assert not degenerate and span > 20.0` → `assert degenerate or span <= 20.0`.
**Passed** — the density gate correctly rejects the 15 sparse collinear pixels
(avg gap ≈ 3.8 px > 2.0 px).

**Fixture impact:** ❌ **0 B failures eliminated** (v12-default: still 5 B
failures).  No other failure mode changed.

**Regressions:** None — v12-clean still 0, v5-default still 6.

**Full suite:** 715 passed (same as baseline).

**Hypothesis for why B failures persisted:** The real "phantom" peaks in
v12-default (cluster 3) are not sparse coincidental noise — they have dense
support (19–68 pixels, most gaps ≤ 2.0 px).  They are real QR-internal edges
(module boundaries, timing patterns, alignment-pattern edges) whose normals are
>12° from finder-boundary normals, so the angular skip in `_assert_no_phantom`
does not catch them.  No density-based gate in `refine_line` can distinguish
these from genuine finder edges — they are structurally indistinguishable at
the pixel level.  Eliminating them likely requires a higher-level consistency
check (e.g. spatial proximity to the expected finder-pattern layout, or
requiring that each candidate edge participates in a valid finder-pattern
triplet geometry).

## D investigation — Vote-scarcity hypothesis DISCONFIRMED (2026-06-30)

Running `debug_hough_failures.py` (baseline, no fixes applied):

| D Edge | GT rho bin score | Vote spread | Strongest competitor | Gap from GT |
|--------|-----------------|------------|---------------------|-------------|
| C1 TL_left (ρ=24.3) | **0** | bins 9,13,14,18,19 | bin 13 @ 5803 (47%) | 11 px |
| C2 BL_left (ρ=22.5) | **0** | bins 7,11,12,16 | bin 7 @ 7116 (71%) | 15 px |

**Finding:** The vote-scarcity hypothesis is **wrong**. Both D edges have
**zero votes at the GT rho bin**.  Votes exist at nearby bins (9–19 and
7–16) but are spread across 4–5 bins with a stronger parallel edge
dominating.  The root cause is **GT-vs-NMS displacement** (11–15 px shift
from blur/noise/jitter in the augmentation pipeline) **plus vote dilution**
(the few votes that DO exist at the correct angle are spread across 4–5
rho bins).

**Implication:** Rho-bin smoothing (Phase 1) and per-theta adaptive
threshold are both non-starters for these D failures:
- Smoothing can't create votes at a zero-vote bin
- A per-theta threshold of `0.25 × 7116 = 1779` still exceeds the true
  edge's per-bin scores (540–1201)

What *might* help: an absolute threshold floor low enough to surface these
weak clusters (~1200).  But we don't know yet whether the displaced votes
(presumably from the finder boundary) produce valid refined segments, or
whether they're from unrelated internal QR edges.

---

## Part 1: Information-gathering phases (run first)

These phases require **test changes only** — no production code modified.
They confirm or rule out hypotheses before we commit to fix designs.

### I1 — Verify D displacement: widen rho tolerance

**Rationale:** The two D failures have votes at the correct angle but
displaced 11–15 px from the GT rho.  Are these votes actually from the
finder boundary (just shifted by augmentation), or from unrelated
internal QR edges?

**Change:** In the fixture harness (`test_hough_harness.py`), widen the
rho tolerance in `_match_peak` from 5 px to 20 px for D-failing edges
only.  Then:
- Check if the widened match now finds a peak for TL_left and BL_left.
- If yes, run `refine_line` on the matched peak and inspect the segment
  (span, endpoints, support dump).
- If the segment is close to the GT span (±20%) and endpoints are near
  GT, then **production-code fix is warranted** (the displaced votes are
  real finder pixels).
- If the segment is wildly wrong or `refine_line` produces garbage, then
  the displaced votes are **not** from the finder boundary and the D
  failures are a test artifact (GT matching needs widening, not Hough).

**Success criteria:** For each D edge, we can answer:
- Which Hough peak does a widened match capture?
- What does `refine_line` produce from it?  Is the segment valid
  (span ≥ 80% GT, endpoints ±5 px)?
- Conclusion: displace votes are / are not finder-boundary pixels.

**No production code changes.** This is a diagnostic-only phase.

---

### I2 — Identify B phantom sources

**Rationale:** The 5 B phantoms in Cluster 3 are dense, real QR edges
whose normals are >12° from finder normals.  We need to know *which*
QR structure creates each one, to design a spatial consistency gate.

**Change:** For each of the 5 phantoms in Cluster 3 (v12-default fixture):
1. Compute the refined segment (`refine_line` + `_describe_support`).
2. Look up the phantom's (θ, ρ) against known QR geometry:
   - Finder boundaries (already checked, angles differ by >12°)
   - Timing-pattern rows/columns
   - Alignment-pattern edges
   - Format-information / version-information module rows
   - Data-module block boundaries
3. Overlay the segment on the NMS image to visually verify.
4. Dump the spatial position of the segment relative to the QR module grid
   (if metadata gives us the model-view transform).

**Success criteria:** For each of the 5 phantoms, we can say "this is the
timing pattern row at col 6" or "this is the alignment-pattern edge at
(row 30, col 30)".  This tells us whether a spatial-proximity gate (e.g.
"reject edges > X px from the expected finder-pattern layout") would
eliminate them.

**Stretch goal:** Determine if any phantom corresponds to a *real
finder boundary* that happens to be at a >12° angle due to perspective
warp — in which case the B classification is a test artifact (the edge
is real, the test just has an overly strict angular match).

---

### I3 — Profile A gap causes

**Rationale:** The A-failing edges (TL_top in C1, BL_bottom in C2) have
4.5–6.5 px NMS gaps.  Phase 2 (adaptive gap tolerance) fixed these but
caused C regressions by bridging 7–9 px gaps to internal edges.  We need
to understand *what* causes the gaps: are they genuine noise dropouts
(fixable by morphological closing) or QR-internal structure aligned at a
different angle (needs wider gap tolerance + angle gating)?

**Change:** For each A-failing edge, inspect the NMS content in the gap
region:
1. Map the gap's projection interval to pixel coordinates on the refined
   line.
2. Check whether any NMS pixels exist within ±2 px of those coordinates
   but at a different angle (e.g. module-edge boundary crossing the
   finder line at 90°).
3. If NMS pixels exist but are suppressed by angle gating → the gap is
   structural (internal QR edge crossing) → wider gap tolerance is safe.
4. If gap region has no NMS pixels at all → the gap is a noise dropout
   in the edge-detection pipeline → morphological closing is the right fix.

**Success criteria:** For each A failure, classify the largest gap as:
- **Structural** (NMS pixels exist at wrong angle) → gap tolerance fix
- **Dropout** (no NMS pixels, edge genuinely broken) → morphological fix
- **Mixed** (some gaps structural, some dropout) → combined approach

---

## Part 2: Fix phases (with revert gates)

### Phase 4 — Angle-gated support collection (targets C)

*Carried over from the original plan.  Not yet attempted.*

**Status:** The isolation test needs repair first — with current
parameters (`distance_thresh=2.0`), the test doesn't actually reproduce
C.  Fix: increase `distance_thresh` to 4.0 so the parallel edge IS
captured, then angle-gate it away.

### Phase I1-result-dependent D fixes

The fix for D depends on I1's conclusion:

- **If I1 shows displaced votes ARE finder pixels:**
  - F5a: **Absolute threshold floor** — `max(rel * acc_max, floor)` where
    `floor` is tuned (start at 1000) to surface the weak displaced clusters.
  - F5b: **Per-theta threshold with floor** — `threshold = max(rel * acc_max,
    rel * max_in_theta_band, floor)` — combines per-theta fairness with
    a floor for sparse theta bands.

- **If I1 shows displaced votes are NOT finder pixels:**
  - F5c: **Widen GT matching tolerance** — the Hough pipeline is fine;
    the 5 px rho tolerance in the fixture tests is too strict given known
    augmentation-induced displacement.  This is a test-only change.

**Gate for F5a:** Absolute floor surfaces ≥1 D-failing edge, zero
regressions in fixture tests, full suite ≥ 715 passes.

**Gate for F5c:** After widening rho tolerance in fixture tests,
≥1 D failure resolved, segments are valid (span ≥ 80% GT, endpoints
±5 px), zero regressions.

---

### Phase 6 — Multi-finder spatial consistency (targets B)

**Rationale:** Phase 3 confirmed that B phantoms are dense QR-internal
edges indistinguishable from finder edges at the pixel level.  A
higher-level check is needed: if a candidate edge is a real finder
boundary, it should be spatially close to the expected finder-pattern
region (e.g. within the finder-pattern's 7×7 module zone).

**Change:** After `refine_line` produces segments for all peaks, filter
out segments whose endpoints are >X px from any expected finder boundary
position (computed from the cluster's alignment-pattern positions and
estimated version).

*Exact change TBD — depends on I2 identifying what the phantoms are.
If they're near the finder pattern but misaligned, the fix might be a
geometric consistency check (e.g. "lines from a valid finder pattern
should form an L-shape with ~90° corner").*

**Isolation test:** Not applicable (B phantoms don't appear in the
synthetic isolation tests — they only appear in real QR fixtures).
Validation is via fixture test B tallies.

**Gate:** ≥1 B failure eliminated, zero A/C/D regressions, full suite
≥ 715 passes.

---

### Phase 7 — Morphological closing on NMS (targets A)

**Rationale:** If I3 shows that A-failing gaps are genuine noise dropouts
(no NMS pixels in the gap region), a morphological closing before Hough
voting would bridge them.  This targets A at the input stage rather than
in `refine_line`.

**Change:** After `extract_thin_edges` returns `(nms, angle)`, apply a
small closing kernel (e.g. 3×3 cross `[[0,1,0],[1,1,1],[0,1,0]]`) to
`nms` before passing to `hough_vote_peaks`.  This preserves the thinned
structure but bridges 1–2 px dropouts.

**If I3 shows structural gaps** (NMS pixels exist at wrong angle), skip
this phase — morphological closing won't help against crossing edges
and may worsen C by creating false connections to parallel edges.

**Isolation test:** New test needed — a synthetic NMS with 1–2 px gaps
that currently fragments, which becomes contiguous after closing.

**Gate:** ≥1 A failure eliminated, no extra C failures (closing may
worsen drift bleed-through), full suite ≥ 715 passes.

---

## Execution order

```
Information-gathering (run first, no production changes):
  I1 (D displacement)
  I2 (B phantom sources)
  I3 (A gap causes)

Fix phases (in dependency order):
  Phase 4 (angle-gated support) → targets C
  Phase 5a/5b/5c (D fix, depends on I1) → targets D
  Phase 6 (multi-finder consistency) → targets B
  Phase 7 (morphological closing) → targets A
```

## Files affected

| File | Phases touching it |
|------|-------------------|
| `src/qr_reader/detector/hough.py` | Phase 1–3 (reverted), 4–7 (production code) |
| `src/qr_reader/tests/detector/test_hough.py` | Phase 1–3 (reverted), 4, 5a, 7 (isolation test flips) |
| `src/qr_reader/tests/detector/test_hough_harness.py` | I1, I2, I3 (diagnostic changes), 6 (fixture validation) |
| `src/qr_reader/scripts/debug_hough_failures.py` | I2, I3 (may add spatial / gap diagnostics) |
| `docs/plan-007-hough-phased-fixes.md` | This document |
| `docs/hough-failure-analysis.md` | Updated with fix results per phase |
