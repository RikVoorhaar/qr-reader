# Hough Failure-Mode Analysis Report

> **Stage:** Analysis only — no production changes. This report documents the
> four observed failure modes in `hough_vote_peaks` + `refine_line`
> (`src/qr_reader/detector/hough.py`), their root causes as confirmed by
> isolation tests and the diagnostic script, and proposed solution
> directions ranked by likely impact.

## Executive summary

| Mode | Symptom | Root cause (confirmed) | Impact | Fix difficulty |
|------|---------|----------------------|--------|---------------|
| **A** | Span too short | `gap_tolerance=2.0` can't bridge 3–7 px NMS gaps | High — fragmented finder boundaries produce short segments | Low |
| **B** | Phantom in blank | No minimum-support / contiguity gate in `refine_line` | Medium — spurious segments in regions with no real edge | Low |
| **C** | Span too long | TLS direction drift + `distance_thresh=1.5` captures parallel edges 3–5 px away | High — segments bleed into internal QR module edges | Medium |
| **D** | Edge missing | Vote dilution + relative threshold (`0.25 * max`) filters out fragmented true edges dominated by stronger parallel edges | High — real finder boundaries have no corresponding peak | Medium |

All four modes reproduce reliably in both the fixture tests
(`test_hough_harness.py::TestFixtureReal`) and the synthetic isolation tests
(`test_hough.py::TestRefineLineRealistic`).  Run
`python src/qr_reader/scripts/debug_hough_failures.py` for the full
diagnostic dump.

---

## Evidence base

- **Fixture tests** (`src/qr_reader/tests/detector/test_hough_harness.py`):
  version-12 default-difficulty QR (seed=42), version-12 clean, version-5
  default.  Five soft-assertions per ground-truth finder edge.
- **Isolation tests** (`src/qr_reader/tests/detector/test_hough.py`, class
  `TestRefineLineRealistic`): 6 minimal synthetic `(nms, angle)` pairs, one
  per failure mode plus a negative-result companion for D.
- **Diagnostic script** (`src/qr_reader/scripts/debug_hough_failures.py`):
  per-cluster accumulator reconstruction, vote-fragmentation analysis,
  peak-suppression simulation, and cross-cluster summary.

### Headline numbers (v12 default, seed=42)

```
  Failure A (Span too short):  2
  Failure B (Phantom in blank): 5
  Failure C (Span too long):   4
  Failure D (Edge missing):    2
  Total failures:              13

  Total GT edges in ROIs:     6
  Total matched (within 5°+5px): 4
  Match rate:                  67%
```

---

## Failure A — Span too short

### Symptom

`refine_line` returns a segment that covers only a fragment of the visible
finder boundary.  The segment span is < 80 % of the ground-truth span.

### Root cause (confirmed)

`gap_tolerance=2.0` (the default in `refine_line`) cannot bridge the 3–7 px
gaps that NMS produces on real finder boundaries.  Gradient leakage from
blur + noise causes intermittent dropouts in the thinned-edge output.  When
a gap exceeds `gap_tolerance`, the contiguous-run logic breaks the support
set into fragments and returns the longest one — which is typically a
fraction of the true span.

### Evidence

**Fixture (Cluster 1, `TL_top`):** GT span = 41.8 px; refined span = 33.1 px
(79 %).  The `_describe_support` dump shows a single 4.5 px gap in the
projection at `[1.2 → 5.7]`, splitting the 46 support pixels into two runs.

**Fixture (Cluster 2, `BL_bottom`):** GT span = 35.8 px; refined span =
26.7 px (75 %).  Six gaps ≥ 1.5 px including a 6.5 px gap.

**Isolation test** (`test_isolation_A_gap_tolerance_insufficient`): four
clusters of 9 px each with 4–7 px gaps.  With `gap_tolerance=2.0` the
longest run is ≤ 9 px; the test asserts `span < 20.0` and passes (bug
confirmed).

### Proposed solutions

1. **Adaptive `gap_tolerance`** — scale `gap_tolerance` by the local median
   inter-pixel spacing in the support set (e.g. `2.0 * median_gap`).
   Minimal change, directly targets the symptom.
2. **Rho-bin vote smoothing** — apply a 1-D Gaussian kernel (σ ≈ 1–2 bins)
   along the rho axis of the accumulator before peak extraction.  This
   merges fragmented votes into a single peak, indirectly improving span.
3. **Morphological closing on NMS** — before Hough voting, bridge small
   gaps in the thinned-edge map with a 3×3 closing.  Changes the input
   rather than the algorithm; may interact with B and C.

**Recommendation:** Option 1 is the lowest-risk first step.  If it proves
insufficient, combine with option 2.

---

## Failure B — Phantom in blank region

### Symptom

`refine_line` produces a non-degenerate segment from sparse coincidentally
aligned pixels in a region with no real continuous edge.  The segment has
non-trivial span (> 20 px) despite the support being sparse.

### Root cause (confirmed)

`refine_line` has no minimum-support count or contiguity gate.  Any peak
that passes `hough_vote_peaks`'s `threshold_rel` filter gets refined, and
the weighted-TLS fit + gap-bridging logic will produce a segment from as few
as 3 collinear pixels (with `gap_tolerance` large enough).  Sparse pixels
that coincidentally align into a single `(theta, rho)` bin can accumulate
enough vote weight to pass the relative threshold, especially when the true
finder edges are fragmented (their votes are diluted).

### Evidence

**Fixture (Cluster 3):** all 8 GT edges are outside the ROI (the cluster
doesn't contain a finder pattern), yet 5 Hough peaks are extracted — all
classified as phantoms with mean NMS strength 470–565 and support counts of
19–68 pixels.  These are strong internal QR module edges, not noise, but
they are phantoms *relative to the finder-boundary detection task*.

**Isolation test** (`test_isolation_B_sparse_noise_creates_phantom`): 15
sparse collinear pixels at strength 120, 3 px apart, produce a Hough peak
(score 1800) whose refined segment has span 53.4 px.  No continuous edge
exists.  The test asserts `span > 20.0` and passes (bug confirmed).

### Proposed solutions

1. **Minimum contiguous-run gate** — in `refine_line`, require the longest
   contiguous run to contain ≥ N pixels (e.g. 5) before returning a
   non-degenerate segment.  Return degenerate otherwise.
2. **Support-density gate** — require the support set (within
   `distance_thresh`) to have ≥ K pixels per unit span (e.g. ≥ 0.5 px⁻¹).
   Sparse phantoms fail this; real edges pass.
3. **Angle-consistency check** — verify that the support pixels' gradient
   angles (from the `angle` array) are within tolerance of the line's
   normal.  Coincidental alignment won't have consistent gradient
   directions.

**Recommendation:** Option 1 is simplest and directly addresses the
isolation test.  Option 3 is more principled (uses the `angle` array that
`refine_line` currently ignores) and would also help with C.

---

## Failure C — Span too long

### Symptom

`refine_line` returns a segment that extends past the ground-truth finder
boundary endpoints into nearby parallel QR module edges.  Refined endpoints
are > 5 px from GT endpoints.

### Root cause (confirmed)

The weighted-TLS fit refines the line normal ~1° away from the Hough peak
normal.  Combined with `distance_thresh=1.5` px, the support set can capture
pixels from a parallel edge 3–5 px away.  The contiguous-run logic then
bridges across the gap between the true edge and the parallel edge,
producing an over-long segment.

### Evidence

**Fixture (Cluster 0, `TR_top`):** GT span = 41.8 px; refined span = 51.0 px
(122 %).  Endpoints `(76.6, 53.2) → (31.5, 76.9)`; GT endpoints
`(30.8, 76.9) → (67.8, 57.5)`.  The `_describe_support` dump shows 64
support pixels with projection span 63.5 px (vs GT 41.8 px), including one
11.6 px gap — the segment bridges across to a parallel internal edge.

**Fixture (Cluster 0, `TR_right`):** GT span = 41.8 px; refined span =
49.6 px (119 %).  Four gaps ≥ 1.5 px, with the support spanning 67.1 px.

**Isolation test** (`test_isolation_C_tls_drift_bridges_parallel_edges`):
two horizontal edges at ρ=25 and ρ=28 (3 px apart).
`refine_line` with `distance_thresh=2.0` bridges across, producing span > 30 px.
The test asserts `span <= 30.0` and passes (bug confirmed).

### Proposed solutions

1. **Angle-gated support collection** — only include pixels whose `angle`
   value is within tolerance (e.g. 10°) of the Hough peak normal.  Parallel
   edges with the same normal would still pass, so this alone is
   insufficient.  Combine with:
2. **Reduce `distance_thresh`** — from 1.5 to 1.0 px.  This narrows the
   capture zone so parallel edges 3+ px away are excluded.  Risk: may
   exclude legitimate edge pixels that are 1–2 px off the Hough rho.
3. **Use Hough peak normal for support collection, TLS only for
   refinement** — collect support pixels using the original Hough normal
   (not the TLS-refined one), then fit TLS only to determine endpoints.
   This prevents TLS drift from widening the capture zone.
4. **Endpoint trimming** — after finding the longest run, trim endpoints
   to the convex hull of high-strength support pixels (above some
   percentile).  Parallel-edge pixels (typically weaker) get trimmed.

**Recommendation:** Option 3 is the most principled — it separates the
concerns of support collection (geometric, using the Hough normal) from
refinement (statistical, using TLS).  Option 1 is a complementary safeguard
that also helps B.

---

## Failure D — Edge missing

### Symptom

A visible finder boundary has no corresponding Hough peak within 5° + 5 px
of the ground-truth `(normal, rho)`.

### Root cause (confirmed — NOT what the handoff originally claimed)

> **Correction:** The original handoff hypothesised theta quantization as the
> root cause ("2° theta quantization pushes rho 10–15 px off").  This is
> **not supported by evidence**.  The isolation test
> `test_isolation_D_quantization_alone_is_not_root_cause` demonstrates that
> mid-bin theta (134.3°) only shifts rho by ~2.5 px — well within the 5 px
> gate.  The real root cause is vote dilution + relative-threshold
> filtering.

The true finder edge's NMS pixels are fragmented (3–7 px gaps from gradient
leakage).  Each fragment votes into a slightly different rho bin because
the edge pixels are at slightly different perpendicular positions (due to
blur, perspective, and NMS interpolation).  The true edge's votes are
spread across 3–5 rho bins, so no single bin accumulates a strong peak.

Meanwhile, a stronger parallel internal QR module edge (continuous, higher
strength) concentrates its votes into a single rho bin, producing a dominant
peak.  The relative threshold (`threshold_rel=0.25 * acc.max()`) adapts to
this dominant peak, filtering out the diluted true-edge peak entirely.

### Evidence

**Fixture (Cluster 1, `TL_left`):** GT at θ=145.7°, ρ=24.3.  Vote
fragmentation analysis shows:
- GT rho bin 24 has **ZERO votes** — the true edge's NMS pixels quantise
  into other rho bins.
- Strongest bin is 13 (ρ=13, score=5803) — a parallel internal edge 11 px
  away.
- Votes are spread across bins 9, 13, 14, 18, 19 — five bins, none at the
  GT position.
- Closest Hough peak P3 is at 0.3° / 11.3 px — angle matches, rho is 11 px
  off (it's the parallel edge, not the true edge).

**Fixture (Cluster 2, `BL_left`):** GT at θ=145.7°, ρ=22.5.  GT rho bin 22
has ZERO votes; strongest bin is 7 (score=7116, 71 % of band) — 15 px away.

**Peak-suppression simulation (Cluster 3):** 98 local maxima above the
absolute floor (5 % of max); only 5 survive the relative threshold (25 %
of max) + NMS.  93 peaks are filtered out.

**Isolation test** (`test_isolation_D_hough_quantization_misses_peak`): a
fragmented true edge at ρ=24 (4-px gaps, strength 150) + a strong continuous
parallel edge at ρ=13 (strength 200).  No peak matches the true edge
within 5° + 5 px.  The test asserts `not found` and passes (bug confirmed).

### Additional observation: GT-vs-NMS displacement

The GT rho bin having **zero** votes is itself noteworthy.  It means the NMS
edge pixels for the finder boundary are displaced from the
metadata-predicted position by 5–15 px.  This displacement likely comes
from the augmentation pipeline (`feather_sigma_range`, `blur_sigma_range`,
`jitter_fraction`) shifting edges from their geometric ideal.  A more
robust GT-edge computation (e.g. rendering the finder boundary and running
edge detection on the clean image to get the true NMS position) would
improve the matching gate — but the underlying vote-dilution problem
remains.

### Proposed solutions

1. **Rho-bin smoothing** — apply a 1-D Gaussian kernel (σ ≈ 1–2 bins) along
   the rho axis of the accumulator before peak extraction.  Fragmented
   votes spread across 3–5 bins merge into a single peak.  Low risk,
   directly targets dilution.
2. **Absolute threshold floor** — add an absolute minimum threshold (e.g.
   `max(threshold_rel * acc.max(), abs_floor)`) so fragmented edges aren't
   filtered out just because a stronger parallel edge exists.  Requires
   tuning `abs_floor`.
3. **Per-theta adaptive threshold** — instead of a single global
   `threshold_rel * acc.max()`, compute the threshold per theta bin (e.g.
   `threshold_rel * acc[theta_bin].max()`).  This prevents strong edges at
   one orientation from suppressing true edges at another orientation.
4. **Vote re-weighting by local contiguity** — weight each edge pixel's
   vote by the number of NMS neighbours in a 3×3 window.  Fragmented
   pixels (few neighbours) get down-weighted, but continuous edges get
   up-weighted.  This doesn't help the fragmented true edge directly, but
   it reduces competition from semi-continuous internal edges.

**Recommendation:** Option 1 (rho-bin smoothing) is the highest-impact,
lowest-risk fix.  It directly addresses vote dilution without changing the
threshold semantics.  Option 3 (per-theta adaptive threshold) is a strong
complement — it addresses the parallel-competition aspect.

---

## Inter-mode dependencies

The four failure modes are not independent:

```
  Fragmentation ──┬──→ A (span too short)
                  ├──→ D (edge missing, when parallel competitor exists)
                  └──→ B (phantom, when fragmented true edge can't out-compete noise)

  TLS drift ──────┬──→ C (span too long)
                  └──→ B (phantom, when TLS drags toward strong nearby edge)
```

- **Fixing A (adaptive gap_tolerance) will partially mitigate D** — if the
  gap-bridging logic can handle larger gaps, the support set for a
  fragmented edge becomes more contiguous, producing a stronger peak in
  the accumulator.
- **Fixing B (minimum-support gate) will partially mitigate C** — a
  minimum contiguous-run requirement prevents the TLS-drifted support from
  extending too far.
- **Rho-bin smoothing (proposed for D) will also help A** — by merging
  fragmented votes into a single peak, the refined line will have a
  stronger, more concentrated support set.

## Recommended fix order

1. **Rho-bin smoothing** (addresses D + partially A) — highest impact, low
   risk.  Implement in `hough_vote_peaks` after accumulator construction.
2. **Adaptive `gap_tolerance`** (addresses A) — low risk, localised to
   `refine_line`.
3. **Minimum contiguous-run gate** (addresses B) — low risk, localised to
   `refine_line`.
4. **Angle-gated support collection / Hough-normal-based collection**
   (addresses C) — medium risk, changes `refine_line` interface semantics.

Each fix has a corresponding isolation test that will fail (signalling the
fix worked) when the bug is resolved.  The fixture tests provide
end-to-end validation.

## Files

| File | Role |
|------|------|
| `src/qr_reader/detector/hough.py` | Production code — **unchanged** |
| `src/qr_reader/tests/detector/test_hough.py` | Isolation tests (6, all confirm bugs) |
| `src/qr_reader/tests/detector/test_hough_harness.py` | Fixture tests + `_describe_support` instrumentation |
| `src/qr_reader/scripts/debug_hough_failures.py` | Diagnostic script with accumulator reconstruction, vote-fragmentation analysis, peak-suppression simulation |

## How to reproduce

```bash
# Isolation tests (all 6 pass = bugs confirmed)
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough.py::TestRefineLineRealistic -v

# Fixture tests (v12-default and v5-default fail = bugs confirmed;
# v12-clean passes)
.venv/bin/python -m pytest src/qr_reader/tests/detector/test_hough_harness.py -v

# Full diagnostic dump
.venv/bin/python src/qr_reader/scripts/debug_hough_failures.py
```
