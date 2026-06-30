# Plan — Debugging short Hough line segments

> Target: diagnose why `refine_line` produces segments that are too short when
> overlaid on finder-pattern ROIs in `scripts/full-pipeline-canny.py`.
> **Scope:** diagnostics and unit tests only — no changes to the existing
> edge extraction or detection pipeline.

## Background

`hough_vote_peaks` detects line candidates (unit normal, signed rho, accumulator
score).  `refine_line` then collects edge pixels near the approximate line,
fits a weighted TLS line, and extracts the longest contiguous support run to
define segment endpoints.

The lines rendered in the Hough overlay figure appear **shorter than the actual
finder-pattern edges they should track**.  The edge extraction (Sobel + NMS)
looks good in the diagnostic subplots, so the problem is likely in the Hough or
post-Hough interpretation.

## Root-cause hypotheses (ordered by likelihood)

| # | Hypothesis | Why plausible |
|---|-----------|---------------|
| H1 | **The diagnostic script only draws segment extents, not the Hough line itself.** We can't tell if the Hough peak correctly identifies the infinite line but the support collection pinches it. | The overlay plot draws `endpoints`, not the infinite line clipped to the ROI. |
| H2 | **Support collection only uses geometric distance, not gradient-angle agreement.** Edge pixels from crossing lines (e.g., QR module edges) pollute the support set, and the weighted TLS gets pulled off-axis, fragmenting the contiguous runs. | `refine_line` checks `abs(p @ normal - rho) < distance_thresh` but ignores whether the edge gradient agrees with the line normal. |
| H3 | **One-theta voting + 2° bins + no accumulator smoothing → one physical finder edge may be split across 2-3 adjacent accumulator bins.** The dominant peak may still be correct, but the support collection happens at a single quantised (normal, rho) that doesn't perfectly match all the edge pixels along that line. At `distance_thresh=1.5`, some pixels just miss. | No smoothing, no angular-window voting. The rho-smear over a ~40 px edge at 2° angular quantisation is ~1.4 px — enough to fragment support at threshold 1.5. |
| H4 | **`distance_thresh=1.5` and `gap_tolerance=2.0` are too tight for real NMS edge scatter.** Finder-pattern edges may have 1-2 px of real scatter from the NMS step, plus the quantisation smear from H3. Combined, the per-pixel support gaps exceed the tolerance. | Default values were chosen heuristically, not from real data. |
| H5 | **QR module edges produce accumulator peaks that compete with the true finder-pattern boundary lines.** Finding the correct 4 lines among many false positives requires more structure than just "top K peaks by score." | The ROI contains dense QR payload edges. Without a rho-gate or finder-specific prior, Hough will happily vote for those. |

## Investigation phases

Each phase includes a concrete diagnostic that can be committed as a unit test
or added to the diagnostic script.  No code changes to production modules until
a hypothesis is confirmed.

---

### Phase I — Distinguish Hough quality from support quality ✅ COMPLETED

**What we don't know:**  Is the Hough peak itself correctly identifying the
finder boundary line?  Or is the peak wrong and support is irrelevant?

**Test:** Add a helper `_draw_infinite_line(normal, rho, H, W)` that returns
two points where the line intersects the ROI boundary.  Modify
`full-pipeline-canny.py` to draw **two layers** in the Hough overlay:

1. **Infinite Hough lines** (dashed, thin, one colour per peak), clipped to ROI.
2. **Refined support segments** (solid, thick, same colour), as currently.

**Validation:** Visual inspection.  If the dashed infinite lines accurately
follow the finder-pattern edges but the solid segments are short, the Hough is
fine and the problem is in support collection (→ Phase II).  If the dashed
lines are also wrong, the problem is in Hough voting (→ Phase III).

**File changes:** `full-pipeline-canny.py` only — add the infinite-line overlay
without modifying `hough.py`.

**Acceptance:** The two-layered plot clearly shows whether Hough peaks match
the expected finder boundaries.

**Implementation notes (2026-06-30):**
- Added `_draw_infinite_line()` helper: computes line–ROI boundary intersections
  by testing x=0, x=W-1, y=0, y=H-1 and returning the first two valid intersection
  points.
- Figure 2 now draws **two layers per peak**:
  - Dashed thin line (α=0.35) for the infinite Hough line, labelled `H{i}`
  - Solid thick line (lw=4, α=0.9) for the refined support segment, labelled `S{i}`
- Segment thickness increased from 2 → 4 for visibility.

---

### Phase II — Diagnose support collection (if H1/H2/H4 confirmed)

**Hypotheses in play:** Support collection too strict, missing gradient-angle
agreement gate, distance/gap thresholds too tight.

**Tests to write (pytest, `tests/detector/test_hough.py`):**

1. **`test_support_on_noisy_synthetic_finder_edge`**
   - Generate a synthetic finder-pattern-like ROI: a 40×40 grayscale image with
     a dark square 10 px from the edge (simulating a ~20 px wide finder), add
     σ=1.0 noise, run `extract_thin_edges`, then `hough_vote_peaks`, then
     `refine_line` for the strongest peak.
   - **Assert:** The refined segment's endpoints span ≥ 80% of the true edge
     length, and the refined normal is within 5° of the true edge normal.
   - This is the **key integration test** — it exercises the whole
     NMS→Hough→refine chain on controlled but realistic data.

2. **`test_grid_crossing_edges_dont_pollute_support`**
   - Create a synthetic image with two crossing lines (e.g., a "+" shape).
     Vote for one line, then refine.
   - **Assert:** The refined line doesn't get pulled off-axis by the crossing
     line's edge pixels, even when `distance_thresh` is generous (3 px).  This
     test currently **will fail** (confirming H2) because we don't filter by
     gradient-angle agreement.

3. **`test_distance_thresh_loosening_increases_span`**
   - On a real or synthetic finder-edge ROI, call `refine_line` with
     `distance_thresh` at 1.0, 2.0, 3.0 px.
   - **Assert:** Span increases monotonically or plateaus (i.e., more pixels
     get included without pulling in noise from crossing lines).

**If H2 confirmed** (crossing edges pollute support):
  - Add an `angle_threshold_deg` parameter to `refine_line` (default ~15°).
  - Only include edge pixels where `angular_difference(edge_normal, line_normal) < angle_threshold`.
  - Update the support-collection mask.
  - Re-run test 2 to confirm crossing edges are excluded.

**If H4 confirmed** (thresholds too tight):
  - Increase `distance_thresh` default from 1.5 → 2.5 px.
  - Increase `gap_tolerance` default from 2.0 → 3.0 px.
  - Re-benchmark: visual inspection in the diagnostic script.

**Acceptance:** At least 3 new passing tests; the integration test passes with
default parameters on a synthetic finder edge.

---

### Phase III — Improve Hough voting quality (if H3/H5 confirmed)

**H3 (bin-fragmentation):**

1. **Test `test_accumulator_has_add_aliasing`**
   - Generate a synthetic straight edge spanning a known starting index to an
     ending index in the accumulator. Compute the accumulator.
   - **Assert:** The peak falls within ±1 bin of the ground-truth (theta, rho),
     and at least 80% of the edge's vote mass lands within ±1 theta bin and
     ±2 rho bins of the peak.

2. **Potential fixes (implement one at a time, re-test):**
   - **Fix A:** Apply a small Gaussian blur to the accumulator before peak
     extraction (sigma ≈ 0.5–1.0 bins).  Cheap, often sufficient.
   - **Fix B:** Switch from one-theta to small-window voting (2–3 bins on each
     side, Gaussian-weighted by angular agreement).  More robust, slightly more
     compute.
   - **Fix C:** Reduce `theta_step_deg` from 2.0 → 1.0° to reduce rho-smear from
     angular quantisation.

   Prefer Fix A (accumulator smoothing) because it's minimal and doesn't change
   the voting logic.  If insufficient, try Fix C (finer bins).  Reserve Fix B
   (windowed voting) for last.

**H5 (QR module clutter):**

1. **Test `test_rho_gating_suppresses_center_clutter`**
   - Create a synthetic 50×50 ROI with an outer finder square (edge at ~|ρ| ≈ s/2)
     and dense random edge clutter in the center (simulating QR payload modules).
   - Run `hough_vote_peaks` with and without a rho-gating parameter.
   - **Assert:** With the gate, the top 4 peaks by score all correspond to true
     outer finder edges.  Without it, at least one peak comes from the center
     clutter.

2. **Potential fix (if confirmed):**
   - Add an optional `rho_gate_center` parameter to `hough_vote_peaks`: a tuple
     `(cy, cx, half_size)` defining the expected finder center and half-size.
   - When provided, a vote is only cast if:
     `abs(abs(rho) - half_size) < rho_gate_fraction * half_size`
     where `rho_gate_fraction` defaults to ~0.25.
   - This requires centered coordinates internally — switch from pixel-origin
     to ROI-centered coordinates when the gate is active.  Return values remain
     in pixel coordinates.

   However, H5 is **less urgent**: the finder-pattern edges are the strongest
   and densest lines in the ROI.  If H1–H4 are addressed, the top-4 peaks may
   already be correct.  Defer H5 until Phase IV (corner extraction) unless the
   diagnostic script shows clear payload-line false positives beating the
   finder lines.

**Acceptance:**
  - The aliasing test passes with the chosen fix.
  - Visual inspection: infinite Hough lines align with finder boundaries, and
    payload-module lines don't dominate the top-4 peaks.

---

### Phase IV — Corner extraction (out of scope, deferred)

Once the refined segments correctly span the finder boundaries:

1. Select 4 lines forming a convex quadrilateral from the `LineSegment` list.
2. Compute pairwise intersections.
3. Validate that intersections fall within the support span of both lines.
4. Order corners consistently (TL, TR, BR, BL) and feed into the existing
   `FinderPattern` pipeline.

---

## Validation strategy

| Stage | What | How |
|-------|------|-----|
| Hough peaks vs. infinite lines | Visual: dashed infinite lines overlay | `full-pipeline-canny.py` Phase I figure |
| Support collection | Integration test on synthetic finder edge | `test_support_on_noisy_synthetic_finder_edge` |
| Gradient-angle gating | Crossing-lines test | `test_grid_crossing_edges_dont_pollute_support` |
| Accumulator aliasing | Bin-fragmentation test | `test_accumulator_has_no_aliasing` |
| Real-ROI end-to-end | Visual: refined segments match finder edges | `full-pipeline-canny.py` final overlay |
| Regression | Full test suite passes | `pytest` |

**Success criterion for this plan:** On a version-12 synthetic QR image in
`full-pipeline-canny.py`, at least one cluster shows 4 refined line segments
that span ≥ 80% of the visible finder-pattern edges, visually verified.

## Non-goals / deferred

- Corner extraction (Phase IV, next plan)
- RHO-gating for payload-module suppression (deferred unless H1–H4 fixes aren't enough)
- Switching to centered coordinates (pixel coords are fine; centered coords only
  needed if we add rho-gating, and even then just internally)
- Changing the `cluster_to_bbox` scale or ROI extraction logic
