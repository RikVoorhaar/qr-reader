# Plan — Hough + Refine Test Harness

> Target: Build a robust test suite for `hough_vote_peaks` + `refine_line` that
> reproduces the observed failure modes with ground truth, then guides algorithm
> fixes.
>
> **Scope:** test harness and diagnostic instrumentation only — no changes to
> production `hough.py` until failure modes are isolated and understood.

## Background

`extract_thin_edges` + `hough_vote_peaks` + `refine_line` produces
`LineSegment`s overlaid on finder-pattern ROIs.  Visual inspection in
`full-pipeline-canny.py` reveals four distinct failure modes on version-12 QR
images:

| Label | Failure | Observable symptom |
|-------|---------|-------------------|
| **A** | Span too short | Segment covers only a fragment of the visible finder boundary; 1–2 px NMS gaps break bridging |
| **B** | Phantom in blank region | Short segment in a white area with no visible finder edge; weak coincidentally-aligned pixels out-compete fragmented true segments |
| **C** | Span too long | Segment bleeds past the finder boundary into nearby parallel QR module edges |
| **D** | Edge missing | A visible finder boundary has no corresponding segment at all |

The current `test_hough.py` tests use clean synthetic edges only and do not
exercise any of these failure modes.

Ground truth is available: the synth augmentation pipeline
(`generate_sample` → metadata[`corners_qr`] + `N` + `pixels_per_module`)
produces composited QR images with known corner positions in image space, from
which finder-pattern boundary lines can be computed exactly.

## Phases

### Phase I — Ground-truth geometry helper

**Goal:** A utility that, given the metadata from `generate_sample`, computes
the **finder-pattern boundary edges** in image space — as infinite lines
(normal, rho) and as segment endpoints clipped to any given ROI.

**What it does:**

1. From `corners_qr` (TL, TR, BR, BL) and `N` (module count), compute the
   four outer edges of each finder pattern:
   - **TL top**: from TL toward TR, `7/N` fraction of the full edge
   - **TL left**: from TL toward BL, `7/N` fraction
   - **TR top**: from TR toward TL, `7/N` fraction (same edge as TL top)
   - **TR right**: from TR toward BR, `7/N` fraction
   - **BL left**: from BL toward TL, `7/N` fraction
   - **BL bottom**: from BL toward BR, `7/N` fraction
   - **BR right**: from BR toward TR, `7/N` fraction
   - **BR bottom**: from BR toward BL, `7/N` fraction

2. Each edge is represented as:
   - An **infinite line** (unit normal `n`, signed rho `ρ`) in image pixel
     coordinates.
   - A **ground-truth segment** — the two endpoints in image coordinates
     defining where the finder boundary physically exists.

3. Given an ROI bounding box (defined by its top-left offset in the full
   image, plus its height and width), clip each ground-truth segment to the
   ROI and convert (n, ρ) to ROI-local pixel coordinates.

**Input:**
```
metadata: dict       # from generate_sample output
roi_offset: (row0, col0)   # top-left of ROI in full-image coords
roi_shape: (H, W)    # dimensions of ROI
```

**Output:**
```
List of:
  - label: str           # "TL_top", "TR_right", etc.
  - normal_roi: (2,)     # unit normal in ROI-local (x=col, y=row)
  - rho_roi: float       # signed distance in ROI-local coords, ≥ 0
  - segment_roi: (2, 2)  # endpoints in ROI-local (x,y), or None if not intersecting
```

**File:** `src/qr_reader/tests/detector/test_hough_harness.py` (new file, this
phase only — no production code)

**Acceptance:** Unit tests on a clean axis-aligned QR (version=1, no
transform, `box_size=10, border=4`) verify:
- TL top edge normal is `(0, 1)` (horizontal edge, gradient vertical)
- TL top edge rho is `border * box_size = 40`
- Segment span in ROI is within 1 px of expected

---

### Phase II — Fixture-based reproduction tests

**Goal:** Tests that exercise the full `extract_thin_edges` →
`hough_vote_peaks` → `refine_line` pipeline on **real** synth pipeline output
and check against ground truth.  These tests *will fail* on first run —
they're documentation of the current bugs.

**How each test works:**

1. `config = AugmentationConfig(version=12, seed=S, difficulty_preset=...)` —
   use a difficulty preset that exercises realistic edge conditions (moderate
   perspective, mild noise/blur).  Do NOT use "easy" — the bugs only manifest
   on real data.

2. `image, metadata = generate_sample(rng, config, background)`

3. Convert to grayscale, binarize, find clusters, extract ROIs (same pipeline
   as `full-pipeline-canny.py` up to `cutout`)

4. For each cluster ROIs, run `extract_thin_edges` → `hough_vote_peaks`

5. Compute ground-truth finder edges via Phase I helper, clipped to the ROI

6. Run `refine_line` on every Hough peak (with default params from `hough.py`)

7. Assertions (see below)

**Assertions per ground-truth edge that intersects the ROI:**

| # | What | Assertion | Failure mode |
|---|------|-----------|-------------|
| 1 | Peak exists | At least one Hough peak has (normal, rho) within 5° and 5 px of the ground-truth edge | D |
| 2 | Segment span adequate | Refined segment span ≥ 80% of ground-truth span within the ROI | A |
| 3 | Segment span not excessive | Refined segment endpoints are within 5 px of ground-truth endpoints | C |
| 4 | No phantom | For peaks that don't match any ground-truth edge, the segment's mean NMS strength is below a threshold (no strong-lines-in-blank-regions) | B |
| 5 | Degeneracy | At least one peak per ground-truth edge produces non-zero endpoints | D |

**Assertion granularity:** Each assertion should be its own `assert` statement
or `pytest_check` so one failure doesn't hide others.  Use soft asserts
(`pytest_check`) from `pytest_check` plugin (or manual `failures: list[str]`
collector if the plugin isn't installed).

**Test cases (at least 3):**

| Test name | Config | Why |
|-----------|--------|-----|
| `test_fixture_version12_default` | version=12, default difficulty (moderate perspective, noise) | The failing case from the diagnostic script |
| `test_fixture_version12_clean` | version=12, noise=0, blur=0, no perspective | Baseline — should pass easily |
| `test_fixture_version5_default` | version=5, default difficulty | Shows whether failure modes are version-dependent |

**Expected result on first run:** Most assertions fail for version-12.  The
failures are recorded with `pytest.skip` / `pytest.xfail` or soft-assert
messages so the suite still terminates cleanly without blocking CI.

**File:** `src/qr_reader/tests/detector/test_hough_harness.py` (same file,
add this phase)

**Acceptance:** Test file exists, runs without crashing, produces failure
messages that clearly identify which assertion failed on which ground-truth
edge in which cluster.  Failure messages include ROI-local coordinates so they
can be cross-referenced with visual output from `full-pipeline-canny.py`.

---

### Phase III — Synthetic isolation tests

**Goal:** Once Phase II confirms the failure modes on real data, build
**minimal synthetic** (nms, angle) pairs that isolate each failure mode
individually.  These tests are fast, deterministic, and don't require the full
synth pipeline or background images.

**For each failure mode, one test:**

| Test | Synthetic setup | Assertion |
|------|----------------|-----------|
| `test_isolation_A_span_short` | Single horizontal edge with 1-px NMS gaps at x=10, 20, 30, 40 over a known visible span [5, 45].  Edge strength: 200.  No noise. | Segment span ≥ 36 px (80% of 45 px) |
| `test_isolation_B_no_phantom` | Strong fragmented edge A (strength 200) spanning x∈[5,45] with 3-px gaps + weak continuous edge B (strength 20) spanning x∈[50,60] at same (normal, rho).  Both pass the Hough distance gate. | The peak assigned to the strongest support cluster produces segment endpoints that fall on pixels with mean NMS strength > 100 (i.e., on edge A, not B) |
| `test_isolation_C_parallel_pollution` | Edge A at rho=25 spanning x∈[10,30], edge B at rho=28 spanning x∈[50,70], both horizontal (same normal).  Run `refine_line` at rho=25. | Segment span ≤ 25 px (doesn't bridge across to edge B) |
| `test_isolation_D_edge_missing` | A single edge with 3 strong pixels (not 2) at known (normal, rho) | `refine_line` returns non-zero endpoints (not degenerate) |

**Design principle:** Each test should be ≤ 30 lines of setup and one
assertion.  No external dependencies beyond `numpy`.  No QR generation, no
synth pipeline.

**File:** `src/qr_reader/tests/detector/test_hough.py` (add to the existing
file, in a new class `TestRefineLineRealistic` or similar)

**Acceptance:** All four tests exist, compile, and fail in a way that
confirms the observed behavior matches the synthetic edge case.

---

### Phase IV — Instrumentation and diagnostics

**Goal:** When a Phase II test fails, the failure message includes enough
context to diagnose without re-running the visual script.

**What to add:**

- A diagnostic dump function `_describe_support(seg, nms, angle)` that, for a
  given `LineSegment`, prints:
  - Total support pixels within `distance_thresh`
  - Projection range (min, max)
  - List of gaps ≥ 1.5 px in the sorted projection
  - The 5 strongest and 5 weakest support pixels with their (x, y) positions
  - Whether any support pixels fall in a high-density region of `nms` vs.
    isolated peaks

- Call this from `pytest_check` helpers so the output appears in the test
  failure log.

**File:** `src/qr_reader/tests/detector/test_hough_harness.py`

**Acceptance:** A failing assertion in a Phase II test produces enough
diagnostic output to identify *why* the gap-bridging failed (e.g., "support
set has 14 pixels, projection span 45 px, 6 gaps ≥ 1.5 px, max gap 18.3 px
between projection 22.1→40.4").

---

## Non-goals / deferred

- Changing default parameters in `hough.py` (`distance_thresh`, `gap_tolerance`,
  `theta_step_deg`)
- Adding angle-gating to `refine_line`
- Clustering-based support extraction
- Corner extraction from line segments
- Accumulator smoothing or windowed voting
- Any production code changes

## Dependencies between phases

```
Phase I (geometry helper) ──→ Phase II (fixtures)
Phase II confirms failures ──→ Phase III (synthetic isolation)
Phase II + III ──→ Phase IV (instrumentation)
```

Only Phase I is a prerequisite for Phase II.  Phases III and IV can be done in
parallel after Phase II is written (or sequentially).

## Directory layout after implementation

```
src/qr_reader/tests/detector/
├── test_hough.py              # Phase III — synthetic isolation tests added here
├── test_hough_harness.py      # Phases I, II, IV — new file
│   ├── _compute_finder_edges()    # Phase I
│   ├── _describe_support()        # Phase IV
│   ├── class TestFixtureReal:     # Phase II
│   └── ...
```
