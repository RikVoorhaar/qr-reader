# Plan 008 — Hough Pipeline Ablation Sweeps

> **Goal:** Run systematic parameter/config sweeps across the full Hough pipeline
> to find the best combination for each component, building cumulatively from
> vote formation through peak extraction, edge continuity, support collection,
> endpoint estimation, and finder-prior scoring.
> 
> **Context:** The deep research report separates the 13 fixture failures into four
> stages: A = edge continuity, D = vote localisation, C = support segmentation,
> B = ROI semantics.  Prior phased-fix experiments (plan-007) exhausted
> single-parameter approaches.  This plan sweeps broader design axes — soft
> voting, hysteresis-lite edge linking, support grouping, finder-prior gates —
> and measures the combinatorial effect.

---

## Overview

Each experiment E1–E8 is a **parameter/config sweep** against the fixture cases
`v12-default`, `v12-clean`, `v5-default` (seed=42).  Experiments build on each
other: the best config from E3 feeds into E4, etc.  The harness writes one CSV
row per parameter set and one diagnostics folder per case.

**Pass criteria** are stated per experiment.  The primary rule across all
experiments: **zero regression on `v12-clean`** (currently 0 failures).
A config that improves `v12-default` but regresses `v12-clean` is never
accepted.

The four algorithmic changes recommended by the report — soft angular voting,
hysteresis-lite, support segmentation, finder-prior gates — are the core of
experiments E3, E5, E6, and E8.  E1 and E2 are diagnostics that inform
whether later experiments need different parameter ranges.  E4 and E7 tune
existing parameters.

---

## Setup Phase — Ablation Harness

### Rationale

Plan-007 isolated individual fixes with accept/revert gates.  Plan-008 needs a
different tool: a deterministic harness that runs the fixture pipeline for many
parameter sets, writes structured results, and generates per-edge diagnostics.

### Change

Build `src/qr_reader/scripts/run_hough_ablation.py`:

```bash
python -m qr_reader.scripts.run_hough_ablation \
    --cases v12-default,v12-clean,v5-default \
    --mode <experiment> \
    --seed 42 \
    --out out/<experiment_dir>
```

The harness wraps the fixture logic from `test_hough_harness.py` and adds:

| Capability | Format |
|---|---|
| **Row-per-config CSV** | `case, config_key, D, A, C, B, peak_hit_rate, peak_snr_mean, peak_snr_p05, support_len_ratio_mean, support_len_ratio_p05, corner_reproj_median, corner_reproj_p95, n_zero_gt_roi, runtime_median_ms, runtime_p95_ms` |
| **Accumulator heatmaps** | Per-cluster per-GT-edge: `(θ,ρ)` heatmap with GT bin marker, nearest peak marker, vote-cloud overlay |
| **Per-peak support maps** | Image-space: edge pixels coloured by orthogonal distance to refined line, refined line clipped to ROI, segment endpoints marked |
| **Support-density plots** | 1-D projection `t ↦ support_density` along the line direction |
| **Edge-angle histograms** | Per-ROI histogram of NMS survivor gradient angles, weighted by magnitude |
| **ROI overlay** | Cluster ROI boundary, candidate centre, GT finder centre, expected `ρ = ±s/2` bands, all Hough lines clipped to ROI |
| **Rho-vs-theta scatter** | Scatter of edge-pixel votes for GT-matched support: `(θ, ρ)` per pixel |

The harness must be **deterministic** — fix all random seeds.  Each run
produces one output directory containing the CSV and per-case diagnostic
subdirectories.

### Dependencies

- All existing modules: `hough.py`, `edges.py`, `roi.py`, `clustering.py`,
  `finder_pattern.py`, `qr_gen.py`
- The fixture logic extracted from `test_hough_harness.py`
- `matplotlib` for diagnostics

### Gate (harness correctness)

- Reproduces the current baseline tallies exactly:
  `v12-default: D=2 A=2 C=4 B=5`, `v12-clean: total=0`, `v5-default: D=2 A=1 C=3 B=0`
- Generates all diagnostic outputs without errors
- Runs from command line with `--mode baseline` to verify

---

## E1 — ROI Audit

### Rationale

The report identifies ROI centering as a first-class experimental variable.
If the ROI centre is offset from the true finder centre, all downstream ρ
computations shift by the projected centre error (~same magnitude in px).
Additionally, ROIs that contain zero GT finder edges should never be sent
through the Hough pipeline — running them is a test artefact that produces
B phantoms.

This is a **diagnostic only** — no code change, just measurement.

### Run

```bash
python -m qr_reader.scripts.run_hough_ablation \
    --cases v12-default,v12-clean,v5-default \
    --mode roi_audit --seed 42 --out out/e1_roi_audit
```

### Metrics collected

| Metric | How computed |
|---|---|
| Centre error (px) | Euclidean distance between ROI centre and GT finder centre, per cluster per case |
| ROI-GT overlap | Number of GT finder edges whose segments intersect the ROI |
| ROI edge coverage fraction | Fraction of each GT edge's segment extent that falls within the ROI |
| Zero-GT-edge ROIs | ROIs where no GT finder edges overlap the ROI |

### Decision rule

- If **any cluster** has zero GT-edge overlap and produces Hough peaks → that
  cluster's B failures are a test artefact (skip in harness; informs E8).
- If centre errors are large (p95 > 4 px) → ROI origin correction is warranted
  before tuning ρ-dependent parameters in E4 and E8.
- If centre errors are small (p95 < 2 px) → ρ-gate experiments in E8 don't need
  origin correction first.
- Flag the cluster(s) with zero GT-edge ROIs; they should be excluded from
  all subsequent experiments' failure tallies.

Record the centre-error distribution and zero-GT-edge ROIs in the summary CSV.

---

## E2 — Vote-Cloud Audit

### Rationale

D failures have empty GT bins in the accumulator.  The report's key question:
is the vote spread in θ (angular quantisation), in ρ (radial displacement), or
in both?  This experiment answers that before we commit to any vote-formation
change.

### Run

```bash
python -m qr_reader.scripts.run_hough_ablation \
    --cases v12-default --mode vote_audit \
    --theta-step-deg 2 --theta-window-deg 0 --rho-step-px 1 \
    --seed 42 --out out/e2_vote_audit
```

### Metrics collected

For each D-failure GT edge, reconstruct the accumulator and classify:

| Diagnostic | How checked |
|---|---|
| `theta_spread` | Votes in GT ρ-bin exist at non-GT θ-bins (within ±5°) |
| `rho_spread` | Votes at GT θ exist in non-GT ρ-bins (within ±5 px) |
| `origin_shift` | Votes cluster around a ρ-value consistently offset from the GT ρ |
| `vote_dilution` | Votes exist at both GT θ and GT ρ but total weight < 0.25 × max_score |
| `empty` | No votes at all within ±5° and ±5 px of GT bin |

For each non-D GT edge (control), record the same metrics to compare
distributions.  Generate per-edge accumulator heatmaps with vote-cloud overlays.

### Parameter matrix

Single config — this is a diagnostic, not a sweep.

| Parameter | Value |
|---|---|
| `theta_step_deg` | 2 (baseline) |
| `theta_window_deg` | 0 (one-bin, baseline) |
| `rho_step_px` | 1 (baseline) |

### Decision rule

- If any D edge has `theta_spread` or `vote_dilution` → soft angular voting
  (E3) is well-motivated.
- If all D edges are `empty` (no votes near GT) → the problem is upstream of
  Hough (edge extraction or ROI centering); E3 won't help D.
- Classify every D edge.  No "unknown" classifications.
- Record the classification table in the summary.  This determines the
  `theta_window` range for E3.

---

## E3 — Angular Sweep

### Rationale

One-bin voting (current baseline) sends each edge pixel's vote to exactly one
quantised θ-bin.  If gradient directions vary by a few degrees under blur/noise,
votes split instead of pooling, and GT bins can be empty.  The report recommends
a small angular soft-window vote: each pixel votes into a ±K window around the
gradient-normal θ, weighted by an angular kernel.

### Change (in production code, `hough.py`)

Add two new parameters to `hough_vote_peaks`:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `theta_window_deg` | `float` | `0.0` | Half-window width in degrees. `0` = one-bin (current behaviour). |
| `vote_scheme` | `str` | `"onebin"` | `"onebin"`, `"gaussian"`, `"dot"` |

Implementation (NumPy, precomputed tables):

```python
if vote_scheme == "onebin":
    # Current behaviour: one theta index per pixel
    ...
elif vote_scheme == "gaussian":
    offsets = np.arange(-K, K + 1, dtype=np.int32)
    weights = np.exp(-0.5 * (np.deg2rad(offsets * theta_step_deg) / sigma_rad) ** 2)
    theta_idx = base_idx[:, None] + offsets[None, :]
    flat = theta_idx * n_rho + rho_idx[:, None]
    acc = np.bincount(flat[valid].ravel(),
                      weights=np.broadcast_to(weights[None, :], flat.shape)[valid].ravel(),
                      minlength=n_theta * n_rho).reshape(n_theta, n_rho)
elif vote_scheme == "dot":
    # w = s * max(0, n_θ · ĝ)^p  — avoids atan2 dependence
    ...
```

`K` is `ceil(theta_window_deg / theta_step_deg)`.  Use `np.bincount` for
accumulation — not `np.add.at`.

### Sweep matrix

| Parameter | Values |
|---|---|
| `theta_step_deg` | `0.5, 1, 2, 5` |
| `theta_window_deg` | `0, 1, 3, 6` |
| `vote_scheme` | `onebin, gaussian, dot` |
| `rho_step_px` | `1` (fixed) |

For `gaussian`: σ = `theta_window_deg / 3` (so weights at ±window are ~1% of peak).
For `dot`: power p = 3 (narrow kernel).

Total: 4 × 4 × 3 = 48 configs per case.

### Metrics

Primary: **GT peak hit rate** (fraction of GT edges whose (θ,ρ) bins contain ≥1
detected Hough peak).  Secondary: peak SNR (ratio of GT-bin score to mean
non-GT bin score in the same θ-band).

### Decision rule

- **Best config** = highest GT peak hit rate on `v12-default`, with zero new B
  on `v12-clean` (allowed: ≤ +1 B regression).
- If `gaussian` with `theta_window_deg=3` achieves ≥ +15% hit rate over
  baseline → soft voting becomes the new default.
- If best config is the same as baseline (`theta_step_deg=2`, `theta_window_deg=0`,
  `vote_scheme="onebin"`) → voting is not the bottleneck; skip further
  vote-formation work.

Record the best config.  Feed it into E4.

---

## E4 — Radial Sweep

### Rationale

Peak extraction currently uses a single NMS radius per axis
(`nms_radius_rho=6`, `nms_radius_theta=3`) and a single relative threshold
(`threshold_rel=0.25`).  The report notes that parallel finder-ring boundaries
are ~3 px apart, so a 6-bin ρ-NMS radius may merge distinct edges.  Separating
θ and ρ suppression scales and adding accumulator smoothing are standard
practices.  This experiment also sweeps ρ-step to test whether coarser bins
concentrate fragmented votes.

### Sweep matrix

| Parameter | Values |
|---|---|
| `rho_step_px` | `0.5, 1, 2` |
| `nms_radius_rho` | `2, 3, 4, 6` |
| `nms_radius_theta` | `1, 2, 3` |
| `acc_smooth` | `none, 1x3_triangular, 1x5_triangular` |

`1×3_triangular` = `[1,2,1]/4` along the ρ-axis only (avoid parallel-line
merging in θ).  `1×5_triangular` = `[1,2,3,2,1]/9`.  These use
`scipy.ndimage.convolve1d`.

Total: 3 × 4 × 3 × 3 = 108 configs.

All configs use the best angular config from E3.

### Metrics

Primary: D-failure count on `v12-default`, C-failure count on `v12-clean`.
Secondary: A/B counts, single-test regression (parallel lines 3 px apart must
remain distinct peaks — tracked via `test_horizontal_edges` /
`test_vertical_edges`).

### Decision rule

- **Best config** = lowest D count on `v12-default`, with C on `v12-clean` ≤ +1
  from baseline, and the edge-unit-test suite passing.
- If no config improves D → peak-extraction tuning is not the bottleneck for D;
  the problem is upstream (vote formation or edge extraction).
- If a config improves D but regresses edge unit tests → it merges parallel
  lines; discard.
- If a config improves C → it correctly de-duplicates parallel edges; note for
  E6.

Record the best config.  Feed it into E5.

---

## E5 — Edge Continuity Sweep

### Rationale

A failures are an edge continuity problem: NMS produces thin edges but doesn't
preserve weak connected continuation (gaps of 4–7 px).  The Canny control
experiment (I10) eliminated A failures on `v12-default` — Canny's hysteresis
(the step that keeps weak pixels when they're connected to strong ones) is the
mechanism.

The report recommends a minimal hysteresis-lite stage: keep the existing
Sobel + NMS, then threshold at a high and low percentile, and 8-connected flood
from strong to weak pixels.  Use linked edges for **support collection** but
vote from the thin NMS image (to keep Hough peaks sharp).

### Change (in `edges.py` or `hough.py`)

Add a function:

```python
def hysteresis_link(nms, angle, high_percentile, low_percentile):
    """8-connected flood from strong (>= high) to weak (>= low) pixels."""
    ...
```

Expose in `hough_vote_peaks` via parameters:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `hysteresis` | `str | None` | `None` | `None` (off), `"lite"` |
| `hysteresis_high_pct` | `float` | `90` | Percentile for strong-edge threshold |
| `hysteresis_low_pct` | `float` | `70` | Percentile for weak-edge threshold |

Voting still uses the raw NMS image.  Support collection in `refine_line` uses
the linked mask.

### Sweep matrix

| Parameter | Values |
|---|---|
| `hysteresis` | `none, lite` |
| `hysteresis_high_pct` | `80, 85, 90, 95` |
| `hysteresis_low_pct` | `60, 65, 70, 75, 80` |

Constrained: `low_pct < high_pct`.  Total: 1 + 4 × 5 − (invalid pairs) ≈ 1 +
(valid pairs).  Sweep `high_pct` and `low_pct` pairs where `high - low ≥ 10`.

Additional sweep on a related axis from the pipeline catalogue:

| Parameter | Values |
|---|---|
| `threshold_percentile` (NMS edge pixels used for voting) | `70, 80, 90, 95` |

This controls which NMS pixels enter the Hough accumulator, complementing the
hysteresis linking (which controls which pixels enter support collection).

Total configs: ≈ 1 (none) + 15 pairs + 4 thresholds = 20 configs.

### Metrics

Primary: A-failure count on `v12-default`.  Secondary: B count on
`v12-clean`, runtime increase.

### Decision rule

- **Best config** = A count reduced by ≥ 50% (from 2 to ≤ 1), with zero new B
  on `v12-clean` and runtime increase < 20%.
- If `hysteresis = lite` improves A → make it the default for the remaining
  experiments.
- If `threshold_percentile` improves A without hysteresis → prefer it (simpler).

Record the best config.  Feed it into E6.

---

## E6 — Support Sweep

### Rationale

`refine_line` currently admits support pixels with `distance_thresh=1.5` px from
the quantised line and then uses longest-contiguous-run for endpoint extraction.
The report identifies three levers: (a) distance threshold, (b) angle threshold
for support admission (Phase 4's opt-in `angle_gate_deg`), and (c) 1-px dilation
for support collection only (never for voting).

### Sweep matrix

| Parameter | Values |
|---|---|
| `distance_threshold_px` | `1, 2, 3, 5` |
| `angle_threshold_deg` | `5, 10, 20, none` |
| `support_dilate` | `0, 1` |
| `support_grouping` | `none, cc` |

`support_dilate` = dilate the NMS/linked edge mask by N pixels before support
collection (vote mask is unchanged).  `cc` = connected-component grouping within
the support set after projecting to line coordinate `t` (replace
longest-contiguous-run with largest-CC).

Total: 4 × 4 × 2 × 2 = 64 configs.

All configs use the best configs from E3, E4, E5.

### Metrics

Primary: C-failure count on `v12-default`, support-length ratio mean
(refined segment span / GT segment span).  Secondary: A count (wider distance
threshold may fix A gaps), B count.

Support-length ratio is computed per GT-matched edge:

```
support_len_ratio = min(refined_span, gt_span) / max(refined_span, gt_span)
```

A ratio near 1.0 means the refined segment matches the GT extent.  Overshoot
gives ratio < 1.0; undershoot also gives ratio < 1.0.

### Decision rule

- **Best config** = highest support-length ratio mean on `v12-default`, with C
  not worse than baseline.  If two configs tie, prefer the one with fewer C failures.
- If `support_grouping = cc` improves C → replace longest-contiguous-run with CC
  grouping.
- If `support_dilate = 1` improves A without worsening C → enable it.
- If `angle_threshold` improves C without regressing v12-clean → set it as
  default (Phase 4 already validated this in isolation).

Record the best config.  Feed it into E7.

---

## E7 — Endpoint-Model Sweep

### Rationale

The current longest-contiguous-run endpoint model chooses one support run
without modelling adjacent structures.  The report recommends testing three
alternatives: (a) the current longest-run (baseline), (b) CC-based longest
(extract endpoints from the largest connected component of the support
projection, after gap-tolerance-based bridging), and (c) RANSAC segment
estimation (model is still a line, inlier set constrained in both orthogonal
distance and t-contiguity).

### Change (in `refine_line`)

Add `endpoint_model` parameter:

| Value | Behaviour |
|---|---|
| `"longest_run"` | Current: longest contiguous run with gap bridging |
| `"cc_longest"` | Group support projections into connected components (threshold = `gap_tolerance`), take largest CC's min/max projection |
| `"ransac_segment"` | RANSAC on the support set: sample pairs, fit line, count inliers within `distance_thresh` and t-contiguous (no gap > `gap_tolerance`), take largest inlier set |

### Sweep matrix

| Parameter | Values |
|---|---|
| `endpoint_model` | `longest_run, cc_longest, ransac_segment` |
| `gap_tolerance_px` | `2, 3, 5` |

Total: 3 × 3 = 9 configs.  Use best configs from E3–E6.

### Metrics

Primary: corner reprojection error (p95) on `v12-default`, C-failure count.
Secondary: A-failure count (different endpoint model may change span adequacy).

Corner reprojection error = Euclidean distance between the refined segment's
endpoints (projected back to image coordinates via homography) and the GT
corner points.  Only computed for edges that match a GT finder boundary.

### Decision rule

- **Best config** = lowest C count, with corner reprojection p95 < 3 px and
  no A regressions.
- If `ransac_segment` improves C without raising runtime > 50% → accept it.
- If `cc_longest` is comparable to `ransac_segment` and faster → prefer it.
- If no config improves over `longest_run` → endpoint model is not the
  bottleneck for C.

Record the best config.  Feed it into E8.

---

## E8 — Finder-Prior Sweep

### Rationale

B failures are structurally indistinguishable from finder edges at the pixel
level, but they don't satisfy the geometric constraint that a finder outer edge
should sit at |ρ| ≈ s/2 from the finder centre (where s is the cluster scale).
A ρ-gate and a quad-scoring function that rewards parallelism, orthogonality,
and centre containment should reject data-region phantoms.

The report also recommends testing two alternative line-detection baselines
(Progressive Probabilistic Hough, LSD) as "oracles" to check whether Hough
voting or segment extraction is the limiting factor.

### Change (in new module `finder_scoring.py` or in `hough.py`)

Add a function:

```python
def score_finder_quad(lines, scale, centre):
    """
    Score a quadruple of lines as a finder outer boundary.

    S = 2.0 * z_support + 1.5 * z_rho + 1.0 * z_parallel
        + 1.0 * z_perp + 1.0 * I_contains_centre
        - 1.5 * z_overshoot - 2.0 * I_zero_gt_overlap
    """
    ...
```

ρ-gate: accept lines where `||ρ| − s/2| < τ_ρ`.  τ_ρ swept as fraction of s.

### Sweep matrix

| Parameter | Values |
|---|---|
| `rho_gate_frac` | `0.10, 0.15, 0.20, 0.25, none` |
| `quad_score` | `basic, finder` |
| `baseline` | `hough (ours), ppht (OpenCV), lsd (OpenCV)` |

`basic` quad score = just endpoint-intersection proximity (current approach).
`finder` quad score = the weighted function above.

`ppht` = `cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)` or
`cv2.HoughLinesP` with probabilistic mode.  `lsd` = `cv2.createLineSegmentDetector()`.

Total: 5 × 2 × 3 = 30 configs.

If `cv2.ximgproc` is unavailable, `lsd` falls back to
`cv2.createLineSegmentDetector()`.  If neither is available, skip those
baseline rows (marked `n/a`).

### Metrics

Primary: B-failure count on `v12-default`.  Secondary: B count on
`v12-clean`, corner reprojection error for matched edges.

For baselines (`ppht`, `lsd`): also record total lines found per ROI,
runtime, and segment-length distribution.

### Decision rule

- **Best config** = B count near zero on `v12-default`, with B count = 0 on
  `v12-clean` (no regression).
- If `finder` quad score reduces B beyond `basic` → accept finder scoring.
- If `ppht` or `lsd` eliminates B while our Hough doesn't → the problem is in
  our Hough voting or refinement, not in the line-detection paradigm.
- If our Hough + `finder` scoring matches the baselines → no need to adopt
  OpenCV line detectors in production.

Record the best config.

---

## Execution Order

```
Setup  (harness)          → validates baseline reproduction
  │
  ├─ E1  (ROI audit)       → diagnostics only
  ├─ E2  (vote-cloud audit) → diagnostics only
  │
  └─ E3  (angular sweep)   → feeds best config into E4
        │
        └─ E4  (radial sweep)   → feeds best config into E5
              │
              └─ E5  (edge continuity) → feeds best config into E6
                    │
                    └─ E6  (support sweep)   → feeds best config into E7
                          │
                          └─ E7  (endpoint model)  → feeds best config into E8
                                │
                                └─ E8  (finder priors)
```

E1 and E2 can run in parallel.  E3–E8 run sequentially — each experiment's
best config becomes the baseline for the next.  E1/E2 results may influence
E3's `theta_window` range and E8's `rho_gate_frac` range (narrower if centre
errors are small, wider if large).

---

## Validation Commands

```bash
# Harness baseline validation
.venv/bin/python -m qr_reader.scripts.run_hough_ablation \
    --cases v12-default,v12-clean,v5-default --mode baseline \
    --seed 42 --out out/baseline

# Full test suite (after any production-code change)
.venv/bin/python -m pytest src/qr_reader/tests/ -q
```

All production-code changes must preserve ≥ 715 passes in the full suite
(no regressions on existing tests).

---

## Target End State

After all accepted configs, `v12-default` should show:

```
Failure A:  0 or 1    (from 2 — E5 hysteresis-lite)
Failure B:  0 or 1    (from 5 — E8 finder-prior scoring)
Failure C:  ≤ 2       (from 4 — E6 support grouping + E7 endpoint model)
Failure D:  0 or 1    (from 2 — E3 soft voting + E4 radial tuning)
Total:      ≤ 5       (from 13 → ≥ 60% reduction)
Match rate: ≥ 83%     (5/6 GT edges matched, from 4/6 = 67%)
```

Plus: `v12-clean` must remain at 0 failures; `v5-default` must improve or
stay level.  Diagnostics in `out/e*/` must provide clear per-failure-mode
explainability.

If any stage fails to produce a beneficial config (no config beats baseline
for its targeted failure mode), document the negative result and the strongest
remaining hypothesis for why that stage is not the bottleneck.  Move to the
next experiment.

## Files Affected

| File | Experiments touching it |
|------|------------------------|
| `src/qr_reader/scripts/run_hough_ablation.py` | Setup (new file) |
| `src/qr_reader/detector/hough.py` | E3 (theta_window, vote_scheme), E4 (rho_step, nms radii, acc_smooth), E5 (hysteresis), E6 (support_grouping, support_dilate), E7 (endpoint_model), E8 (rho_gate, quad_score) |
| `src/qr_reader/detector/edges.py` | E5 (hysteresis_link) |
| `src/qr_reader/detector/finder_scoring.py` | E8 (new file — quad scoring) |
| `src/qr_reader/tests/detector/test_hough.py` | All (verify no regressions) |
| `src/qr_reader/tests/detector/test_hough_harness.py` | All (fixture logic extracted to harness) |
| `docs/plan-008-hough-ablation-sweeps.md` | This document |
| `docs/deep-research-report.md` | Source of experiment design |
