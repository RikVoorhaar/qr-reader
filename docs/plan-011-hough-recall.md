# Plan 011 — Hough Recall: 100% Finder Edge Detection

> **Goal:** Achieve 100% peak recall (and near-100% segment recall) on the
> v12-default benchmark by fixing systematic detection failures isolated in the
> benchmark.

## Current state (v12-default, E6 best, 5 seeds)

| Metric | Value |
|--------|-------|
| Peak recall | 0.806 |
| Segment recall | 0.811 |
| Mean IoU (TP) | 0.903 |
| Missed edges | 34/180 |

## Root cause analysis

Benchmark trace (v12-default seed=42, E6 best) showing **all 7 misses**:

```
TR_left0  ─ no edge pixels near GT line, no votes in theta column
TR_left1  ─ no edge pixels near GT line, no votes in theta column
TR_left2  ─ 14 edge pixels near line (2 correct angle), theta has 1188 votes but GT bin is 0
TL_left0  ─ 19 edge pixels near line (3 correct angle), theta has 609 votes but GT bin is 0
TL_left1  ─ 28 edge pixels near line (10 correct angle), theta has 609 votes but GT bin is 0
TL_left2  ─ 30 edge pixels near line (15 correct angle), theta has 2255 votes but GT bin is 0
BL_left1  ─ 16 edge pixels near line (0 correct angle), theta has 1762 votes but GT bin is 0
```

**Two distinct failure modes:**

1. **No-vote (left0, left1 for TR)**: The edge itself is invisible to the edge detector — no
   NMS pixels within 2px of the GT line with the correct angle. These are the
   innermost finder edges (k=0, k=1) and they face the QR interior, possibly
   obscured by adjacent modules or too low contrast against the background.

2. **Wrong-rho (all other left edges)**: Edge pixels exist near the line with
   roughly correct angles, but their (θ, ρ) votes land in bins **adjacent to**
   the GT bin, not directly in it. The GT theta column has substantial votes
   (up to 2255), but the GT rho bin itself is empty.  With
   `rho_step=1.0` and `nms_radius_rho=2`, this means: ρ quantization jitter
   from edge-position noise causes votes to scatter ±1–2 bins away from the
   true peak, making the GT bin zero while its neighbours have votes.

## Hypotheses

### H1: Edge signal is borderline for inward-facing finder edges
- **Evidence**: left0/left1 edges have ≤10 edge pixels with correct angle
- **Why**: these edges face the QR interior where adjacent dark/light modules
  blur together under rotation + noise, reducing the Sobel response
- **Potential fixes**: increase blur sigma (smooths noise → cleaner edges), use
  hysteresis to recover weak edge pixels connected to strong ones, or lower the
  Hough peak threshold

### H2: Rho quantization flinging scatters votes across adjacent bins
- **Evidence**: edge pixels exist at correct theta but vote into wrong rho
  bins; the GT rho bin is empty while neighbours have votes
- **Why**: with `rho_step=1.0`, edge pixel positions jitter by ±0.5 px from
  noise, and the `round()` quantization maps them to bin ±1 away half the time
- **Potential fixes**: accumulator smoothing in rho (`acc_smooth=1x3` or
  `1x5`), or finer rho step (`rho_step=0.5` — doubles bin count, more
  expensive), or sub-bin peak interpolation

### H3: Strong peaks in the same theta column suppress weak peaks
- **Evidence**: each missed left edge shares a theta column with a much
  stronger peak (1188-2255 total votes), and `nms_radius_rho=2` blanks ±2
  bins around every detected peak
- **Why**: iterative peak NMS zeroes out bins in a fixed radius around each
  detected peak; a strong peak at nearby rho kills the weak one
- **Potential fixes**: reduce NMS radius to 1×1, or use a ranked peak
  suppression that preserves peaks in the same theta column if they're from
  different edges, or add a "minimum rho separation" check

### H4: Inner edges have fewer visible modules → shorter arc → weaker peak
- **Evidence**: inner edges (k=1, k=2) have `k_vis = k` visible modules (1 or
  2 modules of edge pixels), while outer edges (k=0, k=7) have all 7 modules
  visible
- **Why**: the Hough peak strength is proportional to the number of edges
  voting; inner edges simply have fewer pixel votes
- **Potential fixes**: normalize threshold by expected edge length, or use a
  per-edge adaptive threshold

---

## Experiment phases

All experiments run on v12-default with the E6 best config as
baseline.  Acceptance criteria use the standard benchmark
(`--n-images 5`).

### Phase 1: rho-step and acc_smooth

**Goal**: Fix the "wrong rho" failure mode by reducing rho quantization
effects and smoothing the accumulator.

**Sweep grid**:

- `rho_step` ∈ `{0.5, 1.0, 2.0}`
- `acc_smooth` ∈ `{None, "1x3_triangular", "1x5_triangular"}`

**Acceptance**: Peak recall ≥ 0.92 on v12-default.  The "wrong-rho" misses
(left2 edges with edge pixels present) should resolve to HITs.

**Expected**: `rho_step=0.5` should help by aligning edge pixel rho with the
GT bin.  `acc_smooth` should help by pooling adjacent bins → the GT bin gets
some of the neighbour's votes.  Both together should be best.

### Phase 2: theta-step sweep

**Goal**: Confirm the current `theta_step_deg=0.5` is optimal for recall.

**Sweep grid**:

- `theta_step_deg` ∈ `{0.25, 0.5, 1.0, 2.0}`

**Acceptance**: No regression from Phase 1 best.  Expected: 0.5° is likely
optimal; 0.25° may overfit edge-angle jitter; 1.0°+ reduces angular
resolution and may lose signal.

### Phase 3: NMS radius

**Goal**: Prevent strong peaks in the same theta column from suppressing
weaker GT peaks nearby in rho.

**Sweep grid**:

- `nms_radius_rho` ∈ `{1, 2, 3}`
- `nms_radius_theta` ∈ `{1, 2, 3}`

**Acceptance**: Peak recall ≥ 0.95 on v12-default.  If a weak left edge shares
its theta column with a strong edge (e.g., a "right" edge at different rho),
reducing NMS radius should let the weak peak survive.

### Phase 4: threshold_rel

**Goal**: Lower the detection floor to catch weak-edge peaks.

**Sweep grid**:

- `threshold_rel` ∈ `{0.10, 0.15, 0.20, 0.25}`

**Acceptance**: Peak recall ≥ 0.97.  If lowering the threshold creates a flood
of FPs (>20/cluster), this phase fails and we instead need an
edge-length-normalized threshold (Phase 5).

**Risk**: Lower threshold → many weak phantom peaks → precision drops.  But
all FPs should be near-miss (within 12° of a GT normal), so they don't
confuse a downstream detector that picks the best peak per edge.

### Phase 5: (only if needed) Edge-length-adaptive threshold

**Goal**: If Phase 4 fails (too many FPs at low threshold), use GT edge length
to set per-edge expectations.

**Approach**: Instead of a global `threshold_rel`, compute the expected
Hough peak strength for each edge based on the number of edge-pixel modules
visible (`k_vis = min(k, 7-k)`).  Short edges (k=1: 1 module, k=2: 2 modules)
get a proportionally lower threshold.

**Acceptance**: Peak recall ≥ 0.97 without exploding FPs.

### Phase 6: blur_sigma and hysteresis

**Goal**: Fix the "no-vote" failure mode (edges  k=0/k=1 that have zero or
near-zero edge pixels).

**Sweep grid**:

- `blur_sigma` (in `extract_thin_edges`) ∈ `{0.5, 1.0, 1.5, 2.0, 2.5}`
- Hysteresis: `high_pct` ∈ `{90, 80}`; `low_pct` ∈ `{50, 30}`

Hysteresis works by keeping weak edge pixels (below the NMS threshold) if
they're 8-connected to strong ones.  For inward-facing finder edges, the edge
signal may be weak but connected to stronger outer-edge pixels.

**Acceptance**: Peak recall ≥ 0.98.  All "no-vote" edges (left0/left1) must
become HITs.  If hysteresis doesn't help, consider increasing the ROI padding
scale so the edges have more room to be visible.

### Phase 7: best-of-all combinatorial sweep

**Goal**: Find the single best parameter combination across all experiments.

**Sweep**: Grid of top 2 values from each phase (max ~32 combos).

**Test**: All 3 cases (v12-clean, v12-default, v5-default) × 5 seeds.

**Acceptance**:
- Peak recall ≥ 0.98 on v12-default
- Peak recall = 1.00 on v12-clean (no regression)
- Peak recall ≥ 0.95 on v5-default
- Segment F1 ≥ 0.90 on v12-default

### Phase 8: per-edge error analysis

**Goal**: Characterize remaining failures after Phase 7.

For each remaining missed edge, log:
- Is the GT segment degenerate (clipped to a point)?
- How many edge pixels exist near the line?
- What is the accumulator value at the GT bin vs. neighbouring bins?
- Is the GT bin suppressed by a stronger peak's NMS radius?

**Acceptance**: Any remaining FNs are understood and have a documented reason
(e.g., "line lies entirely outside ROI", "edge completely invisible at this
noise level").

---

## Success definition

The pipeline is ready for the next stage when:
1. Peak recall ≥ 0.98 on v12-default (35/36 edges detected, ±1)
2. Peak recall = 1.00 on v12-clean (no regression)
3. Segment F1 ≥ 0.90 on v12-default (i.e., most TP peaks refine to good
   segments)

If Phase 7 doesn't achieve this, the remaining gaps are likely in the ROI
extraction (edges outside the ROI) or edge extraction (edges invisible to
Sobel), and need a different kind of fix (larger ROI, different edge detector).

## Non-goals

- Fixing the near-miss FP problem (all FPs are near-miss — they don't confuse
  a downstream detector that picks the best peak per angle range)
- Segment endpoint accuracy (already good: IoU ≥ 0.90 for TP)
- Runtime optimization (finer rho/theta steps cost more time)
- v5-specific tuning (main target is v12)
