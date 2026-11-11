# Plan: multi-triplet decode with cross-ratio version estimation

Status: proposed, not implemented.

## Goal

Change the main pipeline so that it does not stop at the first valid triplet. Instead:

1. Detect all finder patterns and all valid triplets.
2. Estimate the QR version for each triplet using the cross-ratio estimator in
   `detector/landmarks.py` and `detector/version.py`.
3. Fit a homography for a small number of candidate versions per triplet.
4. Sample and decode.
5. Return the first payload that decodes. If none decode, return a diagnostic
   result explaining what was tried.

This replaces two current behaviours:

- `detector._run_detection` takes `raw_triplets[0]` and gives up if that triplet
  is wrong.
- Version estimation loops over all forty legal `N` values and scores the timing
  pattern. The cross-ratio estimator already produces a ranked version estimate,
  so the full loop is unnecessary.

The change is mostly orchestration. The geometry, finder fitting, homography,
sampling and decoder modules already exist and do not need to be rewritten.

## Non-goals

- Do not change `find_valid_triplets` in this task. Another agent is working on
  triplet finding. The multi-triplet loop only helps when the correct triplet is
  already in the returned list.
- Do not rewrite the decoder or the sampler.
- Do not add alignment-pattern-based homography refinement.
- Do not promise that this fixes `qr_code_ - 15.jpeg`. That image has only one
  valid triplet, and the correct bottom-left finder was never fitted. It is a
  finder-recall/triplet-gate failure, not a version-selection failure.

## Current behaviour

Relevant code:

- `src/qr_reader/detector/detector.py`
  - `_run_detection(image)` runs binarisation, the 1:1:3:1:1 candidate scan,
    clustering, per-cluster ray fitting, deduplication, then
    `find_valid_triplets`.
  - It takes `raw_triplets[0]` (`detector.py`, around line 235).
  - It canonicalises the twelve outer corners of that triplet.
  - It loops `for N_cand in range(21, 178, 4)`, fits DLT + LM, computes the
    reprojection error, scores the timing pattern, and minimises `err - timing`.
  - It returns one `DetectionResult` with `H` and `version`.
- `src/qr_reader/detector/finder_pattern.py`
  - `find_valid_triplets` returns a `list[Triplet]`. It already computes a
    module pitch per finder, but the version estimate is not returned.
- `src/qr_reader/detector/landmarks.py`
  - `build_named_landmarks(triplet, fps)` builds the A–F landmark sets.
  - `get_colinear_quadruples` produces four outer quadruples even when inner
    corners are `None`.
- `src/qr_reader/detector/version.py`
  - `build_constraints`, `filter_constraints`, `estimate_version` implement the
    cross-ratio version estimator. It is currently used only by tests.
- `src/qr_reader/detector/sample.py`
  - `sample_qr_bits(image, H, N)` samples the module matrix.
- `src/qr_reader/decoder/decoder.py`
  - `decode(matrix)` returns the text or raises `DecodeError`.

The public APIs `detect_corners`, `detect_homography` and `detect_sample` all
wrap `_run_detection`, so they currently see only the first triplet and the
full version search.

## Evidence

Cross-ratio version estimation on current ray-fit outer corners:

| image | true version | cross-ratio pick | best score | runner-up |
|---|---:|---:|---:|---:|
| `qr_code_ - 99.jpeg` | 1 | 1 | 0.0045 | 0.1192 (v2) |
| `qr_code_ - 16.jpeg` | 10 | 10 | 0.0003 | 0.0025 (v11) |
| `qr_code_ - 35.jpeg` | 20 | 20 | 0.0002 | 0.0004 (v21) |

`qr_code_ - 15.jpeg` also estimates v20, but its only triplet is wrong: two
false positives in the data region plus the real top-right finder. The real
bottom-left finder was never fitted. The error is upstream of version
estimation.

Performance is the main risk. Captured benchmark sample times for the
multi-triplet failure images are:

| image | detected version | triplets | sample time |
|---|---:|---:|---:|
| `qr_code_ - 11.jpeg` | 20 | 4 | 4.4 s |
| `qr_code_ - 14.jpeg` | 22 | 4 | 5.2 s |
| `qr_code_ - 20.jpeg` | 10 | 5 | 1.6 s |
| `qr_code_ - 60.jpeg` | 14 | 2 | 2.6 s |

A quick prototype that sampled the top two versions for every triplet on those
four images did not finish within 15 minutes. The plan therefore must bound the
number of sampling calls. Do not implement an unbounded `for triplet: for
version: sample+decode` loop.

## Is this a major refactor?

No. It is a medium-sized, contained refactor:

- The expensive image preprocessing is already a straight-line block that can
  be extracted into one function.
- `find_valid_triplets` already returns all triplets.
- The cross-ratio estimator already exists.
- Homography fitting and sampling are already functions.
- The decoder is independent.

Estimated code change: roughly 300–500 lines including tests, benchmark
updates and docs. The risk is runtime and diagnostic clarity, not architectural
complexity.

## Target API

Create a new module `src/qr_reader/pipeline.py` so the detector stays free of
decoder imports.

```python
@dataclass
class DecodeAttempt:
    triplet_index: int
    version: int
    cross_ratio_score: float
    reprojection_error: float
    timing_score: float
    decode_ok: bool
    text: str = ""
    error: str = ""


@dataclass
class DecodeResult:
    success: bool
    text: str = ""
    H: np.ndarray | None = None
    version: int | None = None
    triplet_index: int | None = None
    attempts: list[DecodeAttempt] = field(default_factory=list)
    error: str = ""
    n_candidates: int = 0
    n_finders: int = 0
    n_triplets: int = 0
    timings: DetectionTimings = field(default_factory=DetectionTimings)


def decode_image(
    image: np.ndarray,
    *,
    expected_payload: str | None = None,
    max_triplets: int | None = 5,
    max_versions_per_triplet: int = 2,
    time_budget_s: float | None = 15.0,
) -> DecodeResult:
    ...
```

`decode_image` is the new full pipeline. `expected_payload` is optional; the
benchmark passes `"rikvoorhaar.com"` so a random successful decode cannot count
as a correct result. Without it, any non-raising `decode()` result is accepted.

### Backwards compatibility

Keep the existing APIs:

- `detect_corners(image) -> (corners, version)`
- `detect_homography(image) -> (H, version)`
- `detect_sample(image) -> matrix`

They should be reimplemented on top of the new detector context and cross-ratio
version estimate, but they should still return one candidate without decoding.
This keeps the existing tests and scripts working and avoids making
`detect_homography` depend on the decoder.

`decode_image` is the recommended entry point for "give me the payload".
Update the README to say so.

## Design

### 1. Extract the detector context

Add to `detector/detector.py`:

```python
@dataclass
class FinderContext:
    img_gray: np.ndarray
    fps: list[FinderPattern]
    score_map: dict[int, float]
    triplets: list[Triplet]
    timings: DetectionTimings
    n_candidates: int
    n_finder_patterns: int
    n_triplets: int


def _detect_finder_context(image: np.ndarray) -> FinderContext:
    ...
```

Move the current stages 1–5 into `_detect_finder_context`:

1. grayscale conversion
2. `binarize_image`
3. `find_alignment_patterns_2d`
4. `cluster_candidates`
5. per-cluster `fit_finder_ray`
6. deduplication
7. `find_valid_triplets`

Record the same `DetectionTimings` fields. Raise on failure; the wrappers catch
and convert to `DetectionResult` / `DecodeResult`.

Then `_run_detection` becomes:

```python
def _run_detection(image):
    result = DetectionResult(success=False)
    try:
        ctx = _detect_finder_context(image)
        result.n_candidates = ctx.n_candidates
        result.n_finder_patterns = ctx.n_finder_patterns
        result.n_triplets = ctx.n_triplets
        best = _best_detection_candidate(ctx)
        result.success = True
        result.H = best.H
        result.version = best.version
    except Exception as exc:
        result.success = False
        result.error = exc
    return result
```

Do not decode inside `_run_detection`.

### 2. Cross-ratio version estimation

Add helpers in `detector/detector.py` (or a new
`detector/triplet_candidates.py` if the file gets too large):

```python
@dataclass
class VersionEstimate:
    version: int
    score: float


def _cross_ratio_versions(triplet, ctx) -> list[VersionEstimate]:
    lm = build_named_landmarks(triplet, ctx.fps)
    constraints = filter_constraints(build_constraints(lm))
    if not constraints:
        return []
    v_best, scores = estimate_version(constraints)
    order = np.argsort(scores)
    return [
        VersionEstimate(version=int(i + 1), score=float(scores[i]))
        for i in order
        if np.isfinite(scores[i])
    ]
```

`estimate_version` returns scores for v1..v40. The list is already ranked.
Keep the raw score for diagnostics and ordering.

Fallback if the cross-ratio constraints are empty or all scores are infinite:

```python
def _versions_from_triplet_distance(triplet, ctx) -> list[VersionEstimate]:
    # The old geometric shortcut:
    # m_i = mean outer side length / 7
    # d1 = |TR - TL|, d2 = |BL - TL|, d3 = |TR - BL|
    # N_est = round((d1 + d2 + d3 / sqrt(2)) / (3 * mean(m_i)) + 7)
    # Return V_est and its immediate neighbours.
```

Use this only as a fallback. Do not fall back to the full forty-version search
unless a debug flag is explicitly set.

### 3. Rank triplets

For each triplet, compute the ranked versions once. The cross-ratio estimator is
cheap compared with sampling.

```python
@dataclass
class RankedTriplet:
    index: int
    triplet: Triplet
    versions: list[VersionEstimate]
    best_score: float
    module_spread: float
    right_angle_error: float
```

Sort by:

1. `best_score` ascending (lower cross-ratio error is better)
2. `module_spread` ascending (relative module-pitch spread)
3. `right_angle_error` ascending

Do not rely on the order returned by `find_valid_triplets`.

If a triplet has no finite cross-ratio score, use the triplet-distance fallback
and put it after the scored triplets.

### 4. Fit a homography for a candidate version

Extract the current canonicalisation and 12-point fitting logic into:

```python
def _canonical_triplet_corners(ctx, triplet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # global_u = normalise(TR centre - TL centre)
    # global_v = normalise(BL centre - TL centre)
    # canonicalise each finder's four corners into TL, TR, BR, BL order


def _fit_homography_for_version(ctx, triplet, version) -> CandidateFit:
    N = 17 + 4 * version
    tl_c, tr_c, bl_c = _canonical_triplet_corners(ctx, triplet)
    src, dst = _build_twelve_correspondences(tl_c, tr_c, bl_c, N)
    H = estimate_homography_dlt(src, dst)
    try:
        H = refine_homography_lm(H, src, dst, loss="linear")
    except Exception:
        pass
    err = mean(norm(project_points(H, src) - dst))
    timing = _score_timing_pattern(ctx.img_gray, H, N)
    return CandidateFit(H=H, version=version, N=N,
                        reprojection_error=err, timing_score=timing)
```

Keep the current coordinate conventions and the existing `_canonicalize_corners`
logic. Do not change the homography module.

### 5. Multi-triplet decode loop

`decode_image` algorithm:

```python
ctx = _detect_finder_context(image)
ranked = _rank_triplets(ctx)

attempts = []
t_start = perf_counter()

for rank in ranked[:max_triplets]:
    versions = rank.versions[:max_versions_per_triplet]
    for version_est in versions:
        fit = _fit_homography_for_version(ctx, rank.triplet, version_est.version)
        try:
            bits = sample_qr_bits(ctx.img_gray, fit.H, fit.N)
            text = decode(bits)
        except Exception as exc:
            attempts.append(DecodeAttempt(..., decode_ok=False, error=str(exc)))
            continue

        ok = expected_payload is None or text == expected_payload
        attempts.append(DecodeAttempt(..., decode_ok=ok, text=text))
        if ok:
            return DecodeResult(success=True, text=text, H=fit.H,
                                version=fit.version, triplet_index=rank.index,
                                attempts=attempts, ...)

    if time_budget_s is not None and perf_counter() - t_start > time_budget_s:
        break

return DecodeResult(success=False, attempts=attempts, ...)
```

Important details:

- Catch exceptions from `sample_qr_bits` and `decode` per attempt. One bad
  candidate must not abort the whole search.
- Order the version attempts by cross-ratio score. Try the best version first.
- Use `expected_payload` to reject wrong-but-decodable results when the caller
  knows the expected text.
- Stop at the first accepted payload.
- If `max_triplets` is `None`, try all ranked triplets. Keep the default small
  (5) because sampling dominates runtime.
- If `time_budget_s` is reached, stop and return the attempts collected so far.

### 6. Sampling cost and attempt budget

The sampler is the bottleneck. Do not sample every version for every triplet.

Recommended defaults:

- `max_triplets=5`
- `max_versions_per_triplet=2`
- `time_budget_s=15.0`

Additional cheap optimisations, in order:

1. **One-version-first pass.** Try only the best cross-ratio version for each
   triplet first. Only if all triplets fail, try the second version. This keeps
   the common case to one sample per triplet.
2. **Score-margin gate.** Only try a second version if its cross-ratio score is
   close to the best (for example `score2 < 2 * score1` or absolute margin).
   The v1/v10/v20 examples have clear margins, so a second attempt is usually
   unnecessary.
3. **Vectorise `sample_qr_bits` (optional).** The current implementation calls
   `map_coordinates` once per module for the 3×3 neighbourhood. If the budget is
   exceeded, replace it with one vectorised `map_coordinates` call over all
   module centres or all 3×3 sub-samples. This is a separate, well-contained
   optimisation and should not block the orchestration change.

Record the number of sampling calls in `DecodeResult.attempts` and in the
benchmark output.

## File-by-file changes

### `src/qr_reader/detector/detector.py`

- Add `FinderContext`.
- Extract `_detect_finder_context`.
- Add `_canonical_triplet_corners`, `_build_twelve_correspondences`,
  `_fit_homography_for_version`.
- Add `_cross_ratio_versions` / `_versions_from_triplet_distance`.
- Add `_rank_triplets` and `_best_detection_candidate`.
- Rewrite `_run_detection` to use the context and the cross-ratio version
  estimate. Keep the `DetectionResult` contract and failure behaviour.
- Remove the forty-version loop and the timing-score selection from the normal
  path. Keep `_score_timing_pattern` for diagnostics and for
  `CandidateFit.timing_score`.
- Do not modify `find_valid_triplets`.

### `src/qr_reader/pipeline.py` (new)

- `DecodeAttempt`, `DecodeResult`, `decode_image`.
- Import `decode` from the decoder here, not in `detector.py`.

### `src/qr_reader/scripts/benchmark_captured.py`

- Replace `_run_detection` + `sample_qr_bits` + `decode_qr` with
  `decode_image(gray, expected_payload=PAYLOAD, ...)`.
- Record `n_triplets`, `n_triplets_tried`, `n_versions_tried`,
  `selected_triplet`, `selected_version`, `decode_ok`, `text`,
  `detection_time_s`, `decode_time_s`, `total_time_s`.
- Keep the six-stage funnel but map stage 4 to "at least one candidate fit/
  version" and stage 5 to "decodable".
- Add an attempts/timing summary.

### `src/qr_reader/scripts/qr_benchmark.py`

- Use `decode_image` for decode success instead of `detect_sample` + `decode`.
- Keep `detect_corners` for corner error and version accuracy.
- Record the selected version and the number of attempts.

### `README.md`

- Update the architecture flow:
  - replace "Associate colinear finder patterns / Find L-shaped triplets"
    with "Find all valid triplets"
  - replace "Estimate version (inter-finder distance / pitch)" with
    "Estimate version by cross-ratio for each triplet"
  - add "Try candidate versions, sample and decode until one succeeds"
- Add `decode_image` to the public API section.

### `AGENTS.md`

- Add `pipeline.py` to the module map.
- Update the Data Flow to show the multi-triplet/decode loop.

### `CONTEXT.md`

- Add `Decode Attempt` / `Decode Candidate` terms if the implementation exposes
  them.
- Correct the `Homography` entry: RANSAC exists but is not in the production
  path.
- Correct the `Alignment Pattern` entry if touched: the current
  `find_alignment_patterns_2d` is a 1:1:3:1:1 candidate scanner, not the small
  1:1:1:1:1 alignment patterns.

## Tests

Add tests under `src/qr_reader/tests/`.

### Unit tests

- `test_cross_ratio_versions_outer_only`: build `NamedLandmarks` with
  `inner_corners=None`, call `build_constraints` / `filter_constraints` /
  `estimate_version`, assert the correct version for synthetic v1, v5, v10 and
  v20 corner sets.
- `test_rank_triplets_orders_by_cross_ratio_score`: construct two fake triplets
  with known scores and assert the order.
- `test_versions_from_triplet_distance_fallback`: remove all constraints and
  assert the fallback returns a small neighbour list around `N_est`.
- `test_fit_homography_for_version`: reuse the current 12-corner synthetic
  fixture; assert reprojection error is near zero for the correct version.

### Integration tests

- `test_decode_image_clean_v1` and `test_decode_image_clean_v5`: use
  `make_qr_image` / `generate_test_image`, assert `decode_image` returns the
  expected text and version.
- `test_decode_image_multiple_triplets`: use a captured image with more than one
  triplet if the dataset is available (for example `qr_code_ - 6.jpeg` or
  `qr_code_ - 7.jpeg`); otherwise skip. Assert that at least one attempt is
  recorded and that a successful image still decodes.
- `test_decode_image_expected_payload_mismatch`: pass a wrong expected payload
  and assert the result is unsuccessful but the attempts contain a decoded text.
- `test_decode_image_time_budget`: monkeypatch `sample_qr_bits` to sleep, set a
  short budget, assert the loop stops.
- `test_existing_detection_apis_still_work`: the existing
  `tests/test_detector.py` suite should pass unchanged.

### Regression note

Add `qr_code_ - 15.jpeg` as a documented known failure, not a passing test. The
correct triplet is not in the list; the test should assert the failure is
reported as "no decodable candidate", not as a crash. The triplet agent owns the
underlying finder-recall fix.

## Benchmarks and acceptance criteria

Run:

```sh
pytest
python src/qr_reader/scripts/benchmark_captured.py
python src/qr_reader/scripts/qr_benchmark.py
```

Baseline:

- Captured: 100 images, 49 reach homography/version, 40 decode to
  `rikvoorhaar.com`.
- Multi-triplet failures with current code: `qr_code_ - 11`, `-14`, `-20`,
  `-60`.
- `qr_code_ - 15` has only one triplet and is not expected to improve.

Acceptance:

1. No regression on the existing detector tests.
2. Captured correct-payload count is at least 40. Target: improve the four
   multi-triplet failures where the correct triplet is actually present.
3. Version accuracy on synthetic benchmarks is at least as good as the current
   full search.
4. The median captured total time does not increase by more than 2×. If it does,
   reduce `max_triplets` / `max_versions_per_triplet` or implement the
   vectorised sampler.
5. `DecodeResult.attempts` makes it obvious which triplet and version were
   tried and why each failed.

Do not accept a change that tries all versions for all triplets. The prototype
showed that path is too slow on high-version images.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sampling dominates runtime | Cap triplets/versions, one-version-first pass, time budget, optional vectorised sampler |
| False triplet has a good cross-ratio score | Decode success is the final filter; try next triplet |
| Correct triplet not in the list (image 15) | Out of scope; triplet agent must improve finder recall/gates |
| Changing `_run_detection` breaks existing APIs | Keep wrappers and run the existing test suite |
| Concurrent edits to `detector.py` / `finder_pattern.py` | Coordinate with the triplet agent; do not edit `find_valid_triplets` |
| Cross-ratio constraints empty | Fall back to the triplet-distance estimate and a small neighbour window |
| Wrong version decodes to plausible garbage | Use `expected_payload` in benchmarks; optionally expose it to callers |

## Rollout

1. **Refactor only.** Extract `FinderContext` and `_detect_finder_context`.
   Keep behaviour identical. Run tests.
2. **Cross-ratio version estimation.** Add `_cross_ratio_versions`, make
   `_run_detection` use the best triplet and best version instead of the
   forty-version loop. Run tests and the captured benchmark. Check version
   accuracy and timing.
3. **Multi-triplet decode.** Add `pipeline.decode_image` with caps and
   diagnostics. Run unit/integration tests.
4. **Benchmark and tune.** Update both benchmarks, measure, tune
   `max_triplets`, `max_versions_per_triplet`, and the score-margin gate.
5. **Docs.** Update README, AGENTS.md, CONTEXT.md.
6. **Optional sampler optimisation.** Vectorise `sample_qr_bits` if the
   runtime target is missed.

## Open questions

- Should `decode_image` live in `pipeline.py` or be a new top-level
  `qr_reader.decode_image` export? `pipeline.py` is the safer default.
- Should `detect_*` APIs also use the multi-triplet loop, or stay as
  single-candidate conveniences? The plan keeps them single-candidate to avoid
  decoding inside detection.
- Should the second version attempt be gated by a score margin or always tried?
  Start with a margin gate and measure.
- Is the current sampler fast enough after capping? If not, prioritise the
  vectorised sampler before tuning more candidates.

## Handoff checklist

- [ ] Read `detector/detector.py`, `finder_pattern.py`, `landmarks.py`,
      `version.py`, `sample.py`, `decoder/decoder.py`.
- [ ] Run the existing tests before changing anything.
- [ ] Extract the context without changing behaviour; tests green.
- [ ] Wire cross-ratio version estimation; tests green; captured benchmark
      compared.
- [ ] Add `decode_image`; tests green.
- [ ] Update benchmarks and record before/after numbers.
- [ ] Update docs.
- [ ] Do not modify `find_valid_triplets` without coordinating with the triplet
      agent.
- [ ] Report the image-15 caveat explicitly in the final summary.
