# Plan: multi-triplet decode with cross-ratio version estimation

## Goal

Make the main pipeline try multiple valid triplets and use cross-ratio version
estimation instead of taking `raw_triplets[0]` and looping over all forty
versions. Return the first payload that decodes; otherwise return diagnostics.

## Current behaviour

- `_run_detection` takes `raw_triplets[0]` and ignores the other triplets.
- It loops `N = 21..177`, fits a homography for each size, and scores the timing
  pattern.
- `landmarks.py` and `version.py` already contain the cross-ratio version
  estimator, but only the tests use it.
- `sample_qr_bits` and `decode` are separate calls.

## Desired behaviour

- Extract the finder/triplet stage into a context so preprocessing runs once.
- For each triplet, rank versions with `build_named_landmarks` →
  `build_constraints` → `filter_constraints` → `estimate_version`.
- Sort triplets by their best cross-ratio score.
- For each triplet, try a small number of candidate versions: fit the existing
  12-corner DLT + LM homography, sample, decode. Stop on the first accepted
  payload.
- Add `decode_image(image, expected_payload=None, max_triplets=5,
  max_versions_per_triplet=2, time_budget_s=15)` in `qr_reader/pipeline.py`,
  returning a `DecodeResult` with the attempts and failure reasons.
- Keep `detect_corners`, `detect_homography`, and `detect_sample` as
  single-candidate wrappers.

## Rules

- No fallbacks. If cross-ratio constraints are empty or all scores are
  non-finite, raise `ValueError`. That is a bug in landmark construction, not a
  runtime condition to hide.
- Do not modify `find_valid_triplets`; another agent owns triplet finding.
- Do not optimise or rewrite `sample_qr_bits`. Sampler optimisation is out of
  scope. Measure the runtime increase and report it.
- Do not rewrite the decoder or the homography module.
- `qr_code_ - 15.jpeg` will not be fixed by this: its only triplet is wrong and
  the real bottom-left finder was never fitted. Note that in the final report.

## Files

- `detector/detector.py`: context extraction, cross-ratio ranking, candidate
  fitting, `_run_detection` uses the best triplet/version.
- `pipeline.py` (new): `decode_image`, `DecodeResult`, attempt records.
- `scripts/benchmark_captured.py` and `scripts/qr_benchmark.py`: use
  `decode_image`, record attempts and timing.
- `README.md`, `AGENTS.md`, `CONTEXT.md`: update the pipeline description.
- Tests: cross-ratio with outer corners only, empty constraints raise,
  multi-triplet decode, expected-payload mismatch, time budget; existing tests
  stay green.

## Acceptance

- Existing tests pass.
- Captured correct-payload count is at least 40; report before/after.
- Synthetic version accuracy is not worse than the current full search.
- `DecodeResult.attempts` shows which triplet and version were tried and why
  each failed.
- Report the measured runtime increase. Do not change the sampler.

## Caveats

- A false triplet can have a good cross-ratio score; decode success is the
  final filter.
- Cap the attempts. Sampling is 1.5–5.2 s per high-version attempt, so an
  unbounded triplet × version loop is too slow.
- `qr_code_ - 15.jpeg` remains a finder-recall/triplet-gate failure, not a
  version-selection failure.
