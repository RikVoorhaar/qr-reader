# Plan 015 — Experiment Result

Branch `plan-015-perspective-finder` — all Plan 015 features enabled with no feature-flags or fallbacks.

## Active pipeline

- Per-finder: anisotropic pitch, two-family orientation, projective scanlines, 8-DOF homography refinement — always on.
- Global: DLT from 12 finder corners, LM-refined — always on.

## Test suite

| Suite | Result |
|---|---|
| `pytest src/qr_reader/tests/detector/test_finder_perspective.py -q` | **25 passed, 4 skipped** |
| `pytest src/qr_reader/tests/test_detector.py -q` | **16 failed, 2 passed** (unchanged) |
| `pytest -q` | 724 passed, 18 failed, 5 skipped |

## Full pipeline (qr_benchmark.py)

```
ValueError: No finder-pattern triplet found
```

The per-finder 8-DOF homography refinement produces per-finder corners that
break the cross-finder association logic (`find_all_associations`), so the
full detection pipeline cannot complete on the v12 benchmark.

## Single-finder perspective (synthetic sweep)

| Angle | rho RMSE | homography RMSE |
|---|---|---|
| 30° yaw | 6.85 px | **4.31 px** (37% better) |
| 30° pitch | 6.85 px | **4.40 px** (36% better) |
| 30°+30° | 25.24 px | **20.58 px** (18% better) |

Convergence basin: 93.8% (≥90% target).

## Conclusion

The per-finder fitting chain works well in isolation but the 8-DOF homography
refinement introduces orientation ambiguities between finder patterns that
break the global association step. Until per-finder homography output is
stabilised to a consistent canonical frame across all finders, the full
pipeline will fail on the benchmark.
