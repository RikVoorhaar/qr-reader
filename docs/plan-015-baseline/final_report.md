# Plan 015 Final Report

Generated after Steps 6 and 7 on branch `plan-015-perspective-finder`.

## Final state

| Flag / Setting | Status |
|---|---|
| `estimate_anisotropic_pitch` | **default True** |
| `use_two_families` | **default True** |
| `use_projective_scanlines` | **default True** |
| `use_finder_homography` | default **False** (passes isolated benchmark; associations still unstable globally) |
| `use_global_dlt_from_corners` | default **False** ( DL T over 12 corners overfits on synthetic benchmark; kept as fallback) |

## Test suite

```bash
pytest src/qr_reader/tests/test_detector.py -q
```

**Result as of final commit:** `16 failed, 2 passed in 10.23s`

- The failure count is identical to the baseline (`16 failed, 2 passed`).
- The two passing tests are the frontoparallel / axis-aligned smoke tests.
- All remaining failures are `ValueError: No finder-pattern triplet found`; these are pre-existing from Plan 013 integration and are unchanged by Plan 015.

### Full repository

```bash
pytest -q
```

**Result as of final commit:** `724 passed, 18 failed, 5 skipped`

## QR benchmark (v12-default)

```bash
python src/qr_reader/scripts/qr_benchmark.py
```

**Result:**

- all correct? `False`
- pct incorrect: `48.06%`
- total time: `2.12s`
- detect time: `1.74s`
- sample time: `379.17ms`
- decode time: `1.41ms`

### Comparison with baseline

| Metric | Baseline | Final | Change |
|---|---|---|---|
| pct incorrect | `49.77%` | `48.06%` | **-1.71 pp** (improvement) |
| detect time | `1.74s` | `1.74s` | unchanged |

## Single-finder perspective benchmark

```bash
pytest src/qr_reader/tests/detector/test_finder_perspective.py -q
```

**Result:** `25 passed, 4 skipped in 9.32s`

| Step | Diagnostic | Status |
|---|---|---|
| 2 | `m_u / m_v` monotonic with perspective | Pass |
| 3 | Two-family angle error < 5° at 30° | Pass |
| 4 | Projective scanline RMSE lower than equal-spacing at 30° | Pass |
| 5 | Homography RMSE beats rho; convergence basin ≥ 90% | Pass |

## Architecture updates

- `AGENTS.md` Data Flow and Module Map updated to reflect the current pipeline.
- `README.md` Architecture section updated to reflect the current pipeline.

## Known limitations / follow-up

- `use_finder_homography=True` is not yet safe in the full pipeline: the 8-DOF
  per-finder homography can converge to mirror / cyclic-shifted solutions that
  break `find_all_associations` consistency across the three finder patterns.
  An alignment helper (`finder_fit._align_quad_order`) was added, but complete
  global consistency remains unsolved.
- `use_global_dlt_from_corners=True` lowers the 12-point reprojection error but
  degrades decode performance on the v12 benchmark, indicating overfitting to
  the 12 finder corners. It is retained as an opt-in path with a condition-number
  guard and a similarity-init fallback.

## Conclusion

Plan 015 improves the single-finder fit substantially and does not regress the
baseline test suite or v12 benchmark. The new routines are available for further
work; enabling the per-finder homography and global DLT paths globally is
left as a follow-up once the association-consistency issues are resolved.
