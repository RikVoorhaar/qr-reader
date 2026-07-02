# Plan 015 Baseline Summary

Recorded on branch `plan-015-perspective-finder` at the start of Step 0.

## Test Suite

```bash
pytest src/qr_reader/tests/test_detector.py -q
```

**Result:** `16 failed, 2 passed in 10.44s`

All failures are `ValueError: No finder-pattern triplet found` in
`_run_detection`.  This is the known state after Plan 013 integration.

## QR Benchmark

```bash
python src/qr_reader/scripts/qr_benchmark.py
```

**Result:**

- all correct? `False`
- pct incorrect: `49.77%`
- total time: `2.13s`
- detect time: `1.74s`
- sample time: `390.96ms`
- decode time: `1.41ms`

## Notes

- The baseline must not regress during Plan 015 implementation.
- Step 1 will add an isolated single-finder perspective benchmark; it must
  not change these numbers.
