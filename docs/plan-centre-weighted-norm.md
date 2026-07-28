# Plan — Centre-Weighted ROI Normalization

## Goal

Replace the histogram-bimodal-peak normalization in `ray-profile.py` with
centre-weighted percentiles.  The existing peak-finding approach picks the
wrong dark/bright mapping when the finder pattern is small relative to the
ROI — background pixels dominate the histogram and the finder pattern's
intensities get squashed.

## Core insight

Pixels near the estimated finder-pattern centre are more likely to belong to
the finder pattern.  Weight them more heavily in the normalization so that
the dark/bright mapping reflects the finder pattern's contrast, not the
background's.

## Algorithm

For each pixel `(x, y)` in the ROI, compute its distance `d` from the
estimated centre `center_xy`.  Apply a Gaussian weight:

    w(x, y) = exp( -½ · (d / σ)² )

where `σ = sigma_factor * 3.5 * m_est` — the finder pattern's outer boundary
is at ~3.5 module pitches.  Pixels at the boundary get weight `exp(-0.5) ≈
0.61`; background at 2σ gets `exp(-2) ≈ 0.14`.

Then compute **weighted p10 and p90** of all ROI pixel intensities:

1. Flatten pixel values and weights into 1-D arrays.
2. Sort pixels by intensity.
3. Compute cumulative weight.
4. Find the intensity where cumulative weight crosses 10% and 90% of total
   weight.

Map p10 → 0, p90 → 1:

    roi_norm = clip((roi - dark) / (bright - dark), 0, 1)

## Function signature

```python
def normalize_roi_intensities(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    sigma_factor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize ROI intensities to [0, 1] using centre-weighted percentiles.

    Returns
    -------
    roi_norm : ndarray (H, W)   Normalized intensities in [0, 1].
    dark : float                Weighted p10 (mapped to 0).
    bright : float              Weighted p90 (mapped to 1).
    """
```

Implementation:

```python
def normalize_roi_intensities(
    roi: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    sigma_factor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    H, W = roi.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs.astype(np.float64) - center_xy[0]) ** 2
                   + (ys.astype(np.float64) - center_xy[1]) ** 2)
    sigma = sigma_factor * 3.5 * m_est
    weights = np.exp(-0.5 * (dist / sigma) ** 2)

    vals = roi.ravel().astype(np.float64)
    w = weights.ravel()

    order = np.argsort(vals)
    vals_sorted = vals[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    def _weighted_percentile(percentile: float) -> float:
        if total_w == 0.0:
            return float(np.percentile(vals, percentile))
        target = percentile / 100.0 * total_w
        idx = int(np.searchsorted(cum_w, target))
        idx = max(0, min(idx, len(vals_sorted) - 1))
        return float(vals_sorted[idx])

    dark = _weighted_percentile(10.0)
    bright = _weighted_percentile(90.0)

    span = bright - dark
    if span < 1.0:
        span = 1.0
    roi_norm = np.clip((roi.astype(np.float64) - dark) / span, 0.0, 1.0)
    return roi_norm, dark, bright
```

## Call-site changes

All three call sites in `ray-profile.py` currently use `(roi)` and must
become `(roi, center_xy, m_est)`.  The variables `center_xy` and `m_est`
are already in scope at each site:

| Cell | Old | New |
|------|-----|-----|
| [6] line ~576 | `normalize_roi_intensities(roi)` | `normalize_roi_intensities(roi, center_xy, m_est)` |
| [7] line ~667 | `normalize_roi_intensities(roi)` | `normalize_roi_intensities(roi, center_xy, m_est)` |
| [8] line ~766 | `normalize_roi_intensities(roi)` | `normalize_roi_intensities(roi, center_xy, m_est)` |

## Cleanup

Remove the now-unused import in cell [5]:

```diff
- from scipy.signal import find_peaks
```

## Parameters

```python
DARK_PCT = 10.0             # Weighted percentile mapped to 0
BRIGHT_PCT = 90.0           # Weighted percentile mapped to 1
SIGMA_FACTOR = 1.0          # Multiplier on 3.5*m_est for Gaussian sigma
```
