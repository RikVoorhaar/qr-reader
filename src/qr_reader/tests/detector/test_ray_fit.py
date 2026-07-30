"""Tests for ray-profile-based finder pattern fitting (ray_fit.py)."""

import numpy as np
import pytest

from qr_reader.detector.ray_fit import (
    RayFitResult,
    finder_soft_template,
    fit_all_rays,
    fit_finder_ray,
    normalize_roi_intensities,
    sample_ray_profiles,
)

# Import only for _make_finder_roi (scipy.ndimage available here)
import scipy.ndimage as _ndi

# ---------------------------------------------------------------------------
# 1. sample_ray_profiles
# ---------------------------------------------------------------------------


class TestSampleRayProfiles:
    """Tests for sample_ray_profiles."""

    def test_output_shapes(self):
        """Verify output shapes with default parameters."""
        roi = np.ones((50, 50), dtype=np.uint8) * 128
        profiles, max_dist = sample_ray_profiles(roi, 25, 25, num_rays=16, num_samples=64)
        assert profiles.shape == (16, 64)
        assert max_dist > 0
        assert not np.any(np.isnan(profiles))

    def test_gradient_sampling(self):
        """Values along a horizontal ray match bilinear interpolation."""
        roi = np.zeros((50, 50), dtype=np.float64)
        for x in range(50):
            roi[:, x] = x * 5  # grayscale ramp in x-direction
        profiles, max_dist = sample_ray_profiles(roi.astype(np.uint8), 10, 20, num_rays=1, num_samples=10,
                                                   ray_length=1.0)
        diag = 0.5 * np.hypot(50, 50)
        t_vals = np.linspace(0, diag, 10)
        sx = 10 + t_vals
        sy = np.full(10, 20.0)
        x0 = np.clip(np.floor(sx).astype(int), 0, 49)
        x1 = np.clip(x0 + 1, 0, 49)
        fx = sx - x0.astype(np.float64)
        expected = np.asarray((1 - fx) * roi[20, x0] + fx * roi[20, x1], dtype=np.float64)
        expected = np.clip(expected, 0, 255)
        np.testing.assert_allclose(profiles[0], expected, atol=1.0)

    def test_center_value(self):
        """Sample at distance 0 equals the pixel value at centre."""
        roi = np.random.RandomState(42).randint(0, 256, (30, 30), dtype=np.uint8)
        profiles, _ = sample_ray_profiles(roi, 15, 15, num_rays=1, num_samples=1,
                                          ray_length=1.0)
        # Bilinear interpolation at centre should give the centre pixel value
        assert abs(profiles[0, 0] - roi[15, 15]) < 1.0

    def test_all_angles_covered(self):
        """num_rays directions span [0, 2π)."""
        roi = np.ones((30, 30), dtype=np.uint8) * 200
        num_rays = 36
        profiles, max_dist = sample_ray_profiles(roi, 15, 15, num_rays=num_rays, num_samples=5)
        assert profiles.shape[0] == num_rays
        # The max distance should be positive
        assert max_dist > 0


# ---------------------------------------------------------------------------
# 2. normalize_roi_intensities
# ---------------------------------------------------------------------------


class TestNormalizeRoiIntensities:
    """Tests for normalize_roi_intensities."""

    def test_two_tone_roi(self):
        """Dark centre, bright border → dark ~0, bright ~1."""
        H, W = 60, 60
        roi = np.full((H, W), 200, dtype=np.uint8)
        ys, xs = np.mgrid[0:H, 0:W]
        center_xy = np.array([30.0, 30.0])
        dist = np.sqrt((xs - 30) ** 2 + (ys - 30) ** 2)
        roi[dist < 10] = 50  # dark centre
        roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, center_xy, m_est=5.0)
        assert roi_norm.shape == (H, W)
        assert dark_val < 150
        assert bright_val > 100
        # Dark centre should be low, bright border should be high
        center_val = float(np.mean(roi_norm[dist < 10]))
        border_val = float(np.mean(roi_norm[dist > 20]))
        assert center_val < 0.3
        assert border_val > 0.7

    def test_output_range(self):
        """Normalized intensities are in [0, 1]."""
        rng = np.random.RandomState(7)
        roi = rng.randint(50, 200, (40, 40), dtype=np.uint8)
        roi_norm, _, _ = normalize_roi_intensities(roi, np.array([20.0, 20.0]), m_est=5.0)
        assert float(roi_norm.min()) >= 0.0
        assert float(roi_norm.max()) <= 1.0

    def test_low_contrast_fallback(self):
        """If span < 1, clip anyway (should not crash)."""
        roi = np.full((30, 30), 128, dtype=np.uint8)
        roi_norm, dark_val, bright_val = normalize_roi_intensities(roi, np.array([15.0, 15.0]), m_est=5.0)
        assert roi_norm.shape == (30, 30)
        assert abs(dark_val - 128) < 2
        assert abs(bright_val - 128) < 2


# ---------------------------------------------------------------------------
# 3. fit_all_rays
# ---------------------------------------------------------------------------


class TestFitAllRays:
    """Tests for fit_all_rays."""

    def test_known_m(self):
        """Fit profiles generated from finder_soft_template with a known m."""
        rng = np.random.RandomState(1)
        m_true = 10.0
        num_rays = 8
        num_samples = 100
        max_dist = 60.0
        t_samples = np.linspace(0, max_dist, num_samples)

        profiles = np.zeros((num_rays, num_samples), dtype=np.float64)
        for i in range(num_rays):
            template = finder_soft_template(t_samples, m_true, sigma=1.0)
            noise = rng.normal(0, 0.02, num_samples)
            profiles[i] = np.clip(template + noise, 0, 1)

        m_fitted, mse_arr, success = fit_all_rays(profiles, m_est=10.0, max_dist=max_dist)

        assert success.all()
        median_m = float(np.median(m_fitted))
        assert abs(median_m - m_true) / m_true < 0.05  # within 5 %

    def test_fewer_rays(self):
        """Works with just 1 ray."""
        num_samples = 60
        max_dist = 40.0
        t_samples = np.linspace(0, max_dist, num_samples)
        profile = finder_soft_template(t_samples, 8.0, sigma=1.0)[None, :]
        m_fitted, _mse, success = fit_all_rays(profile, m_est=8.0, max_dist=max_dist)
        assert success[0]
        assert abs(m_fitted[0] - 8.0) < 2.0

    def test_noisy_profile_still_succeeds(self):
        """Moderate noise does not break fitting."""
        rng = np.random.RandomState(3)
        num_samples = 80
        max_dist = 50.0
        t_samples = np.linspace(0, max_dist, num_samples)
        template = finder_soft_template(t_samples, 12.0, sigma=1.0)
        noise = rng.normal(0, 0.05, num_samples)
        profile = np.clip(template + noise, 0, 1)[None, :]
        m_fitted, _, success = fit_all_rays(profile, m_est=12.0, max_dist=max_dist)
        assert success[0]
        assert abs(m_fitted[0] - 12.0) / 12.0 < 0.15


# ---------------------------------------------------------------------------
# 4. Integration: synthetic finder pattern
# ---------------------------------------------------------------------------


def _make_finder_roi(
    size: int = 80,
    m: float = 10.0,
    center_xy: np.ndarray | tuple | None = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic ROI with a 7×7 finder pattern at the centre.

    The finder has a 3-module dark square, then a 1-module white ring, then
    a 1-module dark ring, then a white quiet-zone border using chebyshev
    (L∞) distance so edges are axis-aligned.

    Parameters
    ----------
    size : int
        Square ROI dimension in pixels.
    m : float
        Module pitch in pixels.
    center_xy : array-like or None
        Centre of the finder in (x, y).  Defaults to ``(size/2, size/2)``.
    noise_std : float
        Gaussian noise std added to the final image.
    seed : int
        RNG seed.

    Returns
    -------
    roi : ndarray (size, size) uint8
    corners_gt : ndarray (4, 2) float64
        Ground-truth outer corners (x, y) of the 7×7 finder.
    """
    rng = np.random.RandomState(seed)
    if center_xy is None:
        cx, cy = size / 2.0, size / 2.0
    else:
        cx, cy = float(np.asarray(center_xy).ravel()[0]), float(np.asarray(center_xy).ravel()[1])

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    # Chebyshev (L∞) distance from centre in module units
    u = np.maximum(np.abs(xs - cx), np.abs(ys - cy)) / m

    roi = np.full((size, size), 255.0, dtype=np.float64)
    dark_mask = (u <= 1.5) | ((u >= 2.5) & (u <= 3.5))
    roi[dark_mask] = 0.0

    # Gaussian blur to smooth edges
    from scipy.ndimage import gaussian_filter
    roi = gaussian_filter(roi, sigma=m * 0.15)

    if noise_std > 0:
        roi = np.clip(roi + rng.normal(0, noise_std, roi.shape), 0, 255)

    roi = np.clip(np.round(roi), 0, 255).astype(np.uint8)

    outer_r = 3.5 * m
    corners_gt = np.array([
        [cx - outer_r, cy - outer_r],
        [cx + outer_r, cy - outer_r],
        [cx + outer_r, cy + outer_r],
        [cx - outer_r, cy + outer_r],
    ], dtype=np.float64)

    return roi, corners_gt


class TestFitFinderRay:
    """Integration tests for fit_finder_ray."""

    def test_perfect_square(self):
        """Corners match the ground truth on a clean synthetic finder."""
        roi, corners_gt = _make_finder_roi(size=120, m=12.0)
        center = np.array([60.0, 60.0])
        result = fit_finder_ray(roi, center, m_est=12.0, sigma=0.5)
        assert result.valid
        assert result.score > 0.5
        errors = [float(np.linalg.norm(result.corners[i] - corners_gt[i])) for i in range(4)]
        mean_err = float(np.mean(errors))
        assert mean_err < 3.0, f"Mean corner error: {mean_err:.2f}px ({errors})"

    def test_noisy_square(self):
        """Still works with moderate Gaussian noise."""
        roi, corners_gt = _make_finder_roi(size=120, m=12.0, noise_std=3.0, seed=1)
        center = np.array([60.0, 60.0])
        result = fit_finder_ray(roi, center, m_est=12.0, sigma=1.0)
        assert result.valid
        errors = [float(np.linalg.norm(result.corners[i] - corners_gt[i])) for i in range(4)]
        mean_err = float(np.mean(errors))
        assert mean_err < 5.0, f"Mean corner error: {mean_err:.2f}px"

    def test_offset_center(self):
        """Centre estimate off by 2 px still yields valid corners."""
        roi, corners_gt = _make_finder_roi(size=120, m=12.0)
        center = np.array([62.0, 62.0])  # off by 2 px from true centre (60, 60)
        result = fit_finder_ray(roi, center, m_est=12.0)
        assert result.valid


# ---------------------------------------------------------------------------
# 5. Concentration filter
# ---------------------------------------------------------------------------


def _make_circular_boundary_points(
    center_xy: np.ndarray,
    radius: float,
    num_points: int = 36,
) -> np.ndarray:
    """Generate boundary points that form a circle (no straight edges)."""
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = np.column_stack([
        center_xy[0] + radius * np.cos(theta),
        center_xy[1] + radius * np.sin(theta),
    ])
    return pts.astype(np.float64)


class TestConcentrationFilter:
    """Tests for the false-positive filter in fit_finder_ray."""

    def test_circle_rejected(self):
        """Circular boundary points should be rejected (no 4 straight edges)."""
        # Build an ROI with a circular intensity pattern so that
        # per-ray m estimates produce boundary points on a circle.
        size = 100
        m = 10.0
        roi = np.full((size, size), 200, dtype=np.uint8)
        ys, xs = np.mgrid[0:size, 0:size]
        center_xy = np.array([50.0, 50.0])
        dist = np.sqrt((xs - 50) ** 2 + (ys - 50) ** 2)
        # Make a single dark ring at |u| ≈ 1.5—3.5 (like a thick circle)
        u = dist / m
        roi[(u >= 1.2) & (u <= 3.8)] = 30
        result = fit_finder_ray(roi, center_xy, m_est=m, min_concentration_ratio=0.5)
        # A circle doesn't have 4 straight edges → concentration check
        # should either fail (valid=False) or produce very low score.
        if result.valid:
            assert result.score < 0.5, f"Circle scored too high: {result.score:.3f}"


# ---------------------------------------------------------------------------
# 6. finder_soft_template
# ---------------------------------------------------------------------------


class TestFinderSoftTemplate:
    """Tests for finder_soft_template."""

    def test_output_range(self):
        """Template values are in [0, 1]."""
        t = np.linspace(0, 50, 100)
        tmpl = finder_soft_template(t, m=10.0, sigma=1.0)
        assert float(tmpl.min()) >= 0.0
        assert float(tmpl.max()) <= 1.0

    def test_centre_is_dark(self):
        """At t=0, the template should be dark (< 0.5)."""
        val = float(finder_soft_template(np.array([0.0]), m=10.0, sigma=1.0)[0])
        assert val < 0.5

    def test_beyond_outer_is_bright(self):
        """Well past 3.5m, the template should be bright (> 0.5)."""
        val = float(finder_soft_template(np.array([40.0]), m=10.0, sigma=1.0)[0])
        assert val > 0.5

    def test_three_edges(self):
        """Template has 3 edges: at ~1.5m, ~2.5m, ~3.5m."""
        t = np.linspace(0, 50, 500)
        tmpl = finder_soft_template(t, m=10.0, sigma=0.3)
        grad = np.abs(np.diff(tmpl))
        # Expect 3 peaks in gradient magnitude
        peaks = (grad[1:-1] > grad[:-2]) & (grad[1:-1] > grad[2:])
        n_peaks = int(np.sum(peaks & (grad[1:-1] > 0.01)))
        assert n_peaks >= 3, f"Expected >= 3 gradient peaks, got {n_peaks}"
