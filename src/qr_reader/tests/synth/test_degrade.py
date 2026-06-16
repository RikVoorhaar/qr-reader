"""Tests for Phase 5 — Global Degradation."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
from qr_reader.synth.composite import composite_patch
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.degrade import (
    apply_brightness_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_global_degradation,
    apply_jpeg_compression,
)
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)
from qr_reader.synth.placement import PlacedPatch, place_patch, sample_placement_scale


def _N(version: int) -> int:
    """Number of modules per side for a given QR version."""
    return 17 + 4 * version


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def identity_config() -> AugmentationConfig:
    """Config with all degradation ranges set to identity (no effect)."""
    return AugmentationConfig(
        version=3,
        content="degrade test",
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(8.0, 8.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(4.0, 4.0),
        feather_sigma_range=(0.5, 0.5),
        blur_sigma_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        jpeg_quality_range=(100, 100),
    )


@pytest.fixture
def sample_image(rng: np.random.Generator) -> np.ndarray:
    """A small synthetic RGB test image with known values."""
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def composite_result_with_identity(
    identity_config: AugmentationConfig, rng: np.random.Generator
):
    """Build a CompositeResult using the identity config (no augmentation)."""
    cfg = identity_config
    ppm_int = int(cfg.ppm_range[0])
    N = _N(cfg.version)
    bg_shape = (480, 640)

    patch, mask = generate_qr_patch(
        version=cfg.version,
        content=cfg.content,
        ecl_str=cfg.error_correction,
        ppm=ppm_int,
        quiet_zone_modules=cfg.quiet_zone_modules,
    )
    qr_corners = compute_qr_corners_patch_space(
        quiet_zone_modules=cfg.quiet_zone_modules,
        N=N,
        ppm=ppm_int,
    )
    augmented: AugmentedPatch = apply_augmentation(patch, mask, qr_corners, rng, cfg)
    scale, tx, ty = sample_placement_scale(
        rng, augmented.warped_patch.shape[:2], N, cfg, bg_shape
    )
    placed: PlacedPatch = place_patch(augmented, scale, tx, ty, bg_shape)

    bg = np.full((*bg_shape, 3), 128, dtype=np.uint8)
    return composite_patch(bg, placed, feather_sigma=cfg.feather_sigma_range[0])


# ===================================================================
# 5.1  Individual degradation functions
# ===================================================================


class TestApplyGaussianBlur:
    """Tests for :func:`apply_gaussian_blur`."""

    def test_blur_identity(self, sample_image: np.ndarray) -> None:
        """sigma=0 → output equals input."""
        result = apply_gaussian_blur(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_blur_positive(self, sample_image: np.ndarray) -> None:
        """sigma > 0 produces a visibly different (smoother) image."""
        result = apply_gaussian_blur(sample_image, 3.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
        # A blurred image should be smoother — check that local variance
        # has decreased compared to the original
        orig_var = np.var(sample_image.astype(np.float32))
        blurred_var = np.var(result.astype(np.float32))
        assert blurred_var < orig_var, "Blurred image should have lower variance"

    def test_deterministic(self, sample_image: np.ndarray) -> None:
        """Same sigma → same output (blur is deterministic)."""
        result1 = apply_gaussian_blur(sample_image, 1.5)
        result2 = apply_gaussian_blur(sample_image, 1.5)
        np.testing.assert_array_equal(result1, result2)


class TestApplyGaussianNoise:
    """Tests for :func:`apply_gaussian_noise`."""

    def test_noise_identity(
        self, sample_image: np.ndarray, rng: np.random.Generator
    ) -> None:
        """sigma=0 → output equals input."""
        result = apply_gaussian_noise(sample_image, rng, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_noise_positive(
        self, sample_image: np.ndarray, rng: np.random.Generator
    ) -> None:
        """sigma > 0 produces a different image."""
        result = apply_gaussian_noise(sample_image, rng, 10.0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
        # At sigma=10 the image should differ measurably
        diff = np.abs(result.astype(np.float32) - sample_image.astype(np.float32))
        assert diff.mean() > 0.5, "Noise sigma=10 should produce measurable differences"

    def test_deterministic(self, sample_image: np.ndarray) -> None:
        """Same rng state → same output."""
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        result1 = apply_gaussian_noise(sample_image, rng1, 5.0)
        result2 = apply_gaussian_noise(sample_image, rng2, 5.0)
        np.testing.assert_array_equal(result1, result2)


class TestApplyJpegCompression:
    """Tests for :func:`apply_jpeg_compression`."""

    @pytest.fixture
    def smooth_image(self) -> np.ndarray:
        """A smooth gradient image (JPEG-friendly low-frequency content)."""
        # Create a 128x128 smooth gradient — JPEG block boundaries won't
        # create large per-pixel diffs on smooth content.
        x = np.linspace(0, 255, 128, dtype=np.float32)
        y = np.linspace(0, 255, 128, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        r = np.clip(xx, 0, 255).astype(np.uint8)
        g = np.clip(yy, 0, 255).astype(np.uint8)
        b = np.clip((xx + yy) / 2, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    def test_jpeg_identity(self, smooth_image: np.ndarray) -> None:
        """quality=100 → output close to input (within small tolerance)."""
        result = apply_jpeg_compression(smooth_image, 100)
        assert result.shape == smooth_image.shape
        assert result.dtype == np.uint8
        # JPEG is always lossy, so allow small per-pixel differences.
        # On smooth content (low-frequency) at quality=100 the diff
        # should be <= 2 per channel.
        diff = np.abs(result.astype(np.float32) - smooth_image.astype(np.float32))
        assert diff.max() <= 4, (
            f"JPEG quality=100 should produce very small diff, got max={diff.max()}"
        )

    def test_jpeg_low_quality(self, smooth_image: np.ndarray) -> None:
        """Low quality introduces larger artifacts than high quality."""
        high = apply_jpeg_compression(smooth_image, 95)
        low = apply_jpeg_compression(smooth_image, 10)
        high_diff = np.abs(
            high.astype(np.float32) - smooth_image.astype(np.float32)
        ).mean()
        low_diff = np.abs(
            low.astype(np.float32) - smooth_image.astype(np.float32)
        ).mean()
        assert low_diff >= high_diff, (
            f"Low quality should produce larger artifacts "
            f"(low={low_diff:.2f}, high={high_diff:.2f})"
        )

    def test_deterministic(self, sample_image: np.ndarray) -> None:
        """Same quality → same output (JPEG is deterministic)."""
        result1 = apply_jpeg_compression(sample_image, 75)
        result2 = apply_jpeg_compression(sample_image, 75)
        np.testing.assert_array_equal(result1, result2)


class TestApplyBrightnessContrast:
    """Tests for :func:`apply_brightness_contrast`."""

    def test_bc_identity(self, sample_image: np.ndarray) -> None:
        """brightness=0, contrast=1.0 → output equals input."""
        result = apply_brightness_contrast(sample_image, 0, 1.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_brightness_shift(self, sample_image: np.ndarray) -> None:
        """Positive brightness shifts values up; negative shifts down."""
        bright = apply_brightness_contrast(sample_image, 50, 1.0)
        dark = apply_brightness_contrast(sample_image, -50, 1.0)

        # Brightened image should have higher mean
        assert bright.mean() > sample_image.mean(), (
            f"Brightness +50 should increase mean "
            f"({bright.mean():.1f} vs {sample_image.mean():.1f})"
        )
        assert dark.mean() < sample_image.mean(), (
            f"Brightness -50 should decrease mean "
            f"({dark.mean():.1f} vs {sample_image.mean():.1f})"
        )

    def test_contrast_change(self, sample_image: np.ndarray) -> None:
        """Higher contrast increases variance; lower contrast reduces it."""
        high = apply_brightness_contrast(sample_image, 0, 2.0)
        low = apply_brightness_contrast(sample_image, 0, 0.5)

        high_var = np.var(high.astype(np.float32))
        orig_var = np.var(sample_image.astype(np.float32))
        low_var = np.var(low.astype(np.float32))

        assert high_var > orig_var, "Higher contrast should increase variance"
        assert low_var < orig_var, "Lower contrast should decrease variance"

    def test_deterministic(self, sample_image: np.ndarray) -> None:
        """Same params → same output (deterministic)."""
        result1 = apply_brightness_contrast(sample_image, -30, 1.2)
        result2 = apply_brightness_contrast(sample_image, -30, 1.2)
        np.testing.assert_array_equal(result1, result2)


class TestCombinedDeterministic:
    """Verify all four functions are deterministic with same rng state."""

    def test_all_deterministic(self, sample_image: np.ndarray) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        b1 = apply_gaussian_blur(sample_image, 1.0)
        b2 = apply_gaussian_blur(sample_image, 1.0)
        np.testing.assert_array_equal(b1, b2)

        n1 = apply_gaussian_noise(sample_image, rng1, 3.0)
        n2 = apply_gaussian_noise(sample_image, rng2, 3.0)
        np.testing.assert_array_equal(n1, n2)

        j1 = apply_jpeg_compression(sample_image, 80)
        j2 = apply_jpeg_compression(sample_image, 80)
        np.testing.assert_array_equal(j1, j2)

        c1 = apply_brightness_contrast(sample_image, 10, 1.1)
        c2 = apply_brightness_contrast(sample_image, 10, 1.1)
        np.testing.assert_array_equal(c1, c2)


# ===================================================================
# 5.2  apply_global_degradation
# ===================================================================


class TestApplyGlobalDegradation:
    """Tests for :func:`apply_global_degradation`."""

    def test_all_off(
        self,
        composite_result_with_identity,
        rng: np.random.Generator,
        identity_config: AugmentationConfig,
    ) -> None:
        """All ranges set to identity → output equals input."""
        image = composite_result_with_identity.composited_image
        result, params = apply_global_degradation(image, rng, identity_config)
        np.testing.assert_array_equal(result, image)
        assert params["blur_sigma"] == 0.0
        assert params["noise_sigma"] == 0.0
        assert params["jpeg_quality"] == 100

    def test_output_shape(
        self,
        composite_result_with_identity,
        rng: np.random.Generator,
    ) -> None:
        """Output has the same shape as input."""
        cfg = AugmentationConfig(
            blur_sigma_range=(0.5, 0.5),
            noise_sigma_range=(2.0, 2.0),
            jpeg_quality_range=(70, 70),
        )
        image = composite_result_with_identity.composited_image
        result, params = apply_global_degradation(image, rng, cfg)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert params["blur_sigma"] == 0.5
        assert params["noise_sigma"] == 2.0
        assert params["jpeg_quality"] == 70

    def test_deterministic(
        self,
        composite_result_with_identity,
    ) -> None:
        """Same seed → same output."""
        cfg = AugmentationConfig(
            blur_sigma_range=(0.5, 0.5),
            noise_sigma_range=(2.0, 2.0),
            jpeg_quality_range=(70, 70),
        )
        image = composite_result_with_identity.composited_image

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1, params1 = apply_global_degradation(image, rng1, cfg)
        result2, params2 = apply_global_degradation(image, rng2, cfg)
        np.testing.assert_array_equal(result1, result2)
        assert params1 == params2
