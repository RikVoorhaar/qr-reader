"""Tests for Phase 2 — Perspective Augmentation (isolated patch)."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.synth.augment import (
    AugmentedPatch,
    apply_augmentation,
    jitter_corners,
    perspective_warp,
    sample_patch_ppm,
)
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.patch import (
    compute_qr_corners_patch_space,
    generate_qr_patch,
)


def _N(version: int) -> int:
    """Number of modules per side for a given QR version."""
    return 17 + 4 * version


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def simple_config() -> AugmentationConfig:
    """A config with deterministic-friendly ranges for testing."""
    return AugmentationConfig(
        version=3,
        content="hello test",
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(5.0, 5.0),  # fixed
        rotation_deg_range=(0.0, 0.0),  # fixed (no rotation by default)
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),  # fixed
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def patch_and_corners(
    simple_config: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clean QR patch and its QR-code corners."""
    cfg = simple_config
    ppm_int = int(cfg.ppm_range[0])
    N = _N(cfg.version)
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
    return patch, mask, qr_corners


# ===================================================================
# 2.1  sample_patch_ppm
# ===================================================================


class TestSamplePatchPpm:
    """Tests for :func:`sample_patch_ppm`."""

    def test_ppm_in_range(self, rng):
        """Sampled value is within config.ppm_range."""
        config = AugmentationConfig(ppm_range=(3.0, 20.0))
        for _ in range(50):
            ppm = sample_patch_ppm(rng, config)
            assert 3.0 <= ppm <= 20.0, f"ppm {ppm} outside range"

    def test_ppm_deterministic(self):
        """Same rng state → same value."""
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        config = AugmentationConfig(ppm_range=(4.0, 18.0))
        v1 = sample_patch_ppm(rng1, config)
        v2 = sample_patch_ppm(rng2, config)
        assert v1 == v2


# ===================================================================
# 2.2  jitter_corners
# ===================================================================


class TestJitterCorners:
    """Tests for :func:`jitter_corners`."""

    @pytest.fixture
    def rect_corners(self) -> np.ndarray:
        """A 200×100 rectangle in TL, TR, BR, BL order."""
        return np.array(
            [
                [10.0, 20.0],  # TL
                [210.0, 20.0],  # TR
                [210.0, 120.0],  # BR
                [10.0, 120.0],  # BL
            ],
            dtype=np.float64,
        )

    def test_jitter_zero(self, rect_corners, rng):
        """jitter_fraction=0 returns input corners unchanged."""
        result = jitter_corners(rect_corners, rng, 0.0)
        np.testing.assert_array_equal(result, rect_corners)

    def test_jitter_range(self, rect_corners, rng):
        """All jittered points are within ±jitter_fraction * side of input."""
        jf = 0.1
        result = jitter_corners(rect_corners, rng, jf)

        # Side for this rect: width 200, height 100, avg = 150
        # max_offset = 0.1 * 150 = 15
        max_offset = jf * 150.0

        for orig, jit in zip(rect_corners, result):
            dx = abs(jit[0] - orig[0])
            dy = abs(jit[1] - orig[1])
            assert dx <= max_offset + 1e-9, f"x offset {dx} exceeds {max_offset}"
            assert dy <= max_offset + 1e-9, f"y offset {dy} exceeds {max_offset}"

    def test_deterministic(self, rect_corners):
        """Same rng → same output."""
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        res1 = jitter_corners(rect_corners, rng1, 0.2)
        res2 = jitter_corners(rect_corners, rng2, 0.2)
        np.testing.assert_array_equal(res1, res2)

    def test_corner_count(self, rect_corners, rng):
        """Returns 4 points."""
        result = jitter_corners(rect_corners, rng, 0.1)
        assert result.shape == (4, 2), f"Expected (4, 2), got {result.shape}"


# ===================================================================
# 2.3  perspective_warp
# ===================================================================


class TestPerspectiveWarp:
    """Tests for :func:`perspective_warp`."""

    @pytest.fixture
    def checker(self) -> tuple[np.ndarray, np.ndarray]:
        """A small checkerboard image (100×100) and an all-ones mask."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill top-left quadrant white so we can detect translation
        img[:50, :50, :] = 255
        mask = np.ones((100, 100), dtype=np.float32)
        return img, mask

    def test_identity_warp(self, checker):
        """dst = src produces output identical to input."""
        img, mask = checker
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        dst = src.copy()

        warped_img, warped_mask = perspective_warp(img, mask, src, dst, (100, 100))

        np.testing.assert_array_equal(warped_img, img)
        np.testing.assert_array_equal(warped_mask, mask)

    def test_mask_range(self, checker):
        """Warped mask values are all in [0, 1]."""
        img, mask = checker
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        dst = np.array([[5, 5], [95, 5], [95, 95], [5, 95]], dtype=np.float64)

        _, warped_mask = perspective_warp(img, mask, src, dst, (100, 100))

        assert warped_mask.min() >= 0.0, f"mask min {warped_mask.min()}"
        assert warped_mask.max() <= 1.0, f"mask max {warped_mask.max()}"

    def test_output_shape(self, checker):
        """Output matches output_size."""
        img, mask = checker
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)

        out_w, out_h = 80, 60
        warped_img, warped_mask = perspective_warp(img, mask, src, dst, (out_w, out_h))

        assert warped_img.shape == (out_h, out_w, 3), (
            f"Expected ({out_h}, {out_w}, 3), got {warped_img.shape}"
        )
        assert warped_mask.shape == (out_h, out_w), (
            f"Expected ({out_h}, {out_w}), got {warped_mask.shape}"
        )

    def test_translation(self, checker):
        """Known offset produces expected shift."""
        img, mask = checker
        tx, ty = 20, 10
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
        dst = np.array(
            [[tx, ty], [100 + tx, ty], [100 + tx, 100 + ty], [tx, 100 + ty]],
            dtype=np.float64,
        )

        warped_img, _ = perspective_warp(img, mask, src, dst, (150, 150))

        # The white top-left quadrant of the source should now start at (tx, ty)
        # Check a pixel that was white before (25, 25) → should be at (25+tx, 25+ty)
        assert warped_img[ty + 25, tx + 25].tolist() == [255, 255, 255], (
            "Translated white pixel not at expected location"
        )
        # A pixel that was black before (75, 25) → should stay black at (75+tx, 25+ty)
        assert warped_img[ty + 25, tx + 75].tolist() == [0, 0, 0], (
            "Translated black pixel not at expected location"
        )
        # The area outside the warped image should be black (BORDER_CONSTANT)
        assert warped_img[0, 0].tolist() == [0, 0, 0], "Border should be black"


# ===================================================================
# 2.4  apply_augmentation
# ===================================================================


class TestApplyAugmentation:
    """Tests for :func:`apply_augmentation`."""

    def test_no_rotation(self, rng, simple_config, patch_and_corners):
        """Rotation=0, jitter=0, aspect=1 preserves QR corners as a square."""
        patch, mask, qr_corners = patch_and_corners
        # Config already has rotation=(0,0), jitter=0, aspect=(1,1)
        result = apply_augmentation(patch, mask, qr_corners, rng, simple_config)

        assert isinstance(result, AugmentedPatch)
        warped_corners = result.warped_corners_qr
        assert warped_corners.shape == (4, 2)

        # Check all four sides are equal (within tolerance)
        def dist(i, j):
            return np.linalg.norm(warped_corners[i] - warped_corners[j])

        s01 = dist(0, 1)  # TL→TR
        s12 = dist(1, 2)  # TR→BR
        s23 = dist(2, 3)  # BR→BL
        s30 = dist(3, 0)  # BL→TL

        assert abs(s01 - s12) < 1.0, f"top {s01} != right {s12}"
        assert abs(s12 - s23) < 1.0, f"right {s12} != bottom {s23}"
        assert abs(s23 - s30) < 1.0, f"bottom {s23} != left {s30}"
        assert abs(s30 - s01) < 1.0, f"left {s30} != top {s01}"

        # Check the corners are approximately right-angled (diagonal approx side*sqrt(2))
        diag = dist(0, 2)
        side = (s01 + s12 + s23 + s30) / 4.0
        assert abs(diag - side * np.sqrt(2)) < 2.0, (
            f"Diagonal {diag} != side*sqrt(2) = {side * np.sqrt(2)}"
        )

    def test_warped_qr_corners_visible(self, rng):
        """All 4 warped QR corners are within the output bounds."""
        # Use a config with rotation spanning 90 deg so there is non-trivial warp
        config = AugmentationConfig(
            version=3,
            content="visible test",
            rotation_deg_range=(30.0, 60.0),
            jitter_fraction=0.1,
            aspect_scale_range=(0.9, 1.1),
            ppm_range=(8.0, 8.0),
        )
        ppm_int = int(config.ppm_range[0])
        N = _N(config.version)
        patch, mask = generate_qr_patch(
            version=config.version,
            content=config.content,
            ecl_str=config.error_correction,
            ppm=ppm_int,
            quiet_zone_modules=config.quiet_zone_modules,
        )
        qr_corners = compute_qr_corners_patch_space(
            quiet_zone_modules=config.quiet_zone_modules,
            N=N,
            ppm=ppm_int,
        )

        result = apply_augmentation(patch, mask, qr_corners, rng, config)

        H, W = result.warped_patch.shape[:2]
        for pt in result.warped_corners_qr:
            assert 0 <= pt[0] <= W, f"x={pt[0]} out of bounds [0, {W}]"
            assert 0 <= pt[1] <= H, f"y={pt[1]} out of bounds [0, {H}]"

    def test_warped_qr_corners_vs_modules(self, rng):
        """The warped QR corners still correspond to the QR code proper.

        For a small rotation the warped corners should — when we extract the
        warped region — still show a mix of black and white pixels, confirming
        the corners are on the QR code boundary.
        """
        config = AugmentationConfig(
            version=3,
            content="QR Reader v1",
            rotation_deg_range=(0.0, 15.0),  # mild rotation only
            jitter_fraction=0.0,
            aspect_scale_range=(1.0, 1.0),
            ppm_range=(10.0, 10.0),
        )
        ppm_int = int(config.ppm_range[0])
        N = _N(config.version)
        patch, mask = generate_qr_patch(
            version=config.version,
            content=config.content,
            ecl_str=config.error_correction,
            ppm=ppm_int,
            quiet_zone_modules=config.quiet_zone_modules,
        )
        qr_corners = compute_qr_corners_patch_space(
            quiet_zone_modules=config.quiet_zone_modules,
            N=N,
            ppm=ppm_int,
        )

        result = apply_augmentation(patch, mask, qr_corners, rng, config)

        # Extract the bounding box of the warped QR corners from the warped image
        wc = result.warped_corners_qr
        x_min, y_min = wc.min(axis=0).astype(int)
        x_max, y_max = wc.max(axis=0).astype(int)

        region = result.warped_patch[y_min:y_max, x_min:x_max, :]

        # The QR region should contain both black (0) and white (255) pixels
        assert region.size > 0, "Extracted QR region is empty"
        assert np.any(region == 0), "QR region has no black pixels"
        assert np.any(region == 255), "QR region has no white pixels"

    def test_deterministic(self):
        """Same seed → identical output."""
        config = AugmentationConfig(
            version=3,
            content="det test",
            ppm_range=(7.0, 7.0),
            rotation_deg_range=(10.0, 80.0),
            jitter_fraction=0.08,
            aspect_scale_range=(0.85, 1.15),
        )
        ppm_int = int(config.ppm_range[0])
        N = _N(config.version)
        patch, mask = generate_qr_patch(
            version=config.version,
            content=config.content,
            ecl_str=config.error_correction,
            ppm=ppm_int,
            quiet_zone_modules=config.quiet_zone_modules,
        )
        qr_corners = compute_qr_corners_patch_space(
            quiet_zone_modules=config.quiet_zone_modules,
            N=N,
            ppm=ppm_int,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = apply_augmentation(patch, mask, qr_corners, rng1, config)
        result2 = apply_augmentation(patch, mask, qr_corners, rng2, config)

        np.testing.assert_array_equal(result1.warped_patch, result2.warped_patch)
        np.testing.assert_array_equal(result1.warped_mask, result2.warped_mask)
        np.testing.assert_array_equal(
            result1.warped_corners_qr, result2.warped_corners_qr
        )
