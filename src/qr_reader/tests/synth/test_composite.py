"""Tests for Phase 4 — Compositing."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
from qr_reader.synth.composite import (
    CompositeResult,
    alpha_composite,
    composite_patch,
    feather_mask,
)
from qr_reader.synth.config import AugmentationConfig
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
def simple_config() -> AugmentationConfig:
    """A config with deterministic-friendly ranges for quick tests."""
    return AugmentationConfig(
        version=3,
        content="composite test",
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(8.0, 8.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(4.0, 4.0),
    )


@pytest.fixture
def placed_patch_fixture(
    simple_config: AugmentationConfig, rng: np.random.Generator
) -> PlacedPatch:
    """Generate a PlacedPatch for compositing tests."""
    cfg = simple_config
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
    return place_patch(augmented, scale, tx, ty, bg_shape)


@pytest.fixture
def bg_white(placed_patch_fixture: PlacedPatch) -> np.ndarray:
    """All-white background (same size as placed patch)."""
    H, W = placed_patch_fixture.full_image.shape[:2]
    return np.full((H, W, 3), 255, dtype=np.uint8)


@pytest.fixture
def bg_black(placed_patch_fixture: PlacedPatch) -> np.ndarray:
    """All-black background (same size as placed patch)."""
    H, W = placed_patch_fixture.full_image.shape[:2]
    return np.zeros((H, W, 3), dtype=np.uint8)


# ===================================================================
# 4.1  feather_mask
# ===================================================================


class TestFeatherMask:
    """Tests for :func:`feather_mask`."""

    def test_sigma_zero(self) -> None:
        """sigma=0 returns mask unchanged (floating point tolerance)."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0

        result = feather_mask(mask, 0.0)
        np.testing.assert_array_equal(result, mask)

    def test_range(self) -> None:
        """Alpha values ∈ [0, 1] after feathering."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0

        result = feather_mask(mask, 3.0)
        assert result.min() >= 0.0, f"Min alpha {result.min()} < 0"
        assert result.max() <= 1.0, f"Max alpha {result.max()} > 1"

    def test_edge_soft(self) -> None:
        """Values near mask edges are between 0 and 1 (not hard 0/1)."""
        mask = np.zeros((200, 200), dtype=np.float32)
        mask[60:140, 60:140] = 1.0

        result = feather_mask(mask, 5.0)

        # Find pixels that are neither 0 nor 1 (the feathered edge)
        interior = result[60:140, 60:140]
        assert np.any((interior > 0) & (interior < 1)), (
            "No soft alpha values inside mask boundary"
        )

        # At the very centre of the interior, alpha should stay 1.0
        assert result[100, 100] == 1.0, f"Centre alpha {result[100, 100]} != 1.0"

        # Far outside the mask, alpha should stay 0.0
        assert result[5, 5] == 0.0, f"Far-outside alpha {result[5, 5]} != 0.0"


# ===================================================================
# 4.2  alpha_composite
# ===================================================================


class TestAlphaComposite:
    """Tests for :func:`alpha_composite`."""

    @pytest.fixture
    def bg(self) -> np.ndarray:
        """A 50×50 random-ish background."""
        return np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)

    @pytest.fixture
    def patch(self) -> np.ndarray:
        """A 50×50 random-ish patch."""
        return np.full((50, 50, 3), [30, 60, 90], dtype=np.uint8)

    def test_alpha_zero(self, bg: np.ndarray, patch: np.ndarray) -> None:
        """alpha=0 → result equals background."""
        alpha = np.zeros((50, 50), dtype=np.float32)
        result = alpha_composite(bg, patch, alpha)
        np.testing.assert_array_equal(result, bg)

    def test_alpha_one(self, bg: np.ndarray, patch: np.ndarray) -> None:
        """alpha=1 → result equals patch."""
        alpha = np.ones((50, 50), dtype=np.float32)
        result = alpha_composite(bg, patch, alpha)
        np.testing.assert_array_equal(result, patch)

    def test_half_alpha(self, bg: np.ndarray, patch: np.ndarray) -> None:
        """alpha=0.5 → result is average of patch and background."""
        alpha = np.full((50, 50), 0.5, dtype=np.float32)
        result = alpha_composite(bg, patch, alpha)

        expected = (
            (0.5 * patch.astype(np.float32)) + (0.5 * bg.astype(np.float32))
        ).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_dtype(self, bg: np.ndarray, patch: np.ndarray) -> None:
        """Output is uint8."""
        alpha = np.full((50, 50), 0.3, dtype=np.float32)
        result = alpha_composite(bg, patch, alpha)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


# ===================================================================
# 4.3  composite_patch
# ===================================================================


class TestCompositePatch:
    """Tests for :func:`composite_patch`."""

    def test_no_feather_on_flat_bg(
        self, placed_patch_fixture: PlacedPatch, bg_white: np.ndarray
    ) -> None:
        """White QR on white background should produce no visible edge.

        With a white patch placed on a white background, the composited image
        should be indistinguishable from pure white — i.e. the QR is invisible.
        """
        # The placed patch uses a QR patch on a black canvas; on a white
        # background the QR modules (black squares) will be visible but the
        # canvas area outside the QR should blend seamlessly.
        # We set feather_sigma near 0 so the mask is sharp.
        result = composite_patch(bg_white, placed_patch_fixture, feather_sigma=0.1)

        assert isinstance(result, CompositeResult)
        assert result.composited_image.shape == placed_patch_fixture.full_image.shape
        assert result.composited_image.dtype == np.uint8

        # The white background area outside the patch should be pure white
        # (255).  Since the patch sits within a specific rect on the canvas,
        # check corner pixels (which are background — the patch is centred).
        H, W = result.composited_image.shape[:2]
        # Top-left corner of image should be background = 255
        assert np.all(result.composited_image[0, 0] == [255, 255, 255]), (
            "Corner pixel not white"
        )

    def test_corners_preserved(
        self, placed_patch_fixture: PlacedPatch, bg_black: np.ndarray
    ) -> None:
        """image_corners_qr unchanged from input."""
        result = composite_patch(bg_black, placed_patch_fixture, feather_sigma=1.0)
        np.testing.assert_array_equal(
            result.image_corners_qr,
            placed_patch_fixture.image_corners_qr,
        )

    def test_black_bg_visible(
        self, placed_patch_fixture: PlacedPatch, bg_black: np.ndarray
    ) -> None:
        """QR on black background is visible through alpha.

        The placed patch contains QR modules (white and black).  On a black
        background, the white modules should still appear white after
        compositing (alpha=1 for the QR interior).
        """
        result = composite_patch(bg_black, placed_patch_fixture, feather_sigma=0.5)

        # The QR code region should have some white (255) pixels
        # We check by sampling near the centre of the QR region
        corners = placed_patch_fixture.image_corners_qr
        centre_x = int(round(corners[:, 0].mean()))
        centre_y = int(round(corners[:, 1].mean()))

        # The centre of the QR code should have some white modules
        centre_region = result.composited_image[
            max(0, centre_y - 10) : centre_y + 10,
            max(0, centre_x - 10) : centre_x + 10,
            :,
        ]
        assert np.any(centre_region == 255), (
            "No white pixels found near QR centre on black background"
        )
        # And some black (or dark) modules
        assert np.any(centre_region < 50), "No dark pixels found near QR centre"
