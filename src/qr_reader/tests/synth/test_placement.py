"""Tests for Phase 3 — Placement & Scale."""

from __future__ import annotations

import numpy as np
import pytest

from qr_reader.synth.augment import AugmentedPatch, apply_augmentation
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
        content="placement test",
        error_correction="M",
        quiet_zone_modules=4,
        ppm_range=(8.0, 8.0),  # fixed
        rotation_deg_range=(0.0, 0.0),  # no rotation
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(4.0, 4.0),  # fixed target ppm
    )


@pytest.fixture
def augmented_patch_fixture(
    simple_config: AugmentationConfig, rng: np.random.Generator
) -> AugmentedPatch:
    """Generate an AugmentedPatch for use in place_patch tests."""
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
    return apply_augmentation(patch, mask, qr_corners, rng, cfg)


# ===================================================================
# 3.1  sample_placement_scale
# ===================================================================


class TestSamplePlacementScale:
    """Tests for :func:`sample_placement_scale`."""

    def test_scale_positive(
        self, rng: np.random.Generator, simple_config: AugmentationConfig
    ) -> None:
        """Scale > 0 for any reasonable input."""
        N = _N(simple_config.version)
        # Use a variety of background sizes
        for bg_shape in [(1920, 1280), (1280, 1920), (1920, 1440), (640, 480)]:
            for _ in range(10):
                scale, tx, ty = sample_placement_scale(
                    rng, (400, 300), N, simple_config, bg_shape
                )
                assert scale > 0, f"Scale {scale} <= 0 for bg {bg_shape}"

    def test_translation_in_bounds(
        self, rng: np.random.Generator, simple_config: AugmentationConfig
    ) -> None:
        """tx + scaled_width <= bg_width and ty + scaled_height <= bg_height."""
        N = _N(simple_config.version)
        for bg_shape in [(1920, 1280), (1280, 1920), (1920, 1440)]:
            for _ in range(20):
                scale, tx, ty = sample_placement_scale(
                    rng, (350, 280), N, simple_config, bg_shape
                )
                bg_H, bg_W = bg_shape
                # shape tuple is (H, W) → index 1 is width, index 0 is height
                patch_H, patch_W = 350, 280
                scaled_W = patch_W * scale
                scaled_H = patch_H * scale
                assert tx + scaled_W <= bg_W + 1e-6, (
                    f"tx {tx} + scaled_W {scaled_W} > bg_W {bg_W}"
                )
                assert ty + scaled_H <= bg_H + 1e-6, (
                    f"ty {ty} + scaled_H {scaled_H} > bg_H {bg_H}"
                )
                assert tx >= 0, f"tx {tx} < 0"
                assert ty >= 0, f"ty {ty} < 0"

    def test_deterministic(self, simple_config: AugmentationConfig) -> None:
        """Same rng -> same output."""
        N = _N(simple_config.version)
        bg_shape = (1080, 1920)
        rng1 = np.random.default_rng(777)
        rng2 = np.random.default_rng(777)

        s1, tx1, ty1 = sample_placement_scale(
            rng1, (300, 250), N, simple_config, bg_shape
        )
        s2, tx2, ty2 = sample_placement_scale(
            rng2, (300, 250), N, simple_config, bg_shape
        )

        assert s1 == s2
        assert tx1 == tx2
        assert ty1 == ty2


# ===================================================================
# 3.2  place_patch
# ===================================================================


class TestPlacePatch:
    """Tests for :func:`place_patch`."""

    @pytest.fixture
    def bg_shape(self) -> tuple[int, int]:
        return (480, 640)

    def test_image_corners_qr_in_bounds(
        self,
        augmented_patch_fixture: AugmentedPatch,
        bg_shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        """All QR corners are within the background bounds after placement."""
        N = _N(3)  # version 3 from the fixture config
        config = AugmentationConfig(
            version=3,
            content="corner bounds test",
            quiet_zone_modules=4,
            target_ppm_range=(4.0, 4.0),
        )
        scale, tx, ty = sample_placement_scale(
            rng, augmented_patch_fixture.warped_patch.shape[:2], N, config, bg_shape
        )
        result = place_patch(augmented_patch_fixture, scale, tx, ty, bg_shape)

        bg_H, bg_W = bg_shape
        for pt in result.image_corners_qr:
            assert 0 <= pt[0] <= bg_W, f"x={pt[0]} out of bounds [0, {bg_W}]"
            assert 0 <= pt[1] <= bg_H, f"y={pt[1]} out of bounds [0, {bg_H}]"

    def test_mask_in_bounds(
        self,
        augmented_patch_fixture: AugmentedPatch,
        bg_shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        """Mask non-zero values only occur within the placed patch rectangle."""
        N = _N(3)
        config = AugmentationConfig(
            version=3,
            content="mask bounds test",
            quiet_zone_modules=4,
            target_ppm_range=(4.0, 4.0),
        )
        scale, tx, ty = sample_placement_scale(
            rng, augmented_patch_fixture.warped_patch.shape[:2], N, config, bg_shape
        )
        result = place_patch(augmented_patch_fixture, scale, tx, ty, bg_shape)

        mask = result.full_mask
        bg_H, bg_W = bg_shape

        # The scaled patch occupies (ty:ty+scaled_H, tx:tx+scaled_W)
        scaled_H = int(round(augmented_patch_fixture.warped_patch.shape[0] * scale))
        scaled_W = int(round(augmented_patch_fixture.warped_patch.shape[1] * scale))

        tx_int = int(round(tx))
        ty_int = int(round(ty))

        # Inside the rectangle, there should be some mask > 0
        roi = mask[ty_int : ty_int + scaled_H, tx_int : tx_int + scaled_W]
        assert np.any(roi > 0), "No mask values > 0 inside the expected rectangle"

        # Outside the rectangle, mask should be all 0
        # Check a few regions outside (before, after)
        if tx_int > 0:
            assert np.all(mask[:, :tx_int] == 0), "Mask non-zero left of patch"
        if tx_int + scaled_W < bg_W:
            assert np.all(mask[:, tx_int + scaled_W :] == 0), (
                "Mask non-zero right of patch"
            )
        if ty_int > 0:
            assert np.all(mask[:ty_int, :] == 0), "Mask non-zero above patch"
        if ty_int + scaled_H < bg_H:
            assert np.all(mask[ty_int + scaled_H :, :] == 0), (
                "Mask non-zero below patch"
            )

    def test_scale_1_and_no_translation(
        self,
        augmented_patch_fixture: AugmentedPatch,
        bg_shape: tuple[int, int],
    ) -> None:
        """Scale=1, tx=0, ty=0 maps warped patch TL → image (0, 0)."""
        result = place_patch(augmented_patch_fixture, 1.0, 0.0, 0.0, bg_shape)

        # With scale=1 and no translation, the first pixel of the warped patch
        # should appear at (0, 0) in the full image
        H, W = augmented_patch_fixture.warped_patch.shape[:2]
        np.testing.assert_array_equal(
            result.full_image[:H, :W, :],
            augmented_patch_fixture.warped_patch,
        )
        np.testing.assert_array_equal(
            result.full_mask[:H, :W],
            augmented_patch_fixture.warped_mask,
        )

        # The canvas outside the placed patch should be black (0)
        if W < bg_shape[1]:
            assert np.all(result.full_image[:, W:, :] == 0), (
                "Region right of patch not black"
            )
        if H < bg_shape[0]:
            assert np.all(result.full_image[H:, :, :] == 0), (
                "Region below patch not black"
            )

    def test_image_corners_qr_order(
        self,
        augmented_patch_fixture: AugmentedPatch,
        bg_shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        """QR corners maintain TL, TR, BR, BL order after affine transform."""
        N = _N(3)
        config = AugmentationConfig(
            version=3,
            content="corner order test",
            quiet_zone_modules=4,
            target_ppm_range=(4.0, 4.0),
        )
        scale, tx, ty = sample_placement_scale(
            rng, augmented_patch_fixture.warped_patch.shape[:2], N, config, bg_shape
        )
        result = place_patch(augmented_patch_fixture, scale, tx, ty, bg_shape)

        corners = result.image_corners_qr

        # TL: min x, min y
        # BR: max x, max y
        # TR: max x, min y
        # BL: min x, max y
        xs = corners[:, 0]
        ys = corners[:, 1]

        # TL is near-min x and min y
        tl = corners[0]
        tr = corners[1]
        br = corners[2]
        bl = corners[3]

        assert tl[0] <= tr[0] + 1e-9, f"TL x {tl[0]} > TR x {tr[0]}"
        assert tl[1] <= bl[1] + 1e-9, f"TL y {tl[1]} > BL y {bl[1]}"
        assert br[0] >= bl[0] - 1e-9, f"BR x {br[0]} < BL x {bl[0]}"
        assert br[1] >= tr[1] - 1e-9, f"BR y {br[1]} < TR y {tr[1]}"
