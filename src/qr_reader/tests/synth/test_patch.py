"""Tests for Phase 1 — QR Patch & Mask Generation."""

from __future__ import annotations

import numpy as np
import pytest
from qr_reader.synth.patch import (
    VALID_ECL,
    compute_qr_corners_patch_space,
    generate_qr_patch,
)


def _N(version: int) -> int:
    """Number of modules per side for a given QR version."""
    return 17 + 4 * version


# ===================================================================
# 1.1  generate_qr_patch tests
# ===================================================================


class TestGenerateQrPatch:
    """Tests for :func:`generate_qr_patch`."""

    def test_patch_shape(self):
        """patch_rgb.shape == (H, H, 3) where H = (N + 2*qz) * ppm."""
        version = 5
        ppm = 10
        qz = 4
        N = _N(version)
        expected_H = (N + 2 * qz) * ppm

        patch_rgb, mask = generate_qr_patch(
            version=version,
            content="hello",
            ecl_str="M",
            ppm=ppm,
            quiet_zone_modules=qz,
        )

        assert patch_rgb.shape == (expected_H, expected_H, 3), (
            f"Expected ({expected_H}, {expected_H}, 3), got {patch_rgb.shape}"
        )
        assert mask.shape == (expected_H, expected_H), (
            f"Expected ({expected_H}, {expected_H}), got {mask.shape}"
        )

    def test_mask_all_ones(self):
        """np.all(mask == 1.0)."""
        _, mask = generate_qr_patch(version=1, content="test", ecl_str="L", ppm=5)
        assert mask.dtype == np.float32
        assert np.all(mask == 1.0)

    def test_module_count(self):
        """The QR code proper (excl. quiet zone) has N×N visible modules."""
        version = 3
        ppm = 8
        qz = 4
        N = _N(version)

        patch_rgb, _ = generate_qr_patch(
            version=version,
            content="hello world",
            ecl_str="M",
            ppm=ppm,
            quiet_zone_modules=qz,
        )

        # The inner N×N module area should be non-uniform (not all white, not all
        # black) and its dimensions should match N*ppm.
        inner = N * ppm
        offset = qz * ppm
        inner_region = patch_rgb[offset : offset + inner, offset : offset + inner, :]

        assert inner_region.shape[:2] == (inner, inner), (
            f"Inner region expected ({inner}, {inner}), got {inner_region.shape[:2]}"
        )

        # There should be both black (0) and white (255) pixels in the QR region
        # (unless the content somehow produces a degenerate QR, which is extremely
        # unlikely for a non-trivial payload).
        assert np.any(inner_region == 0), "QR inner region has no black modules"
        assert np.any(inner_region == 255), "QR inner region has no white modules"

    def test_deterministic(self):
        """Same inputs → identical output."""
        kwargs = dict(version=1, content="deterministic", ecl_str="M", ppm=10)
        p1, m1 = generate_qr_patch(**kwargs)
        p2, m2 = generate_qr_patch(**kwargs)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(m1, m2)

    @pytest.mark.parametrize("version", [1, 7, 40])
    def test_version_bounds(self, version):
        """Versions 1, 7, 40 all produce valid output."""
        # Use a short payload for version 1 and padded for version 40
        content_sizes = {1: "x", 7: "hello world", 40: "x" * 200}
        patch_rgb, mask = generate_qr_patch(
            version=version,
            content=content_sizes[version],
            ecl_str="L",
            ppm=5,
        )
        N = _N(version)
        expected_H = (N + 2 * 4) * 5
        assert patch_rgb.shape == (expected_H, expected_H, 3)
        assert mask.shape == (expected_H, expected_H)

    @pytest.mark.parametrize("ecl", sorted(VALID_ECL))
    def test_ecl_all(self, ecl):
        """All four ECLs produce valid output."""
        patch_rgb, mask = generate_qr_patch(
            version=5, content="ecl test", ecl_str=ecl, ppm=6
        )
        N = _N(5)
        expected_H = (N + 2 * 4) * 6
        assert patch_rgb.shape == (expected_H, expected_H, 3)
        assert mask.shape == (expected_H, expected_H)

    def test_invalid_ecl_raises(self):
        """An invalid ECL string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ECL"):
            generate_qr_patch(version=1, content="x", ecl_str="X", ppm=5)


# ===================================================================
# 1.2  compute_qr_corners_patch_space tests
# ===================================================================


class TestComputeQrCornersPatchSpace:
    """Tests for :func:`compute_qr_corners_patch_space`."""

    @pytest.fixture
    def params(self):
        return dict(quiet_zone_modules=4, N=37, ppm=10)

    def test_corners_square(self, params):
        """Points form a square of side N * ppm."""
        corners = compute_qr_corners_patch_space(**params)
        side = params["N"] * params["ppm"]

        # Edge lengths
        def dist(i, j):
            return np.linalg.norm(corners[i] - corners[j])

        assert abs(dist(0, 1) - side) < 1e-9  # TL→TR
        assert abs(dist(1, 2) - side) < 1e-9  # TR→BR
        assert abs(dist(2, 3) - side) < 1e-9  # BR→BL
        assert abs(dist(3, 0) - side) < 1e-9  # BL→TL

        # Diagonals
        diag = side * np.sqrt(2)
        assert abs(dist(0, 2) - diag) < 1e-9  # TL→BR
        assert abs(dist(1, 3) - diag) < 1e-9  # TR→BL

    def test_corners_inside_patch(self, params):
        """All corners are within [0, patch_size]."""
        corners = compute_qr_corners_patch_space(**params)
        patch_size = (params["N"] + 2 * params["quiet_zone_modules"]) * params["ppm"]

        for pt in corners:
            assert 0 <= pt[0] <= patch_size, f"x out of bounds: {pt[0]}"
            assert 0 <= pt[1] <= patch_size, f"y out of bounds: {pt[1]}"

    def test_corners_order(self, params):
        """TL.x ≤ TR.x, TL.y ≤ BL.y, etc."""
        corners = compute_qr_corners_patch_space(**params)
        TL, TR, BR, BL = corners

        assert TL[0] <= TR[0], "TL.x > TR.x"
        assert TL[1] <= BL[1], "TL.y > BL.y"
        assert BL[0] <= BR[0], "BL.x > BR.x"
        assert TR[1] <= BR[1], "TR.y > BR.y"

    @pytest.mark.parametrize("v", [1, 7, 40])
    def test_corners_various_versions(self, v):
        """Works for different versions and PPM values."""
        for ppm in [5, 12]:
            N = _N(v)
            qz = 4
            corners = compute_qr_corners_patch_space(
                quiet_zone_modules=qz, N=N, ppm=ppm
            )
            side = N * ppm
            offset = qz * ppm
            np.testing.assert_allclose(corners[0], [offset, offset])
            np.testing.assert_allclose(corners[1], [offset + side, offset])
            np.testing.assert_allclose(corners[2], [offset + side, offset + side])
            np.testing.assert_allclose(corners[3], [offset, offset + side])
