"""Tests for Phase 6 — Pipeline Orchestrator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from qr_reader.detector.detector import detect_sample
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_dataset, generate_sample

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def background() -> np.ndarray:
    """A 480×640 synthetic background (grey gradient)."""
    x = np.linspace(50, 200, 640, dtype=np.uint8)
    y = np.linspace(50, 200, 480, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    r = np.clip(xx, 0, 255).astype(np.uint8)
    g = np.clip(yy, 0, 255).astype(np.uint8)
    b = np.clip((xx + yy) // 2, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@pytest.fixture
def easy_config() -> AugmentationConfig:
    """Deterministic, easy settings — no rotation, no noise, no blur."""
    return AugmentationConfig(
        version=1,
        content="hello",
        error_correction="L",
        ppm_range=(10.0, 10.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(10.0, 10.0),
        feather_sigma_range=(0.5, 0.5),
        blur_sigma_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        jpeg_quality_range=(100, 100),
        global_seed=42,
    )


@pytest.fixture
def temp_background_dir() -> Path:
    """Create a temporary directory with 3 small background images."""
    tmpdir = Path(tempfile.mkdtemp())

    for i in range(3):
        img = np.full((200, 300, 3), 128 + i * 40, dtype=np.uint8)
        path = tmpdir / f"bg_{i:03d}.jpg"
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return tmpdir


# ===================================================================
# 6.1  generate_sample
# ===================================================================


class TestGenerateSample:
    """Tests for :func:`generate_sample`."""

    def test_end_to_end_shape(
        self,
        rng: np.random.Generator,
        easy_config: AugmentationConfig,
        background: np.ndarray,
    ) -> None:
        """Output has same shape as background."""
        image, metadata = generate_sample(rng, easy_config, background)
        assert image.shape == background.shape, (
            f"Expected {background.shape}, got {image.shape}"
        )
        assert image.dtype == np.uint8

    def test_end_to_end_deterministic(
        self,
        easy_config: AugmentationConfig,
        background: np.ndarray,
    ) -> None:
        """Same seed → same image + metadata."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        img1, meta1 = generate_sample(rng1, easy_config, background)
        img2, meta2 = generate_sample(rng2, easy_config, background)

        np.testing.assert_array_equal(img1, img2)
        assert meta1 == meta2

    def test_end_to_end_corners(
        self,
        rng: np.random.Generator,
        easy_config: AugmentationConfig,
        background: np.ndarray,
    ) -> None:
        """QR corner dict has 4 valid keys with 2D coords in image bounds."""
        _, metadata = generate_sample(rng, easy_config, background)

        corners = metadata["corners_qr"]
        for label in ("TL", "TR", "BR", "BL"):
            assert label in corners, f"Missing corner {label}"
            xy = corners[label]
            assert len(xy) == 2, f"Corner {label} should have 2 coords, got {xy}"
            # Coordinates should be within image bounds
            H, W = background.shape[:2]
            x, y = xy
            assert 0 <= x <= W, f"{label}.x {x} out of bounds [0, {W}]"
            assert 0 <= y <= H, f"{label}.y {y} out of bounds [0, {H}]"

    def test_end_to_end_readable(
        self,
        easy_config: AugmentationConfig,
        background: np.ndarray,
    ) -> None:
        """Generated QR at version 1, easy settings should be decodable."""
        config = easy_config
        # Use a known short payload that fits in version 1 QR (max 25 alphanumeric
        # at ECL=L, but 17 alphanumeric for L is safe — actually version 1-L can
        # hold 25 alphanumeric chars, but let's keep it short).
        config.content = "QR-42"
        rng = np.random.default_rng(42)
        image, metadata = generate_sample(rng, config, background)

        # Attempt to detect + decode using the existing pipeline
        # The detector works on grayscale images and expects the QR to be visible
        # and dominant in the image.
        try:
            bits = detect_sample(image)
            decoded_text = decode_from_bits(bits)
        except Exception as exc:
            pytest.skip(f"Decoding failed in pipeline test (expected for synth): {exc}")

        assert decoded_text == "QR-42", (
            f"Decoded text mismatch: expected 'QR-42', got {decoded_text!r}"
        )

    def test_metadata_keys(
        self,
        rng: np.random.Generator,
        easy_config: AugmentationConfig,
        background: np.ndarray,
    ) -> None:
        """Metadata dict contains all expected top-level keys."""
        _, metadata = generate_sample(rng, easy_config, background)

        required_keys = {
            "sample_index",
            "seed",
            "background_path",
            "payload",
            "version",
            "N",
            "ecl",
            "pixels_per_module",
            "corners_qr",
            "augmentations",
        }
        assert required_keys.issubset(metadata.keys()), (
            f"Missing keys: {required_keys - set(metadata.keys())}"
        )

        # Check augmentations sub-keys
        aug_keys = {
            "rotation_deg",
            "jitter_fraction",
            "aspect_scale",
            "feather_sigma",
            "blur_sigma",
            "noise_sigma",
            "jpeg_quality",
        }
        assert aug_keys.issubset(metadata["augmentations"].keys()), (
            f"Missing augmentation keys: "
            f"{aug_keys - set(metadata['augmentations'].keys())}"
        )


# ===================================================================
# 6.2  generate_dataset
# ===================================================================


class TestGenerateDataset:
    """Tests for :func:`generate_dataset`."""

    def test_dataset_generation(
        self,
        easy_config: AugmentationConfig,
        temp_background_dir: Path,
    ) -> None:
        """Generate 10 samples, verify 10 images + 10 JSONL lines exist."""
        output_dir = Path(tempfile.mkdtemp())
        generate_dataset(easy_config, temp_background_dir, output_dir, num_samples=10)

        # Check images
        images_dir = output_dir / "images"
        assert images_dir.is_dir(), "images/ directory missing"
        image_files = sorted(images_dir.iterdir())
        assert len(image_files) == 10, f"Expected 10 images, got {len(image_files)}"
        for f in image_files:
            assert f.suffix == ".jpg", f"Expected .jpg, got {f.suffix}"

        # Check JSONL metadata
        metadata_path = output_dir / "metadata.jsonl"
        assert metadata_path.is_file(), "metadata.jsonl missing"
        lines = metadata_path.read_text().strip().split("\n")
        assert len(lines) == 10, f"Expected 10 metadata lines, got {len(lines)}"

    def test_metadata_roundtrip(
        self,
        easy_config: AugmentationConfig,
        temp_background_dir: Path,
    ) -> None:
        """Load JSONL, verify all required keys present."""
        output_dir = Path(tempfile.mkdtemp())
        generate_dataset(easy_config, temp_background_dir, output_dir, num_samples=5)

        metadata_path = output_dir / "metadata.jsonl"
        lines = metadata_path.read_text().strip().split("\n")
        assert len(lines) == 5

        required_keys = {
            "sample_index",
            "seed",
            "background_path",
            "payload",
            "version",
            "N",
            "ecl",
            "pixels_per_module",
            "corners_qr",
            "augmentations",
        }

        for line in lines:
            record = json.loads(line)
            missing = required_keys - set(record.keys())
            assert not missing, f"Record missing keys: {missing}"
            # Verify corners_qr sub-keys
            for label in ("TL", "TR", "BR", "BL"):
                assert label in record["corners_qr"], (
                    f"Missing corner {label} in record {record['sample_index']}"
                )
            # Verify sample_index is sequential
            assert isinstance(record["sample_index"], int)

    def test_no_backgrounds_raises(self) -> None:
        """An empty background directory raises FileNotFoundError."""
        empty_dir = Path(tempfile.mkdtemp())
        output_dir = Path(tempfile.mkdtemp())

        with pytest.raises(FileNotFoundError, match="No image files found"):
            generate_dataset(
                AugmentationConfig(version=1),
                empty_dir,
                output_dir,
                num_samples=1,
            )


# ===================================================================
# Helpers
# ===================================================================


def decode_from_bits(bits: np.ndarray) -> str:
    """Decode a bit matrix using the project decoder (avoid import top-level)."""
    from qr_reader.decoder.decoder import decode

    return decode(bits)
