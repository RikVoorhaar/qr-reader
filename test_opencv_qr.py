"""Test OpenCV's QR detector against synthetic QR samples.

Generates 5 easy QR images using the augmentation pipeline, saves them
as PNG (no JPEG recompression) and JPEG, then runs OpenCV's
QRCodeDetector on each and reports results.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

OUT_DIR = Path("/tmp/qr_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 5
SEED = 42

# Easy settings: mild rotation, very little jitter, low noise, no blur, no JPEG
config = AugmentationConfig(
    version=3,
    content="Hello QR Reader!",
    error_correction="L",  # lowest EC = easier to decode
    ppm_range=(8.0, 12.0),  # nice crisp modules
    rotation_deg_range=(0.0, 20.0),  # mild rotation
    jitter_fraction=0.02,  # very little perspective jitter
    aspect_scale_range=(0.95, 1.05),
    target_ppm_range=(6.0, 10.0),
    feather_sigma_range=(0.5, 1.0),
    blur_sigma_range=(0.0, 0.3),  # barely any blur
    noise_sigma_range=(0.0, 2.0),  # low noise
    jpeg_quality_range=(95, 100),  # nearly lossless even for JPEG
    global_seed=SEED,
)

print("=" * 70)
print("OpenCV QR Detector Test — Synthetic QR Samples")
print("=" * 70)
print(f"Config: version={config.version}, content={config.content!r}")
print(f"Output dir: {OUT_DIR}")
print()

# ---------------------------------------------------------------------------
# Create a synthetic gradient background
# ---------------------------------------------------------------------------

H, W = 800, 800
xx = np.linspace(0, 1, W, dtype=np.float32).reshape(1, -1)
yy = np.linspace(0, 1, H, dtype=np.float32).reshape(-1, 1)
background = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
background = np.stack([background] * 3, axis=-1)  # (H, W, 3) RGB

# ---------------------------------------------------------------------------
# Generate samples & test
# ---------------------------------------------------------------------------

results = []

for i in range(NUM_SAMPLES):
    rng = np.random.default_rng(SEED + i)
    image_rgb, metadata = generate_sample(
        rng=rng,
        config=config,
        background=background,
        sample_index=i,
        background_path="synthetic-gradient",
    )

    # Save as JPEG (what the normal pipeline does)
    jpeg_path = OUT_DIR / f"sample_{i:03d}.jpg"
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(jpeg_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save as PNG (no recompression — raw output)
    png_path = OUT_DIR / f"sample_{i:03d}.png"
    cv2.imwrite(str(png_path), image_bgr)

    # Run OpenCV QR detector on the PNG (no JPEG artifacts)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()
    data, bbox, rectified = detector.detectAndDecode(gray)

    decoded_ok = bool(data)
    results.append(
        {
            "index": i,
            "seed": SEED + i,
            "decoded": decoded_ok,
            "payload": data if decoded_ok else None,
            "bbox": bbox.tolist() if bbox is not None else None,
            "jpeg_size": os.path.getsize(jpeg_path),
            "png_size": os.path.getsize(png_path),
        }
    )

    status = "✓ DECODED" if decoded_ok else "✗ FAILED"
    payload_str = f" → {data!r}" if decoded_ok else ""
    print(
        f"  Sample {i:2d} (seed={SEED + i:3d}): {status}{payload_str}"
        f"  |  JPEG={results[-1]['jpeg_size']:>6} B  PNG={results[-1]['png_size']:>6} B"
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("-" * 70)
success = sum(1 for r in results if r["decoded"])
print(f"  Decoded: {success}/{NUM_SAMPLES}")
print()
print(f"Saved files in {OUT_DIR}:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size} B)")
print()
print("(end)")
