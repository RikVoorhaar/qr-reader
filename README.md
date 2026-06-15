# qr-reader

A pure-Python QR code reader built from scratch. Detects QR codes in images and
decodes their text content — no external QR library needed at runtime.

> **Note:** This is work in progress.

## Quick Start

```bash
# Install
pip install -e .

# Run the full pipeline on a synthetic test image
python src/qr_reader/scripts/full-pipeline.py

# Run benchmarks
python src/qr_reader/scripts/qr_benchmark.py

# Run tests
pytest
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `opencv-python`
(for image I/O and binarization), `matplotlib` (visualization), `networkx`,
and `qrcode` (test image generation only).

## Architecture

The reader has two stages: **detection** (find the QR code in an image and sample
its module grid) and **decoding** (extract the text from the module grid).

### Detection Pipeline

```
Image
  → Binarize (Otsu threshold)
  → Alignment patterns (1:1:3:1:1 ratio scan)
  → Cluster candidates
  → Find corner points per cluster (angular NMS)
  → Extract finder patterns (the three large corner squares)
  → Associate colinear finder patterns
  → Find L-shaped triplets
  → Build landmarks (named reference points A–F)
  → Estimate version (cross-ratio matching)
  → Estimate homography (DLT → RANSAC → LM)
  → Sample module bit matrix
```

Three public API functions:
- **`detect_corners(image)`** → `(corners, version)` — 4 corner points in [TL, TR, BR, BL] order
- **`detect_homography(image)`** → `(H, version)` — 3×3 matrix mapping grid → image
- **`detect_sample(image)`** → bit matrix — N×N boolean array (True = dark module)

### Decoding Pipeline

```
Bit matrix
  → Format info (ECL + mask)
  → Version info (cross-check, v≥7)
  → Codeword extraction (unmask + zigzag)
  → De-interleave into data blocks
  → Reed-Solomon error correction (GF(256))
  → Bitstream decode → text
```

One public API function:
- **`decode(matrix)`** → decoded text string

## Module Layout

| Directory | Purpose |
|-----------|---------|
| `src/qr_reader/detector/` | Geometric detection: finder patterns, alignment, homography, version estimation |
| `src/qr_reader/decoder/` | Symbol decoding: format/version info, codewords, Reed-Solomon, bitstream |
| `src/qr_reader/scripts/` | Runnable scripts: full pipeline, benchmarks, debug tools |
| `src/qr_reader/tests/` | Unit tests for detector and decoder subsystems |
| `src/qr_reader/qr_gen.py` | Synthetic QR image generation for testing |

For a detailed file-by-file module map, see [AGENTS.md](AGENTS.md).

## Test Image Generation

`qr_gen.generate_test_image(seed=N)` produces reproducible synthetic QR images:

```
Clean QR → Rotate → Perspective warp → Gaussian noise → Gaussian blur
```

All randomness is seeded, so the same seed always produces the same image.
Use this for benchmarking and regression testing.

## Datasets

### HomeObjects-3K

A dataset of ~3 000 real-world images of household objects across multiple
rooms (`living_room`, `kitchen`, `bedroom`, `bathroom`), from the Ultralytics
model assets.

```bash
# Download and extract (skips if already present)
python src/qr_reader/scripts/download_homeobjects.py
```

The dataset lands in `data/` (gitignored):

```
data/
├── images/          # train/, val/, test/ splits
├── labels/          # YOLO-format annotations per split
├── HomeObjects-3K.yaml
└── LICENSE.txt
```

## Project Status

- ✅ Detection: finder patterns, alignment patterns, version estimation, homography
- ✅ Decoding: format info, version info, codeword extraction, de-interleaving
- ✅ Reed-Solomon error correction over GF(256)
- ✅ Bitstream decoding: numeric, alphanumeric, byte modes
- ⬜ Kanji mode support
- ⬜ Multi-QR detection in a single image
- ⬜ Structured Append (multi-symbol messages)
