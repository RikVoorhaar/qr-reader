# qr-reader

A QR code reader implemented in Python. It detects QR codes in images and decodes
their text without calling OpenCV or another QR library in the reader. OpenCV is
used for testing, image synthesis, and comparison.

## Quick Start

```bash
# Install the package and test dependencies
pip install -e '.[dev]'

# Run the full pipeline on a synthetic test image
python src/qr_reader/scripts/full-pipeline-current.py

# Run benchmarks
python src/qr_reader/scripts/qr_benchmark.py

# Run tests
pytest
```

Requires Python ≥ 3.10. See [`pyproject.toml`](pyproject.toml) for the full
dependency list. OpenCV is installed for supporting scripts and tests, but the
detector and decoder do not import it.

## Architecture

The reader has two stages: **detection** (find the QR code in an image and sample
its module grid) and **decoding** (extract the text from the module grid).

### Detection Pipeline

```
Image
  → Binarize (illumination correction + Otsu)
  → Scan for finder candidates (1:1:3:1:1)
  → Cluster candidates
  → Fit finder pattern per cluster (ray-profile + edge fit)
  → Deduplicate overlapping candidates
  → Find L-shaped triplets (edge associations, offset tolerance 0.62)
  → Rank triplets and versions (cross-ratio landmarks)
  → Fit homography for a candidate (12-corner DLT + LM)
  → Refine homography with interior alignment patterns (versions ≥ 2)
  → Sample module bit matrix
  → Decode; on failure try the next bounded (triplet, version)
```

If the first pass fails, the reader follows one of two paths:

- With exactly two fitted finders, enlarge the grayscale image by 1.25× and
  rerun the standard pass once.
- With ranked triplets, reuse the same triplet and version candidates. First
  sample with the 12-corner homography and a single threshold, then with the
  refined homography and a threshold that varies across the grid.

Three single-candidate detection functions:
- **`detect_corners(image)`** → `(corners, version)` — 4 corner points in [TL, TR, BR, BL] order
- **`detect_homography(image)`** → `(H, version)` — 3×3 matrix mapping grid → image
- **`detect_sample(image)`** → bit matrix — N×N boolean array (True = dark module)

And one full end-to-end function:
- **`decode_image(image, expected_payload=None, ...)`** → `DecodeResult` —
  tries a bounded set of cross-ratio-ranked triplets and versions, samples and
  decodes each, then uses gated retries without widening candidate/version
  search. All retries share the call's time budget and the attempt trace records
  each pass's provenance. The scale retry uses SciPy only and maps its returned
  homography to the original image coordinates.

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
| `src/qr_reader/pipeline.py` | End-to-end `decode_image`: multi-triplet / multi-version attempt loop |
| `src/qr_reader/scripts/` | Runnable scripts: full pipeline, benchmarks, debug tools |
| `src/qr_reader/tests/` | Unit tests for detector and decoder subsystems |
| `src/qr_reader/qr_gen.py` | Synthetic QR image generation for testing |
| `src/qr_reader/synth/` | Synthetic scenes and larger benchmark datasets |
| `blog_figures/` | Article figure generators and supporting benchmark measurements |

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

### Captured QR Benchmark

A benchmark of 100 real-world photos of QR codes encoding
`rikvoorhaar.com`, all 1920×1440 JPEGs. The expected payload is known, but the
finder and QR corner positions are not annotated. `benchmark_captured.py`
reports a detection funnel, decode accuracy, and wall time.

```bash
# Download and extract
mkdir -p data/bench
curl -L -o /tmp/captured-qr-benchmark-v1.tar.gz \
  https://github.com/RikVoorhaar/qr-reader/releases/download/benchmark-v1/captured-qr-benchmark-v1.tar.gz
tar -xzf /tmp/captured-qr-benchmark-v1.tar.gz -C data/bench/
```

The images land in `data/bench/captured/` (gitignored):

```
data/bench/captured/
├── qr_code_ - 1.jpeg
├── qr_code_ - 2.jpeg
└── ... (100 total)
```

Run the benchmark:

```bash
python src/qr_reader/scripts/benchmark_captured.py
```

## Article figures

The article lives in the separate [blog repository](https://github.com/RikVoorhaar/blog).
You don't need it to run or test the reader. To regenerate article assets,
check out the blog repository separately at `blog/` and run the scripts under
`blog_figures/`. The QR reader repository ignores that checkout; dataset inputs
and commands are documented in the individual scripts.

## Project Status

- ✅ Detection: finder patterns, alignment patterns, version estimation, homography
- ✅ Decoding: format info, version info, codeword extraction, de-interleaving
- ✅ Reed-Solomon error correction over GF(256)
- ✅ Bitstream decoding: numeric, alphanumeric, byte modes
- ⬜ Kanji mode support
- ⬜ Multi-QR detection in a single image
- ⬜ Structured Append (multi-symbol messages)
