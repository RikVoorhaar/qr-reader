# AGENTS.md — QR Reader

> **Maintenance rule:** When you modify the detection or decoding pipeline (the
> sequence of function calls in `detector.py` or `decoder.py`), update the
> "Data Flow" and "Module Map" sections below, and `README.md`'s "Architecture"
> section. When you add or rename a domain concept, update `CONTEXT.md`.
> Run the `doc-maintenance` skill afterward to audit for drift.

## Module Map

Every source file under `src/qr_reader/` and its role.

### Root

| File | Purpose | Depends on |
|------|---------|------------|
| `qr_gen.py` | Generate synthetic QR images with noise, rotation, and perspective warp for testing | `qrcode`, `cv2` |
| `__init__.py` | Package root (empty) | — |

### Detector (`detector/`)

| File | Purpose | Depends on |
|------|---------|------------|
| `detector.py` | High-level API: `detect_corners`, `detect_homography`, `detect_sample`. Orchestrates `_run_detection`. | All detector modules, `qr_gen.binarize_image` |
| `alignment.py` | Find candidate alignment patterns via 1:1:3:1:1 ratio scanning (horizontal + vertical cross-validation) | — |
| `clustering.py` | Merge overlapping alignment pattern candidates into `CandidateCluster` groups | — |
| `corner.py` | Angular non-maximum suppression to pick the 4 corner points of a region | — |
| `finder_pattern.py` | Extract `FinderPattern` quads, check pairwise `Association`, find L-shaped `Triplet`s | `geometry.py` |
| `geometry.py` | Low-level geometric primitives: angular distance, local offset, polygon area, segment intersection | — |
| `homography.py` | DLT, RANSAC, Levenberg-Marquardt homography estimation, point projection, QR corner projection | `scipy` |
| `landmarks.py` | Build `NamedLandmarks` from a Triplet, compute colinear quadruples for cross-ratio measurement | `finder_pattern.py` |
| `region.py` | Wave-front fill, 8-connected boundary trace, connected-components on boundaries | `scipy`, `networkx` |
| `sample.py` | Sample the module bit matrix from the rectified QR image using the homography | `homography.py`, `scipy` |
| `version.py` | Cross-ratio measurement, constraint building/filtering, version estimation | `landmarks.py` |
| `roi.py` | Compute padded bounding box from `CandidateCluster`, extract clamped sub-image cutout | `clustering.py` |

### Decoder (`decoder/`)

| File | Purpose | Depends on |
|------|---------|------------|
| `decoder.py` | Top-level `decode()`: format info → version info → codewords → de-interleave → RS → bitstream | All decoder modules |
| `format_info.py` | Read and decode the 15-bit format information from the module matrix | `tables.py` |
| `version_info.py` | Read and decode the 18-bit version information (versions ≥ 7) | `tables.py` |
| `codeword_extractor.py` | Unmask and zigzag-extract raw codewords from the module matrix | `tables.py` |
| `data_block.py` | De-interleave raw codewords into error-correction data blocks | `tables.py` |
| `rs.py` | Reed-Solomon error correction over GF(256): syndrome, Euclidean, Forney | `gf.py` |
| `gf.py` | GF(256) arithmetic tables (exp, log, multiply, inverse) | — |
| `bitstream.py` | Decode corrected data bytes into text: numeric, alphanumeric, byte modes | `tables.py` |
| `tables.py` | QR spec tables: version info, EC block layout, mode indicators, character counts, masking | — |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `full-pipeline.py` | End-to-end pipeline: generate test image → detect → decode, with visualization |
| `qr_benchmark.py` | Benchmark detection/decoding across versions, seeds, and transforms |
| `debug_find_all_associations.py` | Targeted debug script for the `find_all_associations` high-version failure |

### Tests (`tests/`)

| Path | Purpose |
|------|---------|
| `test_detector.py` | Detection pipeline tests |
| `decoder/` | Decoder unit tests (per module) |
| `detector/` | Detector unit tests (per module) |

## Data Flow

### Full pipeline (detect + decode)

```
Image (ndarray)
  │
  ▼
qr_gen.binarize_image()                    → binary image
  │
  ▼
alignment.find_alignment_patterns_2d()     → (rows, cols_all) candidate positions
  │
  ▼
clustering.cluster_candidates()            → list[CandidateCluster]
  │
  ▼ (per cluster)
region.region_fill_wave_front()            → region mask
region.region_boundary_8()                 → boundary trace
region.boundary_connected_components_ndimage() → boundary components
corner.angular_nms_top_radial_indices()    → 4 corners per component
  │
  ▼
finder_pattern.extract_finder_patterns()   → list[FinderPattern]
finder_pattern.find_all_associations()     → list[Association]
finder_pattern.find_triplets()             → list[Triplet] (take first)
  │
  ▼
landmarks.build_named_landmarks()          → NamedLandmarks (image_lm)
version.build_constraints()                → list[Constraint]
version.filter_constraints()               → filtered constraints
version.estimate_version()                 → version (V), module count (N)
  │
  ▼
landmarks.canonical_grid_landmarks()       → grid landmarks
landmarks.build_named_landmarks() [2nd]    → NamedLandmarks (image_lm)
homography.ransac_homography()             → H (initial)
homography.refine_homography_lm()          → H (refined)
  │
  ▼
sample.sample_qr_bits()                    → (N, N) bool bit matrix
  │
  ▼
decoder.decode()                           → decoded text string
```

### Decoder sub-pipeline (`decode()`)

```
bit matrix (N×N bool)
  │
  ├─ format_info.decode_format_info()      → (ecl_idx, mask_idx)
  ├─ version_info.decode_version_info()    → version (cross-check, v≥7 only)
  ├─ codeword_extractor.extract_codewords() → raw codewords (unmasked)
  ├─ data_block.deinterleave()             → data blocks (data + EC)
  ├─ rs.rs_decode()                        → corrected data bytes (per block)
  └─ bitstream.decode_bitstream()          → text string
```

## Key Data Structures

- **`FinderPattern`** (`finder_pattern.py`): `cluster_idx`, `outer_corners` (4×2), `inner_corners` (4×2 | None). Corners are in (row, col).
- **`Association`** (`finder_pattern.py`): `fp1_idx`, `fp2_idx`, plus colinear segment indices.
- **`Triplet`** (`finder_pattern.py`): `top_left_idx`, `top_right_idx`, `bottom_left_idx` — indices into the finder pattern list.
- **`NamedLandmarks`** (`landmarks.py`): Named attributes A–F as (row, col) arrays or None, representing specific points on the QR grid.
- **`Constraint`** (`version.py`): `type` ("outer"/"inner"), `label`, `r_measured`, `line_error`, `span`.
- **`CandidateCluster`** (`clustering.py`): `row`, `cols` (6 boundaries), `height`, `num_candidates`.
- **`Quadruple`** (`landmarks.py`): `points` (4×2), `type` ("outer"/"inner"), `label` — four colinear points for cross-ratio measurement.
- **`DataBlock`** (`data_block.py`): `data: bytes`, `ec: bytes` — one error-correction block after de-interleaving.

## Coordinate Conventions

- **Detector modules** work in **(row, col)** image coordinates. Finder pattern corners, alignment candidates, landmarks all use (row, col).
- **Homography module** works in **(x, y)**. Conversion happens at the call boundary via `pts[:, ::-1]`.
- **Decoder modules** work in **(col, row)** grid coordinates matching the QR spec layout.
- **`detector.py`** contains a `rc_to_xy` helper and a triplet label swap to bridge these conventions.

## Common Modification Tasks

- **Add a new encoding mode** (e.g., Kanji): add tables to `decoder/tables.py`, add a `_decode_kanji` function in `decoder/bitstream.py`, wire it into `decode_bitstream`.
- **Improve finder pattern detection**: modify `detector/finder_pattern.py` (extraction, association, triplet finding). The thresholds (`angle_tol`, `offset_tol`) live in `check_association` and `find_all_associations`.
- **Change the detection pipeline order**: edit `_run_detection` in `detector/detector.py`. Update the Data Flow in this file and the Architecture section in README.md.
- **Add a new QR version table entry**: edit `tables.py` `VERSIONS` dict (EC block layout, alignment pattern positions, etc.).
- **Fix a version estimation issue**: work in `detector/version.py` (cross-ratio computation, constraint filtering, `estimate_version`).
- **Change binarization**: edit `qr_gen.binarize_image`. The default uses Otsu thresholding.
- **Run the full pipeline for debugging**: `python src/qr_reader/scripts/full-pipeline.py`.
- **Run benchmarks**: `python src/qr_reader/scripts/qr_benchmark.py`.

## Testing

- Tests live under `src/qr_reader/tests/`, organized by subsystem (decoder/, detector/).
- Test QR images are generated via `qr_gen.generate_test_image(seed=N)` for reproducibility.
- The `qr_gen.py` pipeline: clean QR → rotate → perspective warp → Gaussian noise → blur.
- Run tests: `pytest` from the repo root.
- `third_party/` is excluded from pytest via `pyproject.toml`.

## Issue Tracking

Issues are managed via the **GitHub MCP** tooling (`issue_write`, `issue_read`,
`list_issues`, `search_issues`, etc.) on `RikVoorhaar/qr-reader` — not in local
markdown files.

### When to create an issue

- You find a bug you cannot fix inline in the current task.
- You complete work and identify a necessary follow-up.
- **Before filing a speculative enhancement**, ask the maintainer first.

### Issue format

Use `issue_write` with `method: create`. Always include:

- `title` — concise summary
- `body` — description, approach, out-of-scope, success criteria
- `labels` — appropriate GitHub labels (e.g. `bug`, `enhancement`, `experiment`)

### Status workflow

Use `issue_write` with `method: update` to transition state:

`open` → `state: closed` / `state_reason: not_planned`

No local files needed — everything lives on GitHub.
