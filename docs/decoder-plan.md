# QR Code Decoder — Implementation Plan

## 1. Scope

Decode standard **QR Code Model 2, Versions 1–40, ECL L/M/Q/H**, with these encoding modes:

- Numeric
- Alphanumeric
- Byte

We assume the input is a **bit matrix** (2D `numpy` boolean array, `True` = dark module, `False` = light module)
representing the full QR code symbol at its native size.  The sampling step (image → bit matrix) is
implemented separately.

### Out of scope

- Micro QR, rMQR, Model 1
- Structured Append, FNC1, ECI, Kanji/Hanzi
- Curved / perspective-distorted symbols (that is the sampler’s job)

---

## 2. Third-party reference files

The directory `third_party/` contains three relevant implementations.  The files listed below are
the ones most useful as reference during implementation.

### 2.1 zxing-cpp (C++ decoder — most complete reference)

| File | Purpose |
|------|---------|
| `core/src/qrcode/QRDecoder.cpp` | Top-level decoder: format → version → codewords → de-interleave → RS → bitstream |
| `core/src/qrcode/QRBitMatrixParser.cpp` | Read format/version info, extract codewords from matrix |
| `core/src/qrcode/QRFormatInformation.h` | Format info data structure, BCH constants |
| `core/src/qrcode/QRFormatInformation.cpp` | BCH decode format info (hamming-distance match against 32 patterns) |
| `core/src/qrcode/QRVersion.h` | Version class, `SymbolSize()`, `DecodeVersionInformation()`, `Number()` |
| `core/src/qrcode/QRVersion.cpp` | Version table for v1–40 (EC block sizes, alignment positions), BCH decode version |
| `core/src/qrcode/QRECB.h` | EC blocks structure (`ECB`, `ECBlocks`) |
| `core/src/qrcode/QRDataBlock.cpp` | De-interleave raw codewords into EC blocks |
| `core/src/qrcode/QRCodecMode.h` | Mode enum, `CharacterCountBits()`, `CodecModeBitsLength()` |
| `core/src/qrcode/QRCodecMode.cpp` | Mode-to-bits mapping, character count tables |
| `core/src/qrcode/QRDataMask.h` | The 8 data mask functions |
| `core/src/qrcode/QRMaskUtil.cpp` | Mask penalty scoring (encoder only, but useful for understanding) |
| `core/src/ReedSolomonDecoder.cpp` | Complete RS decoder (syndromes, Euclidean algo, Forney) |
| `core/src/ReedSolomonDecoder.h` | Interface |
| `core/src/GenericGF.h` | GF(256) arithmetic (exp/log tables, `multiply`, `inverse`) |
| `core/src/GenericGF.cpp` | GF(256) construction (primitive polynomial 0x11D) |
| `core/src/GenericGFPoly.h` | GF polynomial arithmetic |
| `core/src/BitSource.h` / `.cpp` | Bit-level reader for decoded data stream |

### 2.2 nayuki-qr-code-generator (Python encoder — canonical tables)

| File | Purpose |
|------|---------|
| `python/qrcodegen.py` | Encoder with all tables: alignment positions, capacity per version/ECL, format-version BCH generation, mask penalty.  Lines ~405–650 contain the key data tables. |

### 2.3 OpenCV (C++ — decoder + encoder tables)

| File | Purpose |
|------|---------|
| `modules/objdetect/src/qrcode_encoder_table.inl.hpp` | Capacity tables and format/version info tables in C++ |
| `modules/objdetect/src/qrcode.cpp` (lines 2831–2939) | The `decodingProcess()` method shows how OpenCV integrates quirc |
| `modules/objdetect/src/qrcode_encoder.cpp` | GF(256) arithmetic — alternative simple implementation |

---

## 3. Decoder pipeline (overview)

```
BitMatrix (size×size bool)
  │
  ├─ 1. Determine version from size: version = (size - 17) // 4
  │
  ├─ 2. Read + BCH-decode format info → ECL, mask index
  │
  ├─ 3. Read + BCH-decode version info (V ≥ 7 only)
  │
  ├─ 4. Unmask data modules with mask pattern
  │
  ├─ 5. Extract codewords in zigzag order (skip function patterns)
  │
  ├─ 6. De-interleave codewords into error-correction blocks
  │
  ├─ 7. Reed-Solomon error correction per block
  │
  ├─ 8. Reassemble corrected data bytes
  │
  └─ 9. Decode bit stream into text
```

Each step is independently testable.

---

## 4. Phase breakdown

### Phase 0 — Project scaffold + GF(256) Arithmetic ✅ COMPLETED

**Status**: Done.  All 20 tests pass.  `GF256` class with `multiply`, `add`/`subtract` (XOR), `inverse`, `pow`, `exp`, `log`.  Log/exp tables precomputed at module level.

**Files**: `src/qr_reader/decoder/__init__.py`, `src/qr_reader/decoder/gf.py`

**What**: Set up the `decoder` package.  Implement GF(256) arithmetic using log/exp tables
(primitive polynomial `0x11D`, generator `α = 0x02`).

- `GF256` class with `multiply(a, b)`, `add(a, b)` (= XOR), `inverse(a)`, `pow(a, n)`.
- Precompute 256-entry `exp` and `log` tables at module level.
- Note: for QR, generator base `b = 0`, so the generator polynomial is
  `g(x) = (x − α^0)(x − α^1)…(x − α^(2t−1))`.

**Tests**:

- Verify `multiply(α^i, α^j) == α^(i+j)` for all i, j.
- Verify `inverse(x) * x == 1` for all x ≠ 0.
- Verify known products from spec examples.

**References**: `zxing GenericGF.h/.cpp`, `opencv qrcode_encoder.cpp` (gfMul/gfDiv/gfPow).

---

### Phase 1 — Tables (version, capacity, alignment, masks, mode) ✅ COMPLETED

**Status**: Done.  All 43 tests pass.  Version info with EC blocks (zxing format), format/version BCH patterns, mask functions (0–7), character count bit lengths, mode indicators, alphanumeric table.  `tables.py` is pure data, no logic beyond table construction.

**Files**: `src/qr_reader/decoder/tables.py`

**What**: Hard-code all spec tables as Python dataclasses / dicts.  No logic, just data.

1. **Version info** per version (1–40):
   - Symbol size (`17 + 4*v`)
   - EC blocks per ECL (L/M/Q/H): list of `(num_data_codewords, num_blocks)` pairs.
     (See `QRVersion.cpp` lines 38–200+ for the full table.)
   - Alignment pattern centers (list of ints).

2. **Format information table**: the 32 valid 15-bit masked patterns and their
   (ECL, mask) interpretations.  Used for BCH matching in Phase 2.

3. **Version information table** (versions 7–40): the 34 valid 18-bit patterns
   mapping to version numbers.  Used for BCH matching in Phase 3.

4. **Character count indicator bit lengths** per mode per version range:
   - Numeric: 10 / 12 / 14 for v1–9 / v10–26 / v27–40
   - Alphanumeric: 9 / 11 / 13
   - Byte: 8 / 16 / 16

5. **Mask functions** (0–7) as callables `mask(x, y) → bool`.

6. **Function module pattern mask**: for a given version, which (x, y) positions are
   occupied by function patterns (finder, timing, alignment, format, version info).
   This can reuse logic from `nayuki qrcodegen.py: _draw_function_patterns()`.

**Tests**:

- Verify total codewords for each (version, ECL) match spec.
- Verify alignment positions match spec.
- Verify mask functions against known patterns.

**References**:
- `nayuki qrcodegen.py` (tables inline in class, lines ~405–650)
- `zxing QRVersion.cpp`
- `zxing QRDataMask.h`
- `zxing QRCodecMode.cpp`

---

### Phase 2 — Format Information BCH Decoding ✅ COMPLETED

**Files**: `src/qr_reader/decoder/format_info.py`

**What**: Read and decode the 15-bit format information.

1. **Read raw bits** from the two locations in the bit matrix:
   - Location 1: around the top-left finder pattern (bits at positions defined in the spec).
   - Location 2: along the left edge and top edge near the bottom-left finder.

2. **BCH decode via hamming-distance matching**: compare each of the two 15-bit readings
   against the 32 valid patterns (already XOR-masked per the `tables.py` data).
   Pick the one with smallest hamming distance.
   If distance ≤ 3, it's a decode.  Return ECL (2 bits) and mask index (3 bits).

3. **No algebraic BCH decode** — just exhaustive lookup, matching the design decision.

**Tests**: 42 tests in `tests/decoder/test_format_info.py`

- Bit-position reading via manual pattern placement and full roundtrip of all 32 patterns.
- Integration with `qrcode` library for real QR codes across versions 1–40 and all ECLs.
- Error correction: tolerates 1–3 bit flips, rejects heavily corrupted data.
- Cross-copy redundancy: one clean copy compensates for errors in the other.
- Shared test helper `tests/decoder/helpers.py` with `make_qr_bitmatrix()`.

**References**:
- `zxing QRFormatInformation.cpp`
- `nayuki qrcodegen.py: _draw_format_bits()` (encodes but shows the bit layout)

**Handover notes**:
- `VALID_FORMAT_PATTERNS` in `tables.py` already includes the 0x5412 XOR mask — compare raw reads directly, no unmasking needed.
- `_read_format_info_bits()` uses (x, y) tuples from the spec; lookup is `matrix[y, x]` (row-major).
- Bit ordering matches nayuki's encoder: MSB at `(0,8)` for location 1, `(8, size-1)` for location 2.
- `decode_format_info(matrix, version)` returns `(ecl_index, mask_index)` — ecl_index matches `ECL_L=0`, `ECL_M=1`, `ECL_Q=2`, `ECL_H=3`.
- Dark module at `(8, size-8)` is correctly skipped in location 2 bit reading.

---

### Phase 3 — Version Information BCH Decoding (V ≥ 7) ✅ COMPLETED

**Files**: `src/qr_reader/decoder/version_info.py`

**What**: Read and decode the 18-bit version information (only needed for V ≥ 7).

1. **Read raw bits** from the two locations (near top-right and bottom-left finders).

2. **BCH decode via hamming-distance matching** against the 34 valid patterns.
   If distance ≤ 3, return version number.

**Tests**:

- Verify on v7+ QR codes.
- Verify no false positives on v1–v6.

**References**:
- `zxing QRVersion.cpp: DecodeVersionInformation()`
- `nayuki qrcodegen.py: _draw_version()`

**Handover notes**:
- `read_version_bits(bit_matrix)` reads two 18-bit copies — bit order follows the nayuki _draw_version layout: bit 0 = row 0, col dim-11; per group of 3 bits increment row; increment within group increments col.
- `decode_version(bits_a, bits_b)` does hamming-distance matching with threshold ≤ 3.
- `decode_version_info(bit_matrix)` is the top-level convenience function — returns None for dim < 45 (versions < 7).
- Test helper `_make_qr_matrix(version, ecc, mask)` uses `qr.makeImpl(False, mask)` to force a specific mask pattern (the public `make()` API does not support mask forcing in the installed qrcode library).
- All 63 tests pass (0.57s). No integration with Phase 4 yet; `decode_version_info` accepts a raw bit matrix independently.

---

### Phase 4 — Codeword Extraction ✅ COMPLETED

**Files**: `src/qr_reader/decoder/codeword_extractor.py`

**What**: Extract raw codeword bytes from the bit matrix.

1. **Build function pattern mask** for the version (which modules are occupied by
   finders, timing, alignment, format, version info).  Uses tables from Phase 1.

2. **Walk the zigzag pattern**: columns from right to left in pairs, alternating
   up/down direction, skipping column 6 (vertical timing), skipping function modules.

3. **Unmask each data module** with the mask pattern (XOR).

4. **Pack bits into bytes** (big-endian within each byte).

5. Return `bytes` of total codewords.  Verify length matches `total_codewords` for version.

**Key implementation details**:
- Finder patterns occupy 8×8 modules (not 9×9) because the encoder clamps the
  separator row/col at the matrix boundary.
- Alignment patterns whose centre module overlaps a finder pattern are skipped
  by the encoder; our mask replicates this skip.
- **Remainder bits**: some versions have 0–7 extra data modules that don't form
  a complete byte. The encoder leaves them as 0; our extractor stops reading
  after collecting `total_codewords × 8` bits to avoid padding artifacts.

**Tests**:

- Generate a QR code, extract codewords, verify they match the encoder's output codewords.
- Test with all 8 mask patterns.
- Test zigzag order correctness on small versions.
- Full pipeline: extract → deinterleave → RS decode → bitstream decode.

**References**:
- `zxing QRBitMatrixParser.cpp: ReadQRCodewords()`
- `nayuki qrcodegen.py: _draw_codewords()` (the inverse process)

---

### Phase 5 — Data Block De-interleaving ✅ COMPLETED

**Status**: Done.  All 20 tests pass.

**Files**: `src/qr_reader/decoder/data_block.py`

**What**: Split raw codewords into the error-correction blocks defined by the version+ECL table,
then reassemble the data bytes in order.

1. Look up EC block layout from the table: for each group, `(num_data_codewords, num_blocks)`.
2. Allocate block buffers of the correct size.
3. De-interleave: raw codewords come as `byte_0_of_block_0, byte_0_of_block_1, …, byte_N_of_block_0, …`
   followed by EC codewords in the same interleaved order.
4. Return a list of `(data_codewords, ec_codewords)` blocks.

**API**:

- `DataBlock` dataclass with `data: bytes` and `ec: bytes` fields.
- `deinterleave(raw: bytes, version: int, ecl: str) -> List[DataBlock]` — main entry point.
  Validates raw length against `total_codewords()`, raises `ValueError` on mismatch.

**Tests** (20 total):

- Single block (v1 all ECLs: no interleaving).
- Round-trip: manual interleave → deinterleave → exact byte recovery (equal blocks, unequal
  blocks, many blocks, v5-H two groups).
- Structural verification across all 160 version×ECL combos: block count, data lengths,
  EC lengths, total byte preservation.
- Specific multi-group versions: v5-H, v8-M, v10-L, v40-H.
- Error cases: invalid version (<1 or >40), wrong raw length, invalid ECL.

**Handover notes for Phase 6 (RS Error Correction)**:
- The RS decoder receives a list of `DataBlock` objects from this phase.
- Each `DataBlock` contains `data: bytes` and `ec: bytes`.
- RS should attempt to correct errors in each block independently using the GF(256) arithmetic
  from Phase 0.
- The RS decoder output would be the corrected `data` bytes concatenated in block order.

**Handover notes for Phase 7 (Bit Stream Decoding)**:
- After RS correction, the corrected data bytes from all blocks are concatenated sequentially.
- The bit stream decoder receives this concatenated byte sequence and decodes mode indicators,
  character counts, and payload data.

**References**:
- `zxing QRDataBlock.cpp`
- `zxing QRECB.h`

---

### Phase 6 — Reed-Solomon Error Correction over GF(256)  ✅ COMPLETED

**Files**: `src/qr_reader/decoder/rs.py`, `tests/decoder/test_rs.py`

**What**: Given a block of `data + ec` bytes (with possible errors), correct the data bytes.

Algorithm (from `zxing ReedSolomonDecoder.cpp`):

1. **Syndrome calculation**: evaluate the received polynomial at α^0, α^1, …, α^(2t−1).
   If all zero → no errors.

2. **Euclidean algorithm**: find error locator polynomial σ(x) and error evaluator ω(x)
   such that `σ(x) * S(x) ≡ ω(x) mod x^(2t)` with `deg(ω) < deg(σ) ≤ t`.

3. **Find error locations**: brute-force search for roots of σ(x) over GF(256).
   For each root α^i, the error position is `inverse(α^i)` = α^(255−i).

4. **Find error magnitudes** using Forney's formula:
   `e_i = ω(α^(−j)) / σ'(α^(−j))`  (with generator base=0 adjustment).

5. **Correct**: XOR error magnitudes at the identified positions.

6. **Validate**: re-calculate syndromes; they should all be zero.

7. Return corrected data bytes.

**Tests** (36 tests, 0.11s):

- GF256Poly: construct, normalize, evaluate, add, multiply, divide (15 tests)
- Generator polynomial: building, roots check, leading coefficient (4 tests)
- RS decode — no errors: identity for various sizes (3 tests)
- RS decode — with errors: 1 error, t errors, random up to t, EC-only errors, mixed data+EC errors (5 tests)
- RS decode — too many errors: t+1 errors, all bytes corrupted (2 tests)
- Syndrome calculation stand-alone (2 tests)
- Euclidean algorithm stand-alone (2 tests)
- Forney formula / error locations (1 test)
- Round-trip encode→corrupt→decode for various params (1 test)
- Integration with qrcode library (1 test, placeholder)

**API**:

```python
def rs_decode(received: list[int], num_ec: int) -> list[int] | None
```

Takes a list of byte values (data + EC), num_ec = number of EC codewords.
Returns corrected byte list or None if uncorrectable.

**Implementation notes**:
- `GF256Poly` class: immutable polynomial over GF(256) with add, multiply, multiply_by_scalar, evaluate_at.
- Generator base = 0 (QR code convention).
- Euclidean algorithm: port of zxing-cpp `RunEuclideanAlgorithm()` with explicit quotient/remainder via `_poly_divide()`.
- Error position: `message_len - 1 - GF256.log(error_location)` matching zxing.
- Forney denominator: product of `(1 - errorLocations[j] * xiInverse)` over all j ≠ i.
- Re-validation step ensures corrected codeword passes syndrome check.

**Design decisions**:
- Polynomial division via `_poly_divide()` returns (quotient, remainder) as new instances — cleaner than the zxing mutating style.
- Syndrome return: `None` means "all zero" (no errors) to disambiguate from a valid `[0, 0, ...]` list.

**References**:
- `zxing ReedSolomonDecoder.cpp`
- `zxing GenericGF.h/.cpp`
- `zxing GenericGFPoly.h/.cpp` (polynomial arithmetic — we implement a minimal subset)

---

### Phase 7 — Bit Stream Decoding ✅ COMPLETED

**Status**: Done.  32 tests pass (0.06s).  `BitBuffer` bit reader + `decode_bitstream()`
for numeric, alphanumeric, and byte modes.

**Files**: `src/qr_reader/decoder/bitstream.py`, `tests/decoder/test_bitstream.py`,
`tests/decoder/conftest.py`

**What**: Decode the corrected data byte stream into a text string.

1. **Bit reader**: wrap the byte array, provide `read_bits(n)`, `peek_bits(n)`, `available()`.

2. **Segment loop**:
   - Read 4-bit mode indicator.
   - If terminator (0x0) or remaining bits < 4 and all zero → stop.
   - Read character count (variable bits depending on mode + version).
   - Decode segment:
     - **Numeric**: groups of 3 digits (10 bits), then 2 digits (7 bits), then 1 digit (4 bits).
     - **Alphanumeric**: groups of 2 chars (11 bits), then 1 char (6 bits).
       Use the alphanumeric table (digits 0–9, A–Z, space, $%*+-./:).
     - **Byte**: read 8 bits per character.

3. Return the concatenated string.

**Tests** (32 tests, 0.06s):

- BitBuffer: available, read_all, read_multibyte, zero bits, past end, peek, cross-byte-boundary, empty (8 tests)
- Numeric: 10/11/12 digits, single, two, three digits (6 tests)
- Alphanumeric: short string, space, charset subset, single, pair, odd count, URL-like (7 tests)
- Byte: ASCII, lowercase, punctuation, single char (4 tests)
- Mixed boundary, edge cases (unrecognized mode, short data), EC codeword counts, version count bits (7 tests)

**Handover notes**:
- Tests use v1 QR codes only (single EC block), extracting raw data codewords via
  `qr.data_cache[:data_codewords(version, ecl)]`. For multi-block codes the data is
  interleaved and would need Phase 5 (de-interleaving) applied first.
- `decode_bitstream(data_bytes, version)` expects *corrected* data codewords as input
  (after RS error correction from Phase 6).
- The `BitBuffer.read_bits()` gracefully clamps when reading past the end of data —
  this handles padding/trailing bits safely.
- `conftest.py` provides `make_qr_bitmatrix(content, version, ecl, mask)` shared helper.
- Unrecognised mode indicators raise `ValueError`.

**API**:

```python
def decode_bitstream(data_bytes: list[int], version: int) -> str
def data_codewords(version: int, ecl: str) -> int
class BitBuffer(data: list[int])
```

**References**:
- `zxing QRDecoder.cpp: DecodeBitStream()` (lines 230–320)
- `zxing QRCodecMode.cpp: CharacterCountBits()`
- Spec Table 5 (alphanumeric encoding).

---

### Phase 8 — Top-level Decoder ✅ COMPLETED

**Status**: Done. All 85 tests pass (400 total across decoder suite). `decode(matrix)` orchestrates the full pipeline end-to-end.

**Files**: `src/qr_reader/decoder/decoder.py`, `tests/decoder/test_decoder.py`

**What**: Orchestrate all phases.

```python
def decode(matrix: np.ndarray) -> str:
    # matrix: 2D bool (True=dark) of size (17+4*v) × (17+4*v)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    version = (matrix.shape[0] - 17) // 4
    assert 1 <= version <= 40

    # 1. Format info → ECL, mask
    mask_idx, ecl = decode_format_info(matrix)

    # 2. Version info (v >= 7) — cross-check
    if version >= 7:
        version2 = decode_version_info(matrix)
        assert version == version2

    # 3. Build function pattern mask
    fn_mask = build_function_pattern(version)

    # 4. Extract codewords (unmask + zigzag)
    raw = extract_codewords(matrix, fn_mask, mask_idx)

    # 5. De-interleave into data blocks
    blocks = get_data_blocks(raw, version, ecl)

    # 6. RS error correction
    corrected_data = []
    for data_bytes, ec_bytes in blocks:
        corrected = rs_correct(data_bytes, ec_bytes)
        corrected_data.extend(corrected)

    # 7. Decode bit stream
    text = decode_bitstream(bytes(corrected_data), version)
    return text
```

**Tests** (integration):

- Round-trip: encode text with `qrcode` library → extract bit matrix → decode → verify match.
- Test all version/ECL combinations (sampled — not all 160).
- Test with all 8 masks.
- Test numeric, alphanumeric, and byte modes.
- Test v1–v6 (no version info) and v7+ (with version info).
- Test error correction: introduce a few flipped bits → should still decode.

---

## 5. Test infrastructure

All tests live under `tests/decoder/`.  We use two Python test dependencies:

1. **`qrcode`** (already in `pyproject.toml`): generate QR codes programmatically with
   `qrcode.QRCode()`.  We can extract the bit matrix from its `modules` attribute after
   calling `make()`.

2. **`opencv-python`** (already in `pyproject.toml`): only for integration comparison tests.
   Generate QR images from `qrcode`, then decode with OpenCV to cross-check.

For speed, **unit tests avoid OpenCV entirely** — they work at the bit-matrix or byte level.

### Test plan per file

| Test file | What it tests | Dependencies |
|-----------|---------------|-------------|
| `tests/decoder/test_gf.py` | GF(256) arithmetic | none |
| `tests/decoder/test_tables.py` | Table consistency | none |
| `tests/decoder/test_format_info.py` | Format BCH decode | `qrcode` |
| `tests/decoder/test_version_info.py` | Version BCH decode | `qrcode` |
| `tests/decoder/test_codeword_extract.py` | Zigzag + unmask | `qrcode` |
| `tests/decoder/test_data_block.py` | De-interleaving | `qrcode` |
| `tests/decoder/test_rs.py` | RS error correction | `qrcode` |
| `tests/decoder/test_bitstream.py` | Bit stream → text | `qrcode` |
| `tests/decoder/test_decoder.py` | Full pipeline | `qrcode` |

Shared test helper in `tests/decoder/conftest.py` or `tests/decoder/helpers.py`:

```python
def make_qr_bitmatrix(content: str, version: int, ecl: str, mask: int) -> np.ndarray:
    """Return a bool 2D array of the QR modules (True = dark)."""
    qr = qrcode.QRCode(version=version, error_correction=...)
    qr.add_data(content)
    qr.make()  # uses best mask or force mask
    modules = np.array(qr.modules, dtype=bool)  # True = black
    return modules
```

**Total test runtime target**: < 10 seconds.  This means:
- Use tiny QR codes (v1–v3) for most tests.
- Avoid `opencv` in unit tests.
- Keep RS test matrices small.
- Use `pytest` fixtures with module scope where possible.

---

## 6. Implementation order & dependencies

```
Phase 0 (GF arithmetic)           ← no deps
Phase 1 (Tables)                  ← no deps
Phase 2 (Format BCH decode)       ← Phase 1
Phase 3 (Version BCH decode)      ← Phase 1
Phase 4 (Codeword extraction)     ← Phase 1, Phase 2
Phase 5 (Data block de-interleave) ← Phase 1
Phase 6 (Reed-Solomon)            ← Phase 0
Phase 7 (Bit stream)              ← Phase 1
Phase 8 (Top-level decoder)       ← Phases 2–7
```

Phases 1–7 are independent of each other (except for table dependencies) and can be
implemented in parallel.

---

## 7. File tree

```
src/qr_reader/decoder/
    __init__.py
    gf.py               # GF(256) arithmetic
    tables.py           # All spec tables (version, ECL, alignment, masks, mode counts)
    format_info.py      # Format BCH decode → (ECL, mask_idx)
    version_info.py     # Version BCH decode → version
    function_pattern.py # Build function module mask for a version
    codeword_extractor.py  # Zigzag extraction + unmasking
    data_block.py       # De-interleave raw codewords into blocks
    rs.py               # Reed-Solomon error correction
    bitstream.py        # Bit reader + mode/segment decoding
    decoder.py          # Top-level decode(matrix) → str

tests/decoder/
    __init__.py
    conftest.py         # Shared fixtures (qrcode generation helpers)
    test_gf.py
    test_tables.py
    test_format_info.py
    test_version_info.py
    test_codeword_extract.py
    test_data_block.py
    test_rs.py
    test_bitstream.py
    test_decoder.py
```

---

## 8. Key design decisions

1. **No parameterized fallbacks**: we do *not* try all 8 masks or re-derive format info from multiple copies with fallbacks.  The happy path: read format info, get mask+ECL, proceed.  If format info is invalid → fail.  (As requested: "avoid unnecessary fallbacks, focus on the happy path.")

2. **BCH via exhaustive match**, not algebraic decode.  Both zxing-cpp and nayuki do this — there are only 32 (format) or 34 (version) valid codewords, so a Hamming-distance scan is simpler and handles errors up to 3 bits.

3. **RS error correction: brute-force error location search**, not Chien search.  GF(256) is only 255 non-zero elements, a full scan is ~255 evaluations per block — trivially fast for our purposes.  (zxing-cpp does this too, see `FindErrorLocations`.)

4. **Data types**: bit matrix as `np.ndarray(dtype=bool)` (True = dark).  Bytes as `bytes` or `bytearray`.  Text output as `str`.

5. **Encoding**: byte-mode decoded bytes are returned as-is (`latin-1` decoding).  We do not attempt character-set detection (no ECI support).

6. **Testing uses `qrcode` library** (Python) to generate known QR codes.  We extract `qr.modules` (a list of lists of bool) and convert to numpy.  This avoids OpenCV entirely for unit tests.
