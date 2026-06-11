# Report: `dev3.py` sampling/decoding failure

## Summary

`src/qr_reader/dev3.py` now uses the project sampler and decoder instead of OpenCV. The pipeline reaches the final sampled QR grid, but decoding fails inside the project decoder during Reed-Solomon correction.

The sampled grid appears visually valid and can be decoded by a phone scanner, but it is not in the exact matrix orientation/layout expected by the project decoder. Visually, the sampled grid appears to be transposed from the expected QR layout: a reflection across the main diagonal. Phone QR readers are tolerant of this kind of orientation issue; the project decoder expects a canonical matrix.

## How to reproduce

From the repository root:

```bash
python -m qr_reader.dev3
```

Observed failure:

```text
Sampled grid shape: (29, 29), white fraction: 0.485
✗ Decode failed from sampled bits: RS error correction failed for version 3 ECL H (block with 13 data + 22 EC bytes)

AssertionError: Decode failed for V=3
```

Earlier stages appear to succeed:

- Finder patterns are detected.
- Version is inferred correctly as `V=3`, so `N=29`.
- Homography is estimated.
- Sampling produces a `29x29` grid.
- The final plotted “Sampled QR bits” image is scannable by a phone.

## What is going wrong

The decoder gets far enough to read format information, infer error correction/mask data, extract codewords, and deinterleave blocks. It then fails here:

```text
RS error correction failed for version 3 ECL H
```

That means the project decoder is receiving a QR matrix that is close enough to look like a QR code, but not arranged in the canonical row/column orientation expected by the decoder.

In other words: this is probably not a Reed-Solomon bug directly. Reed-Solomon is failing because the extracted codewords are wrong.

## Probable cause

The likely cause is a coordinate convention mismatch between sampling and decoding.

The sampler returns:

```python
bits[row, col]
```

with:

```text
True = white/light
False = black/dark
```

The decoder expects:

```python
matrix[row, col]
```

with:

```text
True = dark/black
False = light/white
```

`dev3.py` currently handles polarity by doing:

```python
matrix = ~bits
```

That inversion is necessary, but it is probably not sufficient. The sampled grid visually appears to be transposed relative to the decoder’s expected canonical QR matrix — effectively a reflection along the main diagonal.

The likely minimal fix is therefore at the sampler-to-decoder boundary:

```python
matrix = (~bits).T
```

instead of:

```python
matrix = ~bits
```

Then call:

```python
decode(matrix)
```

This should still be verified against a known-good matrix, but based on the visual symptom this is probably a simple row/column transpose issue rather than a decoder or Reed-Solomon problem.

## Why OpenCV previously worked

The old path rebuilt an image from sampled bits and then let OpenCV decode it. OpenCV performs QR detection and orientation normalization internally.

The project decoder does not do that. It assumes the input matrix is already in canonical QR orientation:

- top-left finder pattern at top-left,
- top-right finder pattern at top-right,
- bottom-left finder pattern at bottom-left,
- rows and columns matching QR spec order.

Therefore, OpenCV could hide an orientation issue that the project decoder exposes.

## Files likely involved

- `src/qr_reader/dev3.py`
- `src/qr_reader/sample.py`
- `src/qr_reader/homography.py`
- `src/qr_reader/landmarks.py`
- `src/qr_reader/decoder/decoder.py`
- `src/qr_reader/decoder/codeword_extractor.py`

## Recommended next step

Do not start by changing Reed-Solomon or the bitstream decoder.

First, verify the matrix convention at the sampler/decoder boundary:

```text
matrix[row, col] == canonical QR module at row, col
```

Then confirm whether `sample_qr_bits(...)` returns canonical orientation or a transposed orientation.

The likely fix is a small orientation normalization step between sampling and decoding. Based on the visual appearance, the expected transform is probably just transpose after polarity inversion:

```python
matrix = (~bits).T
```

This should be confirmed against a known-good matrix, but the issue appears to be a simple row/column transpose at the boundary.
