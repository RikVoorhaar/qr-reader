# QR Reader

A pure-Python QR code reader that detects and decodes QR codes from images using
a two-stage pipeline: geometric detection followed by symbol decoding.

## Language

### Detection

**Finder Pattern**:
One of the three corner squares of a QR code. Detected as a 4-corner quadrilateral
with inner and outer concentric rings.
_Avoid_: Position detection pattern, eye, corner marker

**Association**:
A colinear alignment between two Finder Patterns, indicating they belong to the
same QR code.
_Avoid_: Pair, alignment, link

**Triplet**:
Three Finder Patterns forming an L-shape that identifies the top-left, top-right,
and bottom-left corners of a QR code.
_Avoid_: Corner set, pattern group

**Landmark**:
A named reference point (A–F) on the QR code grid used for homography estimation.
_Avoid_: Anchor point, reference point

**Alignment Pattern**:
A smaller 1:1:3:1:1 ratio pattern scattered across the QR code (versions ≥ 2)
used to locate candidate regions.
_Avoid_: Timing pattern, positioning mark

**Candidate Cluster**:
A merged group of overlapping Alignment Pattern candidates that likely belong to
the same QR code region.
_Avoid_: Region group, pattern cluster

**Cross-ratio**:
A projective invariant computed from four colinear points, used to estimate the
QR code version by comparing measured ratios against expected values.
_Avoid_: Ratio, projective measure

**Homography**:
A 3×3 matrix mapping grid coordinates to image coordinates, estimated via DLT,
refined with RANSAC and Levenberg-Marquardt optimization.
_Avoid_: Perspective transform, warp matrix

### Decoding

**Module**:
A single square cell of the QR code grid — the atomic unit of the symbol.
_Avoid_: Pixel, cell, element

**Codeword**:
An 8-bit unit of encoded data or error correction. Extracted from the module
matrix via zigzag unmasking.
_Avoid_: Byte, word, symbol

**ECL (Error Correction Level)**:
One of four recovery capacities: L (~7%), M (~15%), Q (~25%), H (~30%).
_Avoid_: Correction strength, redundancy level

**Data Block**:
A group of data codewords paired with error-correction codewords, de-interleaved
from the raw codeword stream.
_Avoid_: Segment, chunk

**Format Info**:
A 15-bit pattern encoding the ECL and mask pattern, read from fixed strips near
the finder patterns.
_Avoid_: Format bits, mask info

**Mask Pattern**:
One of eight XOR patterns applied to the data/EC modules to break up
problematic visual patterns. The mask is identified in the Format Info.
_Avoid_: XOR pattern, masking function

**Timing Pattern**:
Fixed alternating dark/light rows and columns used to determine module
size and pitch. Runs between finder patterns.
_Avoid_: Clock track, timing track

**Zigzag**:
The serpentine column-pair reading order used to extract codeword bits
from the module matrix, per the QR spec.
_Avoid_: Serpentine scan, swizzle

**Version Info**:
An 18-bit pattern encoding the QR version number (6 bits) and its complement,
present only in versions ≥ 7.
_Avoid_: Version bits
