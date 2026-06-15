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

### Synthesis

**Synthetic Image**:
A composited image produced by the augmentation pipeline: a QR code (patch)
blended onto a natural background with controlled degradation.
_Avoid_: Augmented image, fake image, generated image

**Patch**:
The QR code image rendered with its quiet zone, in isolation, before compositing.
Always a square RGB uint8 array.
_Avoid_: QR image, QR bitmap, sticker

**Mask**:
A single-channel float32 image, same dimensions as the patch, with value 1.0
inside the patch rectangle and 0.0 outside. Used as the alpha channel for
compositing and as input to feathering.
_Avoid_: Alpha mask, blend mask

**Quiet Zone**:
The 4-module blank border around the QR code proper, required by the QR spec.
Included in the patch but excluded from QR corner ground truth.
_Avoid_: Margin, border, padding

**QR Corners**:
The 4 image-space [x, y] coordinates (TL, TR, BR, BL) of the QR code proper
(excluding quiet zone). Serves as ground truth for the detector.
_Avoid_: Patch corners, sticker corners, bounding-box corners

**Feathering**:
Gaussian blur applied to the mask boundary only, creating a smooth alpha
transition at the patch edge so it blends into the background.
_Avoid_: Edge softening, alpha blur

**Placement**:
Scaling and translating the augmented patch onto the background canvas such
that the full patch (including quiet zone) stays within the image bounds.
_Avoid_: Positioning, pasting

**Placement Scale**:
The uniform scale factor computed from `target_ppm` such that the QR code
modules in the final image have approximately the desired pixel density.
_Avoid_: Zoom factor, resize ratio

**Global Degradation**:
Post-composite image-wide augmentations applied to the entire synthetic image:
Gaussian blur, sensor noise, JPEG compression. Simulates camera pipeline
artifacts.
_Avoid_: Post-processing, image corruption, camera simulation

**Augmentation Config**:
A pydantic model capturing all tuneable parameters for the pipeline:
ppm ranges, jitter fraction, feather sigma, blur/noise/JPEG ranges,
and difficulty presets.
_Avoid_: Config dict, parameter set, settings

**Corner Jitter**:
Small random perturbation applied to each corner of the source quad before
computing the perspective homography. The primary mechanism for introducing
perspective distortion.
_Avoid_: Corner noise, corner perturbation, projective jitter
