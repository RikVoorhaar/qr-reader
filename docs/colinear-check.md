# QR Code Landmark Identification and Homography Plan

This plan outlines the steps to extract and associate finder patterns, identify the top-left corner, extract all landmarks, and map them to the QR code grid.

## 1. Geometric Primitives and Colinearity Check
Create a module (e.g., `src/qr_reader/geometry.py`) to handle geometric calculations.
- **Line Segments**: Define segments using endpoints $(p_1, p_2)$ and $(q_1, q_2)$.
- **Angular Distance**: Compute the angle between two line segments.
- **Point-Line Distance**: Compute the orthogonal distance from a point to an infinite line defined by a segment.
- **Max Offset**: For segments P and Q, compute `max(d(q1,P), d(q2,P), d(p1,Q), d(p2,Q)) / L`, where `L` is the distance between the midpoints of P and Q.
- **Intersection Test**: Check if two line segments (or their bounding boxes/polygons) intersect to avoid pathological cases where finder patterns overlap.
- **Area Calculation**: Move the `area` computation function from `dev3.py` into this module.
- *Test*: Write unit tests for all geometric primitives.

## 2. Finder Pattern Pair Analysis
Create a module (e.g., `src/qr_reader/finder_patterns.py`) to analyze pairs of finder patterns.
- **Data Structure**: Create a class/dataclass to represent a Finder Pattern (e.g., grouping its outer and inner corners).
- **Outer Corners**: Use the area calculation to filter and keep only the outer corners of each finder pattern (the largest area).
- **Association Score**: For each pair of finder patterns, compute an association score.
- **Colinear Segments Detection**: 
  - For each pair of finder patterns, compute angular distances between all pairs of their outer segments (4 segments per pattern, 16 pairs total).
  - Find roughly parallel segments (should be 8 pairs).
  - Compute the offset for these parallel segments. We should find exactly two pairs with low offset, indicating alignment.
  - Record the pair of colinear segments for each valid association.
- *Integration*: Run this logic on the example in `dev3.py` to verify it makes sense visually and logically.

## 3. Finder Pattern Triplet Identification
Extend the finder pattern module to identify the triplet of finder patterns that make up a QR code.
- **Find Triplet**: Look for situations where pattern A is associated with B, and B is associated with C, but A is NOT colinear/associated with C. 
- **Identify Top-Left**: Ensure that the colinear segments aligning A and B are orthogonal (or at least different) from the colinear segments aligning B and C. In this case, B is the top-left corner.
- *Integration*: Test finding the triplet and identifying the top-left corner in `dev3.py` on the example image.

## 4. Landmark Extraction
Create a module (or extend existing) to extract the exact landmarks and prepare them for homography.
- **Extract Landmarks**: From the identified triplet, extract the 24 landmarks (4 outer + 4 inner corners for each of the 3 finder patterns).
- *Integration*: (Skipping homography step per instructions, as it requires knowing the QR version).

## Testable Chunks
- **Chunk 1**: Implement `geometry.py` with angular distance, point-line distance, max offset, intersection test, and area. Write full `pytest` unit tests for these.
- **Chunk 2**: Implement finder pattern data structures and the pairwise comparison logic (extract outer corners, find colinear segments). Test in `dev3.py`.
- **Chunk 3**: Implement the triplet finding and top-left identification logic. Test in `dev3.py`.
