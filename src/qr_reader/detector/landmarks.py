"""Landmarks: corner ordering, named image landmarks, canonical grid coordinates,
and colinear quadruple definitions.

Coordinate convention:
  - All point arrays are (row, col) = (y, x) internally.
  - Conversion to (x, y) happens only at the homography/decode boundary.
"""

from dataclasses import dataclass

import numpy as np

from qr_reader.detector.finder_pattern import FinderPattern, Triplet

# ---------------------------------------------------------------------------
# Step B — local basis + corner ordering
# ---------------------------------------------------------------------------


def local_basis(
    triplet: Triplet, fps: list[FinderPattern]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (right, down) unit vectors in (row, col) space from a triplet.

    right = normalize(center_TR - center_TL)
    down  = normalize(center_BL - center_TL)
    """
    fp_map = {fp.cluster_idx: fp for fp in fps}

    tl = fp_map[triplet.top_left_idx]
    tr = fp_map[triplet.top_right_idx]
    bl = fp_map[triplet.bottom_left_idx]

    center_tl = tl.outer_corners.mean(axis=0)
    center_tr = tr.outer_corners.mean(axis=0)
    center_bl = bl.outer_corners.mean(axis=0)

    right = center_tr - center_tl
    right = right / (np.linalg.norm(right) + 1e-12)

    down = center_bl - center_tl
    down = down / (np.linalg.norm(down) + 1e-12)

    return right, down


def order_square_corners(
    points4: np.ndarray, right: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Order 4 corners of a square into [TL, BL, BR, TR] using a local basis.

    points4: (4, 2) array in (row, col).
    right, down: unit vectors in (row, col) from local_basis.

    Each point is projected onto right (→ r) and down (→ d), then assigned by
    sign quadrant:  (r<0, d<0) → TL=0  (r<0, d>0) → BL=1
                    (r>0, d>0) → BR=2  (r>0, d<0) → TR=3

    Falls back to atan2(d, r) sort if quadrant assignment is degenerate.
    """
    centroid = points4.mean(axis=0)
    centered = points4 - centroid
    r = centered @ right  # projection onto the right vector
    d = centered @ down  # projection onto the down vector

    ordered = np.empty_like(points4)
    assigned = np.zeros(4, dtype=bool)

    # Try quadrant assignment
    # (r<0, d<0) → TL(0),  (r<0, d>0) → BL(1),
    # (r>0, d>0) → BR(2),  (r>0, d<0) → TR(3)
    quadrant_map = {
        (-1, -1): 0,  # TL
        (-1, 1): 1,  # BL
        (1, 1): 2,  # BR
        (1, -1): 3,  # TR
    }

    ok = True
    for i in range(4):
        qr = -1 if r[i] < 0 else 1
        qd = -1 if d[i] < 0 else 1
        idx = quadrant_map.get((qr, qd))
        if idx is None:
            ok = False
            break
        if assigned[idx]:
            ok = False
            break
        ordered[idx] = points4[i]
        assigned[idx] = True

    if ok and assigned.all():
        return ordered

    # Fallback: sort by angle around centroid in (r, d) space
    angles = np.arctan2(d, r)
    sort_idx = np.argsort(angles)
    # We want: TL ≈ (-r, -d) → angle ≈ -135° → position 0;
    # after sort we need to shift so that the most-negative-angle quadrant is TL.
    # Simple approach: pick the point with r<0 and d<0 (most TL-like) as anchor.
    tl_candidates = np.where((r < 0) & (d < 0))[0]
    if len(tl_candidates) == 1:
        anchor = tl_candidates[0]
        # Rotate sorted order so anchor is first
        pos = np.where(sort_idx == anchor)[0][0]
        sort_idx = np.roll(sort_idx, -pos)
    return points4[sort_idx]


# ---------------------------------------------------------------------------
# Named landmarks
# ---------------------------------------------------------------------------


@dataclass
class NamedLandmarks:
    """Six corner sets, each (4, 2) in (row, col), ordered [TL, BL, BR, TR].

    A, B — from the top-left finder pattern (outer, inner).
    C, D — from the top-right finder pattern (outer, inner).
    E, F — from the bottom-left finder pattern (outer, inner).
    """

    A: np.ndarray  # TL outer
    B: np.ndarray | None  # TL inner (may be None if only one quad detected)
    C: np.ndarray  # TR outer
    D: np.ndarray | None  # TR inner
    E: np.ndarray  # BL outer
    F: np.ndarray | None  # BL inner


def build_named_landmarks(triplet: Triplet, fps: list[FinderPattern]) -> NamedLandmarks:
    """Build NamedLandmarks from a triplet of finder patterns."""
    fp_map = {fp.cluster_idx: fp for fp in fps}

    tl_fp = fp_map[triplet.top_left_idx]
    tr_fp = fp_map[triplet.top_right_idx]
    bl_fp = fp_map[triplet.bottom_left_idx]

    right, down = local_basis(triplet, fps)

    A = order_square_corners(tl_fp.outer_corners, right, down)
    B = (
        order_square_corners(tl_fp.inner_corners, right, down)
        if tl_fp.inner_corners is not None
        else None
    )
    C = order_square_corners(tr_fp.outer_corners, right, down)
    D = (
        order_square_corners(tr_fp.inner_corners, right, down)
        if tr_fp.inner_corners is not None
        else None
    )
    E = order_square_corners(bl_fp.outer_corners, right, down)
    F = (
        order_square_corners(bl_fp.inner_corners, right, down)
        if bl_fp.inner_corners is not None
        else None
    )

    return NamedLandmarks(A=A, B=B, C=C, D=D, E=E, F=F)


# ---------------------------------------------------------------------------
# Step C — canonical grid coordinates + colinear quadruples
# ---------------------------------------------------------------------------


def canonical_grid_landmarks(N: int) -> NamedLandmarks:
    """Return NamedLandmarks with canonical (x, y) grid coordinates for version N.

    Grid coords use (x, y) convention (x right, y down), matching the QR spec.
    Outer squares span 0..7; inner squares (white ring at offset 1) span 1..6.
    """
    # Outer squares (row, col) = (y, x)
    A = np.array(
        [
            [0, 0],
            [7, 0],
            [7, 7],
            [0, 7],  # TL, BL, BR, TR
        ],
        dtype=np.float64,
    )
    C = np.array(
        [
            [0, N - 7],
            [7, N - 7],
            [7, N],
            [0, N],
        ],
        dtype=np.float64,
    )
    E = np.array(
        [
            [N - 7, 0],
            [N, 0],
            [N, 7],
            [N - 7, 7],
        ],
        dtype=np.float64,
    )

    # Inner squares (white ring at offset 1, coords 1..6)
    B = np.array(
        [
            [1, 1],
            [6, 1],
            [6, 6],
            [1, 6],
        ],
        dtype=np.float64,
    )
    D = np.array(
        [
            [1, N - 6],
            [6, N - 6],
            [6, N - 1],
            [1, N - 1],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [N - 6, 1],
            [N - 1, 1],
            [N - 1, 6],
            [N - 6, 6],
        ],
        dtype=np.float64,
    )

    return NamedLandmarks(A=A, B=B, C=C, D=D, E=E, F=F)


# ---------------------------------------------------------------------------
# Colinear quadruple definitions
# ---------------------------------------------------------------------------


@dataclass
class Quadruple:
    """Four points (in order) that are colinear in the grid/image."""

    points: np.ndarray  # (4, 2) in (row, col) from the landmark sets
    type: str  # "outer" or "inner"
    label: str  # e.g. "A0-A1-E0-E1"


def get_colinear_quadruples(landmarks: NamedLandmarks) -> list[Quadruple]:
    """Return the 8 colinear quadruples from NamedLandmarks.

    Outer quadruples (A, C, E):
      left edge:   (A0, A1, E0, E1)
      left edge reverse: (A3, A2, E3, E2)
      top edge:    (A0, A3, C0, C3)
      top edge reverse:  (A1, A2, C1, C2)

    Inner quadruples (B, D, F) — may be None if inner corners missing:
      left edge:   (B0, B1, F0, F1)
      left edge reverse: (B3, B2, F3, F2)
      top edge:    (B0, B3, D0, D3)
      top edge reverse:  (B1, B2, D1, D2)

    Point indices follow the canonical [TL,BL,BR,TR] = [0,1,2,3] order.
    """
    quads = []

    # --- Outer ---
    # left edge: A[0]=TL, A[1]=BL, E[0]=TL, E[1]=BL  (these are on the left column)
    quads.append(
        Quadruple(
            points=np.array(
                [landmarks.A[0], landmarks.A[1], landmarks.E[0], landmarks.E[1]]
            ),
            type="outer",
            label="A0-A1-E0-E1",
        )
    )
    # left edge reverse: A[3]=TR, A[2]=BR, E[3]=TR, E[2]=BR
    quads.append(
        Quadruple(
            points=np.array(
                [landmarks.A[3], landmarks.A[2], landmarks.E[3], landmarks.E[2]]
            ),
            type="outer",
            label="A3-A2-E3-E2",
        )
    )
    # top edge: A[0]=TL, A[3]=TR, C[0]=TL, C[3]=TR
    quads.append(
        Quadruple(
            points=np.array(
                [landmarks.A[0], landmarks.A[3], landmarks.C[0], landmarks.C[3]]
            ),
            type="outer",
            label="A0-A3-C0-C3",
        )
    )
    # top edge reverse: A[1]=BL, A[2]=BR, C[1]=BL, C[2]=BR
    quads.append(
        Quadruple(
            points=np.array(
                [landmarks.A[1], landmarks.A[2], landmarks.C[1], landmarks.C[2]]
            ),
            type="outer",
            label="A1-A2-C1-C2",
        )
    )

    # --- Inner (only if available) ---
    if landmarks.B is not None and landmarks.D is not None and landmarks.F is not None:
        quads.append(
            Quadruple(
                points=np.array(
                    [landmarks.B[0], landmarks.B[1], landmarks.F[0], landmarks.F[1]]
                ),
                type="inner",
                label="B0-B1-F0-F1",
            )
        )
        quads.append(
            Quadruple(
                points=np.array(
                    [landmarks.B[3], landmarks.B[2], landmarks.F[3], landmarks.F[2]]
                ),
                type="inner",
                label="B3-B2-F3-F2",
            )
        )
        quads.append(
            Quadruple(
                points=np.array(
                    [landmarks.B[0], landmarks.B[3], landmarks.D[0], landmarks.D[3]]
                ),
                type="inner",
                label="B0-B3-D0-D3",
            )
        )
        quads.append(
            Quadruple(
                points=np.array(
                    [landmarks.B[1], landmarks.B[2], landmarks.D[1], landmarks.D[2]]
                ),
                type="inner",
                label="B1-B2-D1-D2",
            )
        )

    return quads
