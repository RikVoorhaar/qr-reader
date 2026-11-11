"""QR sampling & thresholding: convert a perspective-distorted grayscale
image of a QR code into a clean boolean grid via supersampling and adaptive
thresholding.

All functions assume a grayscale ``uint8`` image.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from qr_reader.detector.homography import project_points

# Sub-cell sample offsets within a module, and the 3×3 weight kernel.
_OFFSETS = np.array([0.25, 0.5, 0.75], dtype=np.float64)
_DY, _DX = np.meshgrid(_OFFSETS, _OFFSETS, indexing="ij")
_WEIGHTS = np.ones((3, 3), dtype=np.float64)
_WEIGHTS[1, 1] = 2.0  # centre weight 2


def _sample_points(
    image_f64: np.ndarray, H: np.ndarray, grid_xy: np.ndarray
) -> np.ndarray:
    """Bilinearly sample *image_f64* at many QR-grid ``(x, y)`` points.

    Args:
        image_f64: Grayscale image already cast to ``float64``.
        H: 3×3 homography QR-grid ``(x, y)`` → image ``(x, y)``.
        grid_xy: ``(n, 2)`` array of QR-grid ``(x, y)`` coordinates.

    Returns:
        ``(n,)`` float64 array of sampled values.
    """
    img_xy = project_points(H, grid_xy)  # (n, 2) in (x, y) order
    # map_coordinates expects (row, col) order
    coords = np.stack([img_xy[:, 1], img_xy[:, 0]])  # (2, n): (y, x)
    return map_coordinates(image_f64, coords, order=1, mode="nearest")


def _supersample_cells(
    image_f64: np.ndarray, H: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`supersample_cell` over many cells.

    Args:
        image_f64: Grayscale image already cast to ``float64``.
        H: 3×3 homography QR-grid ``(x, y)`` → image ``(x, y)``.
        rows, cols: 1-D integer arrays of cell rows / cols, equal length.

    Returns:
        ``(len(rows), 3, 3)`` float64 array; entry ``[k, i, j]`` is the value
        at ``(row + dy[i], col + dx[j])``.
    """
    n = rows.shape[0]
    x = cols[:, None, None] + _DX[None, :, :]  # (n, 3, 3)
    y = rows[:, None, None] + _DY[None, :, :]  # (n, 3, 3)
    grid_xy = np.stack([x, y], axis=-1).reshape(-1, 2)  # (n*9, 2)
    return _sample_points(image_f64, H, grid_xy).reshape(n, 3, 3)


def supersample_cell(
    image: np.ndarray, H: np.ndarray, row: int, col: int
) -> np.ndarray:
    """Sample a 3×3 neighborhood in the grayscale *image* for QR cell *(row, col)*.

    The nine sub-cell positions are spaced at 0.25, 0.5, 0.75 offsets from
    the cell origin, projected through the homography *H* and sampled with
    bilinear interpolation.

    Args:
        image: Grayscale ``uint8`` image, shape ``(H_img, W_img)``.
        H: 3×3 homography mapping QR-grid ``(x, y)`` → image ``(x, y)``.
        row: QR module row (0..N−1).
        col: QR module col (0..N−1).

    Returns:
        ``np.ndarray`` of shape ``(3, 3)``, dtype ``float64`` — the 9 sampled
        values.  Index ``[1, 1]`` corresponds to the exact cell centre
        ``(col+0.5, row+0.5)``.
    """
    return _supersample_cells(
        image.astype(np.float64),
        H,
        np.array([row], dtype=np.float64),
        np.array([col], dtype=np.float64),
    )[0]


def finder_pattern_known_cells(N: int) -> tuple[list, list]:
    """Return ``(black_cells, white_cells)`` — the known cells from the
    three finder patterns (TL, TR, BL).

    Each element is a ``list[tuple[int, int]]`` of ``(row, col)`` positions.

    The three 7×7 finder patterns are located at:
      - TL: rows 0..6, cols 0..6
      - TR: rows 0..6, cols N−7..N−1
      - BL: rows N−7..N−1, cols 0..6

    Within each pattern the known cells are:

      ================  =============================  =====
      Ring               Coordinates                   Value
      ================  =============================  =====
      Outer border       row∈{0,6} or col∈{0,6}        Black
      White ring         rows 1–5, cols 1–5 (5×5)     White
      Inner 3×3          rows 2–4, cols 2–4            Black
      ================  =============================  =====

    Black count per pattern: 24 (border) + 9 (inner) = 33.
    White count per pattern: 25 (5×5) − 9 (inner) = 16.
    Total: 99 black, 48 white cells (duplicates not deduplicated).
    """
    # Three pattern origins: (row_start, col_start)
    origins = [
        (0, 0),  # TL
        (0, N - 7),  # TR
        (N - 7, 0),  # BL
    ]

    black: list[tuple[int, int]] = []
    white: list[tuple[int, int]] = []

    for r0, c0 in origins:
        # White: the 5×5 at offset (1,1) — but we subtract the inner 3×3
        for dr in range(1, 6):
            for dc in range(1, 6):
                if 2 <= dr <= 4 and 2 <= dc <= 4:
                    continue  # inner 3×3 → black
                white.append((r0 + dr, c0 + dc))

        # Black outer border
        for dc in range(7):
            black.append((r0, c0 + dc))  # top row
            black.append((r0 + 6, c0 + dc))  # bottom row
        for dr in range(1, 6):
            black.append((r0 + dr, c0))  # left col
            black.append((r0 + dr, c0 + 6))  # right col

        # Black inner 3×3
        for dr in range(2, 5):
            for dc in range(2, 5):
                black.append((r0 + dr, c0 + dc))

    return black, white


def compute_adaptive_threshold(image: np.ndarray, H: np.ndarray, N: int) -> float:
    """Compute a single global threshold separating black from white modules.

    Uses the **centre** sub-pixel (index ``[1, 1]``) of each known finder-pattern
    cell as ground truth, then returns the midpoint of the two medians:

        ``threshold = (median(black_vals) + median(white_vals)) / 2.0``.

    Args:
        image: Grayscale ``uint8`` image.
        H: 3×3 homography QR-grid ``(x, y)`` → image ``(x, y)``.
        N: QR module count (e.g. 21 for version 1).

    Returns:
        A ``float`` threshold value.
    """
    black_cells, white_cells = finder_pattern_known_cells(N)
    image_f64 = image.astype(np.float64)

    def _centre_values(cells: list[tuple[int, int]]) -> np.ndarray:
        rows = np.array([r for r, _ in cells], dtype=np.float64)
        cols = np.array([c for _, c in cells], dtype=np.float64)
        grid_xy = np.column_stack([cols + 0.5, rows + 0.5])
        return _sample_points(image_f64, H, grid_xy)

    black_vals = _centre_values(black_cells)
    white_vals = _centre_values(white_cells)

    threshold = (np.median(black_vals) + np.median(white_vals)) / 2.0
    return float(threshold)


def sample_qr_bits(
    image: np.ndarray,
    H: np.ndarray,
    N: int,
    threshold: float | None = None,
) -> np.ndarray:
    """Sample every QR module and return a decoder-ready bit matrix.

    Each cell is sampled with a 3×3 supersampling neighbourhood.  The
    weighted majority vote uses centre weight 2 and surrounding weight 1
    (total weight = 10, majority ≥ 5 white votes).

    The sampled light-module grid is converted at the API boundary so the
    returned matrix matches ``qr_reader.decoder.decode``: ``True`` = dark/black.

    Args:
        image: Grayscale ``uint8`` image.
        H: 3×3 homography QR-grid ``(x, y)`` → image ``(x, y)``.
        N: QR module count.
        threshold: Optional pre-computed threshold.  If ``None``,
            ``compute_adaptive_threshold`` is called.

    Returns:
        ``np.ndarray`` of shape ``(N, N)``, dtype ``bool``.
    """
    if threshold is None:
        threshold = compute_adaptive_threshold(image, H, N)

    # Cells flattened in C order over (row, col), sampled in one batch.
    rows, cols = np.meshgrid(
        np.arange(N, dtype=np.float64),
        np.arange(N, dtype=np.float64),
        indexing="ij",
    )
    vals = _supersample_cells(
        image.astype(np.float64), H, rows.ravel(), cols.ravel()
    ).reshape(N, N, 3, 3)

    white_votes = np.sum((vals > threshold) * _WEIGHTS, axis=(-2, -1))
    light_modules = white_votes >= 5.0

    return (~light_modules).T
