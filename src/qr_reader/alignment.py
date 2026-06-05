"""Finding candidate alignment patterns in a binary QR code image."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def run_length_encoding(row: np.ndarray) -> list[tuple]:
    """Compute run-length encoding of a 1-D boolean array."""
    run_lengths = []
    current_run = 0
    current_value = row[0]
    for i in range(len(row)):
        if row[i] == current_value:
            current_run += 1
        else:
            run_lengths.append((current_value, current_run))
            current_value = row[i]
            current_run = 1
    return run_lengths


def find_alignment_patterns(img_binary: np.ndarray, max_error: float):
    """
    Find 1:1:3:1:1 ratio patterns in a 2-D binary image, scanning row-wise.

    Returns (rows, columns_all) where rows are the row indices of candidate patterns
    and columns_all is an (N, 6) array of the column indices marking the boundaries
    of the 5-segment pattern.
    """
    # Expected ratio 1:1:3:1:1 (white:black:white:black:white)
    expected = np.array([1, 1, 3, 1, 1], dtype=np.float64)
    expected = expected / expected.sum()
    log_expected = np.log(expected)

    rows, columns = np.where(np.diff(img_binary))

    row_changes = np.diff(rows) > 0
    run_lengths_smart = np.diff(columns).astype(np.float64)  # keep float for scoring below
    run_lengths_smart[row_changes] = 0

    seqs = sliding_window_view(run_lengths_smart, window_shape=5) + 1e-8
    scores = np.max(
        np.abs(np.log(seqs / np.sum(seqs, axis=1, keepdims=True)) - log_expected),
        axis=1,
    )

    (candidate_indices,) = np.where(scores < max_error)
    candidate_rows = rows[candidate_indices]

    indices_to_add = np.arange(6)
    candidate_indices_add = candidate_indices.reshape(-1, 1) + indices_to_add
    candidate_columns_all = columns[candidate_indices_add]
    return candidate_rows, candidate_columns_all


def find_alignment_patterns_2d(img_binary: np.ndarray, max_error: float):
    """
    Find alignment patterns by scanning both horizontally and vertically,
    keeping only those that have intersecting horizontal and vertical matches.

    Returns (rows, cols_all) for validated patterns.
    """
    # Horizontal scan
    rows_x, cols_x_all = find_alignment_patterns(img_binary, max_error)

    # Vertical scan at the x-centers of horizontal candidates
    x_values = (cols_x_all[:, 2] + cols_x_all[:, 3]) // 2
    x_values_unique = np.unique(x_values)

    img_flipped = np.ascontiguousarray(img_binary[:, x_values_unique].T)
    cols_y, rows_y_all = find_alignment_patterns(img_flipped, max_error)
    cols_y = x_values_unique[cols_y]

    # Cross-validate: horizontal pattern's middle segment must overlap with
    # a vertical pattern's middle segment
    cond1 = rows_x.reshape(-1, 1) >= rows_y_all[:, 2].reshape(1, -1)
    cond2 = rows_x.reshape(-1, 1) <= rows_y_all[:, 3].reshape(1, -1)
    cond3 = cols_y.reshape(1, -1) >= cols_x_all[:, 2].reshape(-1, 1)
    cond4 = cols_y.reshape(1, -1) <= cols_x_all[:, 3].reshape(-1, 1)
    valid = (cond1 & cond2 & cond3 & cond4).any(axis=1)

    return rows_x[valid], cols_x_all[valid]
