"""ROI extraction from CandidateCluster for edge-based corner finding."""

import numpy as np

from qr_reader.detector.clustering import CandidateCluster


def cluster_to_bbox(
    cluster: CandidateCluster, scale: float = 1.5
) -> tuple[int, int, int, int]:
    """
    Compute a padded bounding box around a CandidateCluster.

    Parameters
    ----------
    cluster : CandidateCluster
        Merged alignment pattern cluster.
    scale : float
        Multiplier applied to the half-extent for padding (default 1.5).

    Returns
    -------
    tuple[int, int, int, int]
        (row_start, row_end, col_start, col_end) — integer bounds, not clamped.
    """
    center_row = cluster.row
    center_col = (cluster.cols[2] + cluster.cols[3]) / 2.0

    width = float(cluster.cols[5] - cluster.cols[0])
    height = float(cluster.cols[3] - cluster.cols[2])
    half_extent = max(width, height) / 2.0

    r0 = int(center_row - scale * half_extent)
    r1 = int(center_row + scale * half_extent)
    c0 = int(center_col - scale * half_extent)
    c1 = int(center_col + scale * half_extent)

    return r0, r1, c0, c1


def cutout(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract a sub-image from `image` defined by `bbox`, clamped to image bounds.

    Parameters
    ----------
    image : np.ndarray
        Source image (2-D grayscale or 3-D).
    bbox : tuple[int, int, int, int]
        (row_start, row_end, col_start, col_end) — may extend beyond image bounds.

    Returns
    -------
    np.ndarray
        Clamped sub-image slice.
    """
    r0, r1, c0, c1 = bbox
    h, w = image.shape[:2]

    r0 = max(0, r0)
    r1 = min(h, r1)
    c0 = max(0, c0)
    c1 = min(w, c1)

    return image[r0:r1, c0:c1]
