"""Corner finding from contour points via angular non-maximum suppression."""

import numpy as np


def angular_nms_top_radial_indices(
    radial_distances: np.ndarray,
    angles: np.ndarray,
    *,
    angular_nms_rad: float,
    k: int = 4,
) -> np.ndarray:
    """
    Pick ``k`` contour indices with largest radial distance (from centroid), with
    angular non-maximum suppression: after each pick, suppress candidates within
    ``angular_nms_rad`` (radians) of that pick's angle, wrapping at ±π.

    Raises ``ValueError`` if no unsuppressed candidates remain before ``k`` picks.
    """
    radial_distances = np.asarray(radial_distances, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if radial_distances.shape != angles.shape:
        raise ValueError("radial_distances and angles must have the same shape")
    if radial_distances.ndim != 1:
        raise ValueError("expected 1-D arrays")
    n = radial_distances.shape[0]
    if n == 0:
        raise ValueError("empty contour")
    supressed_mask = np.ones(n, dtype=bool)
    max_inds: list[int] = []
    neg_inf = -np.finfo(np.float64).max
    for pick in range(k):
        if not np.any(supressed_mask):
            raise ValueError(
                f"angular NMS: no candidates left before pick {pick + 1}/{k}; "
                "increase angular separation (angular_nms_rad) or reduce k."
            )
        masked_scores = np.where(supressed_mask, radial_distances, neg_inf)
        argmax = int(np.argmax(masked_scores))
        max_inds.append(argmax)
        argmax_angle = angles[argmax]
        angular_distances = np.abs(angles - argmax_angle)
        angular_distances = np.minimum(angular_distances, 2 * np.pi - angular_distances)
        supressed_mask[angular_distances < angular_nms_rad] = False
    return np.asarray(max_inds, dtype=np.intp)
