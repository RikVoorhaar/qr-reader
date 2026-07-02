"""Homography estimation: DLT, RANSAC, LM refinement, and QR corner projection.

All functions take/produce (x, y) points. Convert (row, col) → (x, y) at the
call boundary using ``pts[:, ::-1]``.

Coordinate conventions:
  - *src*: canonical grid points in (x, y).
  - *dst*: image points in (x, y).
"""

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalization_transform(points: np.ndarray) -> np.ndarray:
    """Compute the sqrt(2) mean-distance normalization transform.

    ``points``: (n, 2) array in (x, y).

    Returns T (3×3): ``T @ [x, y, 1]ᵀ`` normalizes the points so their centroid
    is at (0, 0) and their mean distance from the origin is sqrt(2).
    """
    n = points.shape[0]
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))
    if mean_dist < 1e-12:
        mean_dist = 1.0
    scale = np.sqrt(2.0) / mean_dist

    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return T


def _apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3×3 transform to (n, 2) points."""
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    transformed = homogeneous @ T.T
    return transformed[:, :2]


# ---------------------------------------------------------------------------
# DLT
# ---------------------------------------------------------------------------


def dlt_design_matrix_condition(src_xy: np.ndarray, dst_xy: np.ndarray) -> float:
    """Condition number of the *normalized* DLT design matrix.

    Uses the same isotropic normalization as ``estimate_homography_dlt``.
    A value >> 1 indicates a poorly conditioned DLT fit (e.g. nearly
    collinear point triplets or degenerate point layouts).
    """
    T_src = normalization_transform(src_xy)
    T_dst = normalization_transform(dst_xy)

    src_norm = _apply_transform(T_src, src_xy)
    dst_norm = _apply_transform(T_dst, dst_xy)

    n = src_norm.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A[2 * i] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        A[2 * i + 1] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]

    s = np.linalg.svd(A, compute_uv=False)
    smin = s[s > 1e-12].min()
    return float(s.max() / smin)


def estimate_homography_dlt(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """Estimate homography via normalized DLT.

    ``src_xy``, ``dst_xy``: (n, 2) arrays in (x, y).
    ``H @ [x_src, y_src, 1]ᵀ`` maps src → dst.

    Returns H (3×3), scaled so H[2, 2] = 1.
    """
    assert src_xy.shape == dst_xy.shape
    assert src_xy.shape[0] >= 4

    # Normalize
    T_src = normalization_transform(src_xy)
    T_dst = normalization_transform(dst_xy)

    src_norm = _apply_transform(T_src, src_xy)
    dst_norm = _apply_transform(T_dst, dst_xy)

    # Build 2n×9 matrix
    n = src_norm.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A[2 * i] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        A[2 * i + 1] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]

    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    h = Vt[-1]  # last row of Vt = nullspace
    H_norm = h.reshape(3, 3)

    # Denormalize
    T_dst_inv = np.linalg.inv(T_dst)
    H = T_dst_inv @ H_norm @ T_src

    # Scale so H[2,2] = 1
    H = H / H[2, 2]
    return H


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_points(H: np.ndarray, src_xy: np.ndarray) -> np.ndarray:
    """Project (x, y) points through homography H.

    Returns (n, 2) array of projected (x, y) coordinates.
    """
    n = src_xy.shape[0]
    ones = np.ones((n, 1))
    homogeneous = np.hstack([src_xy, ones])
    projected = homogeneous @ H.T
    projected = projected[:, :2] / projected[:, 2:]
    return projected


def project_points_with_jac(
    H: np.ndarray, src_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project points and compute the analytical Jacobian.

    Returns (pts, J) where:
      pts — (n, 2) array of projected (x, y) coordinates.
      J   — (2*n, 8) Jacobian of the projected coords w.r.t. the 8 free
            homography parameters [h00, h01, h02, h10, h11, h12, h20, h21]
            (h22 = 1 fixed).

    Derivatives:
      ∂u/∂h00 = x/w,  ∂u/∂h20 = -x·u/w
      ∂v/∂h10 = x/w,  ∂v/∂h20 = -x·v/w
      ...and similarly for y/w and 1/w.
    """
    x = src_xy[:, 0]  # (n,)
    y = src_xy[:, 1]  # (n,)
    ones = np.ones_like(x)

    # Numerator and denominator (w)
    num_u = H[0, 0] * x + H[0, 1] * y + H[0, 2]  # (n,)
    num_v = H[1, 0] * x + H[1, 1] * y + H[1, 2]  # (n,)
    w = H[2, 0] * x + H[2, 1] * y + 1.0  # (n,)

    inv_w = 1.0 / w

    u = num_u * inv_w  # projected x
    v = num_v * inv_w  # projected y
    pts = np.column_stack([u, v])

    # Build Jacobian row by row: 2*n rows, 8 columns
    # Layout: [h00, h01, h02, h10, h11, h12, h20, h21]
    n = src_xy.shape[0]
    J = np.empty((2 * n, 8), dtype=np.float64)

    # Common terms
    x_inv_w = x * inv_w
    y_inv_w = y * inv_w
    xu_div_w = x * u * inv_w  # ∂u/∂h20 = -x·u/w = -xu/w
    yu_div_w = y * u * inv_w  # ∂u/∂h21 = -y·u/w = -yu/w
    xv_div_w = x * v * inv_w  # ∂v/∂h20 = -x·v/w = -xv/w
    yv_div_w = y * v * inv_w  # ∂v/∂h21 = -y·v/w = -yv/w

    for i in range(n):
        # u residual row
        J[2 * i, 0] = x_inv_w[i]  # ∂u/∂h00
        J[2 * i, 1] = y_inv_w[i]  # ∂u/∂h01
        J[2 * i, 2] = inv_w[i]  # ∂u/∂h02
        J[2 * i, 3] = 0.0  # ∂u/∂h10
        J[2 * i, 4] = 0.0  # ∂u/∂h11
        J[2 * i, 5] = 0.0  # ∂u/∂h12
        J[2 * i, 6] = -xu_div_w[i]  # ∂u/∂h20
        J[2 * i, 7] = -yu_div_w[i]  # ∂u/∂h21

        # v residual row
        J[2 * i + 1, 0] = 0.0  # ∂v/∂h00
        J[2 * i + 1, 1] = 0.0  # ∂v/∂h01
        J[2 * i + 1, 2] = 0.0  # ∂v/∂h02
        J[2 * i + 1, 3] = x_inv_w[i]  # ∂v/∂h10
        J[2 * i + 1, 4] = y_inv_w[i]  # ∂v/∂h11
        J[2 * i + 1, 5] = inv_w[i]  # ∂v/∂h12
        J[2 * i + 1, 6] = -xv_div_w[i]  # ∂v/∂h20
        J[2 * i + 1, 7] = -yv_div_w[i]  # ∂v/∂h21

    return pts, J


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------


def ransac_homography(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    threshold: float = 2.0,
    iters: int = 1000,
    min_inliers: int = 12,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust homography estimation via RANSAC.

    Returns (H, inlier_mask) where inlier_mask is a boolean array.
    If the best model has fewer than *min_inliers* inliers, returns
    (identity, all-false mask) — the caller should check the inlier count.
    """
    n = src_xy.shape[0]
    assert n >= 4

    rng = np.random.default_rng(seed)

    best_H = None
    best_inliers = None
    best_count = -1

    for _ in range(iters):
        # Sample 4 random indices
        sample_idx = rng.choice(n, size=4, replace=False)

        # Degeneracy check: skip if points are nearly colinear or duplicates
        pts = src_xy[sample_idx]
        center = pts.mean(axis=0)
        centered = pts - center
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        if len(S) < 2 or S[1] / (S[0] + 1e-12) < 1e-4:
            continue

        # Fit DLT on sample
        H = estimate_homography_dlt(src_xy[sample_idx], dst_xy[sample_idx])

        # Count inliers
        projected = project_points(H, src_xy)
        dists = np.linalg.norm(projected - dst_xy, axis=1)
        inliers = dists < threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

            # Early exit if perfect
            if count == n:
                break

    if best_H is None or best_count < min_inliers:
        return np.eye(3), np.zeros(n, dtype=bool)

    # Refit on all inliers
    if np.sum(best_inliers) >= 4:
        best_H = estimate_homography_dlt(src_xy[best_inliers], dst_xy[best_inliers])

    return best_H, best_inliers


# ---------------------------------------------------------------------------
# Levenberg-Marquardt refinement
# ---------------------------------------------------------------------------


def refine_homography_lm(
    H_init: np.ndarray,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    loss: str = "linear",
) -> np.ndarray:
    """Refine a homography via LM optimization on reprojection error.

    Optimises 8 parameters (h33=1 fixed). Uses scipy ``least_squares``
    with an analytical Jacobian for speed.

    Returns refined H (3×3).
    """
    H_flat = H_init.ravel()
    x0 = H_flat[:8].copy()
    n = src_xy.shape[0]

    def fun(params: np.ndarray) -> np.ndarray:
        H = np.eye(3)
        H.ravel()[:8] = params
        projected, _ = project_points_with_jac(H, src_xy)
        return (projected - dst_xy).ravel()

    def jac(params: np.ndarray) -> np.ndarray:
        H = np.eye(3)
        H.ravel()[:8] = params
        _, J = project_points_with_jac(H, src_xy)
        return J

    result = least_squares(
        fun,
        x0,
        jac=jac,
        method="lm",
        loss=loss,
        max_nfev=2000,
    )

    H_refined = np.eye(3)
    H_refined.ravel()[:8] = result.x
    return H_refined


# ---------------------------------------------------------------------------
# QR corner projection
# ---------------------------------------------------------------------------


def compute_qr_corners(H: np.ndarray, N: int) -> np.ndarray:
    """Project the 4 QR corners into the image using the homography.

    Projects (0,0), (N,0), (N,N), (0,N) → returns (4, 2) in (x, y)
    order [TL, TR, BR, BL] for OpenCV.

    Note: N is the QR module count (e.g., 21 for version 1).
    """
    grid_corners = np.array(
        [
            [0.0, 0.0],  # TL
            [N, 0.0],  # TR
            [N, N],  # BR
            [0.0, N],  # BL
        ],
        dtype=np.float64,
    )
    return project_points(H, grid_corners)
