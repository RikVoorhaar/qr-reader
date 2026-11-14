"""Interior alignment-pattern landmarks for a better homography estimate.

The default pipeline fits a homography from only the 12 **outer** finder corners
(three finder patterns × four corners).  Because those corners are clustered at
the symbol's periphery, a small amount of corner noise is extrapolated outward
and can push interior module centres off their true positions.

This module adds the interior **5×5 alignment patterns** (versions ≥ 2) as
extra, spread-out correspondences.  Their canonical grid positions are known
from the QR version, so each is located by normalized template matching around
the position predicted by the initial homography, and the homography is refit
from the combined point set with DLT + LM.

Everything works in ``(x, y)`` image coordinates (matching the homography
module's convention).  No OpenCV is used: the normalized cross-correlation is
implemented directly with numpy/scipy (``scipy.signal.fftconvolve`` + integral
images) rather than ``cv2.matchTemplate``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from qr_reader.decoder.tables import VERSIONS
from qr_reader.detector.homography import (
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)


# ---------------------------------------------------------------------------
# Alignment-pattern template matching (numpy/scipy — no OpenCV)
# ---------------------------------------------------------------------------


def alignment_pattern_template(
    module_px: float,
    pad_modules: float = 1.0,
) -> np.ndarray:
    """Render a synthetic 5×5 alignment-pattern template at *module_px*.

    The template covers ``5 + 2*pad_modules`` modules so it includes a white
    quiet ring around the dark border, giving the matcher a distinctive
    signature.  Values are in ``[0, 1]`` (0 = dark, 1 = bright).
    """
    half_modules = 2.5 + pad_modules
    n = int(np.ceil(2.0 * half_modules * module_px))
    if n % 2 == 0:
        n += 1

    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2.0
    u = np.maximum(np.abs(xs - c), np.abs(ys - c)) / module_px

    # dark centre (u < 0.5), white ring (0.5..1.5), dark border (1.5..2.5),
    # white outside (u >= 2.5).
    t = np.where(u < 0.5, 0.0, 1.0)
    t = np.where((u >= 1.5) & (u < 2.5), 0.0, t)

    return t.astype(np.float32)


def _ncc_match(window: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalized cross-correlation (== cv2.TM_CCOEFF_NORMED).

    Returns a ``(H-th+1, W-tw+1)`` array where entry ``(i, j)`` is the
    correlation between the template and the window patch whose top-left corner
    is ``(i, j)``.
    """
    th, tw = template.shape
    t = template.astype(np.float64)
    t = t - t.mean()
    t_norm = float(np.linalg.norm(t))
    if t_norm < 1e-12:
        return np.zeros((window.shape[0] - th + 1, window.shape[1] - tw + 1))

    img = window.astype(np.float64)

    # Numerator: cross-correlation of the zero-mean template with the image.
    num = fftconvolve(img, t[::-1, ::-1], mode="valid")

    # Denominator: ||t|| * per-patch population std * sqrt(n), via integral
    # images for the local sum and sum-of-squares.
    S = np.pad(np.cumsum(np.cumsum(img, axis=0), axis=1), ((1, 0), (1, 0)))
    S2 = np.pad(np.cumsum(np.cumsum(img * img, axis=0), axis=1), ((1, 0), (1, 0)))

    nh, nw = num.shape
    i = np.arange(nh)[:, None]
    j = np.arange(nw)[None, :]
    sum1 = S[i + th, j + tw] - S[i, j + tw] - S[i + th, j] + S[i, j]
    sum2 = S2[i + th, j + tw] - S2[i, j + tw] - S2[i + th, j] + S2[i, j]
    n = th * tw
    var = (sum2 - sum1 * sum1 / n) / n
    std = np.sqrt(np.clip(var, 0.0, None))

    denom = t_norm * std * np.sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = num / denom
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def _parabolic_peak(res: np.ndarray, mx: int, my: int) -> tuple[float, float]:
    """Sub-pixel refinement of a correlation peak at integer ``(mx, my)``."""
    h, w = res.shape
    dx = 0.0
    dy = 0.0
    if 0 < mx < w - 1:
        f = res[my, mx - 1 : mx + 2].astype(np.float64)
        denom = f[0] - 2.0 * f[1] + f[2]
        if abs(denom) > 1e-12:
            dx = 0.5 * (f[0] - f[2]) / denom
    if 0 < my < h - 1:
        f = res[my - 1 : my + 2, mx].astype(np.float64)
        denom = f[0] - 2.0 * f[1] + f[2]
        if abs(denom) > 1e-12:
            dy = 0.5 * (f[0] - f[2]) / denom
    return float(dx), float(dy)


def match_alignment_pattern(
    img_gray: np.ndarray,
    center_xy: np.ndarray,
    module_px: float,
    search_radius_px: float | None = None,
) -> tuple[np.ndarray, float]:
    """Locate an alignment-pattern centre via normalized template matching.

    Parameters
    ----------
    img_gray : ndarray (H, W)
        Grayscale image.
    center_xy : ndarray (2,)
        Initial centre estimate in ``(x, y)`` image coordinates (e.g. projected
        through an initial homography).
    module_px : float
        Local module pitch in pixels at the pattern location.
    search_radius_px : float or None
        Half-extent of the search window (defaults to 2 modules).

    Returns
    -------
    (image_xy, confidence) where ``image_xy`` is the sub-pixel centre in
    ``(x, y)`` and ``confidence`` is the normalized correlation in ``[0, 1]``.
    """
    if search_radius_px is None:
        search_radius_px = 2.0 * module_px

    template = alignment_pattern_template(module_px)
    th, tw = template.shape
    ctr_x = (tw - 1) / 2.0
    ctr_y = (th - 1) / 2.0

    half = int(np.ceil(search_radius_px)) + int(max(ctr_x, ctr_y)) + 1
    cx, cy = float(center_xy[0]), float(center_xy[1])

    h, w = img_gray.shape
    x0 = int(round(cx - half))
    x1 = int(round(cx + half)) + 1
    y0 = int(round(cy - half))
    y1 = int(round(cy + half)) + 1

    # Pad the crop so the template can be matched even near image borders.
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    window = img_gray[sy0:sy1, sx0:sx1].astype(np.float32)
    if pad_l or pad_t or pad_r or pad_b:
        window = np.pad(window, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")

    if window.shape[0] < th or window.shape[1] < tw:
        return np.array([cx, cy]), 0.0

    res = _ncc_match(window, template)
    my, mx = np.unravel_index(int(np.argmax(res)), res.shape)
    max_val = float(res[my, mx])
    dx, dy = _parabolic_peak(res, mx, my)

    image_x = (x0 - pad_l) + (mx + dx) + ctr_x
    image_y = (y0 - pad_t) + (my + dy) + ctr_y

    confidence = float(max(0.0, min(1.0, max_val)))
    return np.array([image_x, image_y]), confidence


# ---------------------------------------------------------------------------
# Canonical positions and refined homography
# ---------------------------------------------------------------------------


def alignment_centers_grid(version: int) -> list[tuple[float, float]]:
    """Return the canonical ``(x, y)`` grid coordinates of alignment centres.

    Excludes the three corner patterns that coincide with the finder patterns.
    Returns an empty list for version 1 (no alignment patterns).
    """
    positions = VERSIONS[version].alignment_positions
    if not positions:
        return []

    centers: list[tuple[float, float]] = []
    first = positions[0]
    last = positions[-1]
    for row_pos in positions:
        for col_pos in positions:
            # Skip the three corners occupied by finder patterns.
            if (row_pos == first and col_pos == first) or (
                row_pos == first and col_pos == last
            ) or (row_pos == last and col_pos == first):
                continue
            centers.append((col_pos + 0.5, row_pos + 0.5))
    return centers


def _local_module_px(H: np.ndarray, gx: float, gy: float) -> float:
    p0 = project_points(H, np.array([[gx, gy]]))[0]
    px = project_points(H, np.array([[gx + 1.0, gy]]))[0]
    py = project_points(H, np.array([[gx, gy + 1.0]]))[0]
    return float((np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) / 2.0)


def _canonicalize_xy(
    corners_xy: np.ndarray, right: np.ndarray, down: np.ndarray
) -> np.ndarray:
    """Order four finder corners as [TL, TR, BR, BL] using a triplet basis."""
    centre = corners_xy.mean(axis=0)
    uv = corners_xy - centre
    u = uv @ right
    v = uv @ down
    idx_tl = int(np.argmin(u + v))
    idx_tr = int(np.argmax(u - v))
    idx_br = int(np.argmax(u + v))
    idx_bl = int(np.argmin(u - v))
    return corners_xy[np.array([idx_tl, idx_tr, idx_br, idx_bl])]


def _triplet_axes_xy(global_outer_xy: dict, triplet) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, down) unit vectors in (x, y) for a triplet."""
    tl = global_outer_xy[triplet.top_left_idx].mean(axis=0)
    tr = global_outer_xy[triplet.top_right_idx].mean(axis=0)
    bl = global_outer_xy[triplet.bottom_left_idx].mean(axis=0)
    right = tr - tl
    right = right / (float(np.linalg.norm(right)) + 1e-12)
    down = bl - tl
    down = down / (float(np.linalg.norm(down)) + 1e-12)
    return right, down


def _outer_grid_coords(N: int) -> dict[str, np.ndarray]:
    """Canonical outer finder corners in ``(x, y)``, [TL, TR, BR, BL]."""
    return {
        "TL": np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64),
        "TR": np.array([[N - 7, 0], [N, 0], [N, 7], [N - 7, 7]], dtype=np.float64),
        "BL": np.array([[0, N - 7], [7, N - 7], [7, N], [0, N]], dtype=np.float64),
    }


def build_refined_homography(
    img_gray: np.ndarray,
    H_base: np.ndarray,
    triplet,
    fps: list,
    version: int,
) -> tuple[np.ndarray, dict]:
    """Fit the pipeline homography from outer corners + alignment patterns.

    Parameters
    ----------
    img_gray : grayscale image.
    H_base : initial 12-corner homography used to seed the alignment search.
    triplet : Triplet naming the three finder patterns.
    fps : list of :class:`FinderPattern` (outer corners are read from these).
    version : estimated QR version.

    Returns
    -------
    (H, stats) where *stats* reports ``n_outer``, ``n_align``, ``n_corr`` and
    the mean reprojection error ``err``.
    """
    fp_map = {fp.cluster_idx: fp for fp in fps}
    global_outer_xy = {ci: fp.outer_corners[:, ::-1] for ci, fp in fp_map.items()}

    N = 4 * version + 17
    right, down = _triplet_axes_xy(global_outer_xy, triplet)
    grid = _outer_grid_coords(N)
    idx = {
        "TL": triplet.top_left_idx,
        "TR": triplet.top_right_idx,
        "BL": triplet.bottom_left_idx,
    }

    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for role in ["TL", "TR", "BL"]:
        ci = idx[role]
        corners = _canonicalize_xy(global_outer_xy[ci], right, down)
        g = grid[role]
        for k in range(4):
            src.append(g[k])
            dst.append(corners[k])

    for (gx, gy) in alignment_centers_grid(version):
        center0 = project_points(H_base, np.array([[gx, gy]]))[0]
        module_px = _local_module_px(H_base, gx, gy)
        if module_px < 0.5:
            continue
        image_xy, _conf = match_alignment_pattern(img_gray, center0, module_px)
        src.append(np.array([gx, gy]))
        dst.append(image_xy)

    src_arr = np.array(src, dtype=np.float64)
    dst_arr = np.array(dst, dtype=np.float64)

    H = estimate_homography_dlt(src_arr, dst_arr)
    try:
        H = refine_homography_lm(H, src_arr, dst_arr, loss="linear")
    except Exception:
        pass

    proj = project_points(H, src_arr)
    err = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
    stats = {
        "n_outer": 12,
        "n_align": len(src_arr) - 12,
        "n_corr": len(src_arr),
        "err": err,
    }
    return H, stats
