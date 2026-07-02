"""Finder-pattern edge fitting via orientation histogram + 1D profile analysis.

Replaces Hough-based edge detection with a QR-specific model-fitting approach
that exploits the known 7×7 module structure of finder patterns.

All geometry in (x, y) = (col, row) pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FinderFit:
    """Fitted finder pattern geometry from orientation + 1D profile analysis.

    Attributes
    ----------
    center : ndarray (2,)
        Fitted finder center in (x, y) pixel coordinates.
    e1 : ndarray (2,)
        First local axis (unit vector).  e2 is perpendicular.
    e2 : ndarray (2,)
        Second local axis (unit, perpendicular to e1).
    m : float
        Fitted module pitch in pixels.
    outer_lines : dict
        Keys ``"u+"``, ``"u-"``, ``"v+"``, ``"v-"``; values ``(normal, rho)``.
    corners : ndarray (4, 2)
        4 outer corners (x, y): [(-,-), (+,-), (+,+), (-,+)].
    score : float
        Template fit score (populated by Phase 4).
    phi : float
        Orientation angle in radians (mod π/2).
    """

    center: np.ndarray  # (2,)
    e1: np.ndarray  # (2,)
    e2: np.ndarray  # (2,)
    m: float
    outer_lines: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    score: float = 0.0
    phi: float = 0.0
    m_u: float | None = None
    m_v: float | None = None


def estimate_orientation(
    nms: np.ndarray,
    angle: np.ndarray,
    center_xy: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate finder-pattern orientation via 4-fold symmetric gradient histogram.

    Computes ``z = Σ w_i exp(4j α_i)`` across all NMS edge pixels,
    where ``α = angle mod π``.  The factor 4 collapses the square's
    4-fold symmetry, giving a robust orientation estimate without
    needing to identify which edges belong to which side.

    Parameters
    ----------
    nms : ndarray (H, W)
        NMS edge magnitudes from ``extract_thin_edges``.
    angle : ndarray (H, W)
        Edge-normal angles ``atan2(gy, gx)`` in [-π, π].
    center_xy : ndarray (2,)
        Approximate finder center in (x, y) pixel coordinates
        (used only for discarding distant edge pixels).

    Returns
    -------
    phi : float
        Orientation angle in radians (mod π/2).
    e1 : ndarray (2,)
        First local axis unit vector ``(cos φ, sin φ)``.
    e2 : ndarray (2,)
        Second local axis unit vector ``(-sin φ, cos φ)``.
    """
    ys, xs = np.nonzero(nms)
    if len(ys) == 0:
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        return 0.0, e1, e2

    w = nms[ys, xs].astype(np.float64)

    alpha = np.fmod(angle[ys, xs], np.pi)
    neg = alpha < 0
    alpha = np.where(neg, alpha + np.pi, alpha)

    z = np.sum(w * np.exp(4.0j * alpha))
    phi = (np.angle(z) / 4.0) % (np.pi / 2.0)

    e1 = np.array([np.cos(phi), np.sin(phi)])
    e2 = np.array([-np.sin(phi), np.cos(phi)])

    return float(phi), e1, e2


def build_projection_profile(
    nms: np.ndarray,
    angle: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    m_est: float,
    angle_gate_deg: float = 22.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 1D projection profile of edge strength along *axis*.

    Gated to edge pixels whose gradient normal is within *angle_gate_deg*
    of *axis* (mod π).  Bin width = m_est/4 (quarter-module resolution).

    Parameters
    ----------
    nms : ndarray (H, W)
        NMS edge magnitudes.
    angle : ndarray (H, W)
        Edge-normal angles in [-π, π].
    center_xy : ndarray (2,)
        Finder center estimate (x, y).
    axis : ndarray (2,)
        Unit vector along which to project.
    m_est : float
        Estimated module pitch in pixels.
    angle_gate_deg : float
        Half-angle gate in degrees.  Default 22.5°.

    Returns
    -------
    positions : ndarray (n_bins,)
        Bin centre positions along *axis* relative to center.
    profile : ndarray (n_bins,)
        Weighted edge strength per bin.
    """
    ys, xs = np.nonzero(nms)
    if len(ys) == 0:
        return np.array([]), np.array([])

    w = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    axis_angle = float(np.arctan2(axis[1], axis[0])) % np.pi
    alpha = np.fmod(angle[ys, xs], np.pi)
    alpha = np.where(alpha < 0, alpha + np.pi, alpha)

    angle_diff = np.abs(alpha - axis_angle)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    gate = angle_diff < np.deg2rad(angle_gate_deg)

    if not np.any(gate):
        return np.array([]), np.array([])

    proj = (points[gate] - center_xy) @ axis
    w_proj = w[gate]

    bin_width = m_est / 4.0
    half_span = 4.0 * m_est
    n_bins = int(np.ceil(2.0 * half_span / bin_width))
    n_bins = max(n_bins, 1)

    positions_edges = np.linspace(-half_span, half_span, n_bins + 1)
    positions = (positions_edges[:-1] + positions_edges[1:]) / 2.0

    profile = np.zeros(n_bins, dtype=np.float64)
    binned = np.digitize(proj, positions_edges[1:])
    for i, b in enumerate(binned):
        if 0 <= b < n_bins:
            profile[b] += w_proj[i]

    return positions, profile


def _interp_profile(positions: np.ndarray, profile: np.ndarray, x: float) -> float:
    """Linear interpolation of *profile* at position *x*.  Returns 0 outside range."""
    if len(positions) < 2 or x < positions[0] or x > positions[-1]:
        return 0.0
    idx = np.searchsorted(positions, x)
    if idx == 0:
        return float(profile[0])
    if idx >= len(positions):
        return float(profile[-1])
    lo, hi = positions[idx - 1], positions[idx]
    if hi == lo:
        return float(profile[idx])
    t = (x - lo) / (hi - lo)
    return float((1.0 - t) * profile[idx - 1] + t * profile[idx])


def fit_finder_1d(
    profile: np.ndarray,
    positions: np.ndarray,
    m_est: float,
    expected_offsets: np.ndarray | None = None,
    n_center: int = 21,
    n_m: int = 25,
    miss_weight: float = 0.3,
) -> dict:
    """Fit the 6 expected edge-transition positions in a 1D profile.

    Searches over ``(center_offset, m)`` to maximise the sum of
    interpolated profile values at ``{±3.5, ±2.5, ±1.5} * m + du``,
    penalised by mean profile value away from expected positions.

    Parameters
    ----------
    profile : ndarray (n_bins,)
        1D projection profile.
    positions : ndarray (n_bins,)
        Bin centre positions.
    m_est : float
        Initial estimate of module pitch.
    expected_offsets : ndarray or None
        Expected transition offsets in module units.  Default:
        [-3.5, -2.5, -1.5, 1.5, 2.5, 3.5].
    n_center : int
        Number of search points for centre offset.
    n_m : int
        Number of search points for module pitch.
    miss_weight : float
        Penalty weight for non-peak regions.

    Returns
    -------
    dict
        Keys: ``center_offset``, ``m_fitted``, ``peak_positions``, ``score``,
        ``profile_values``.
    """
    if expected_offsets is None:
        expected_offsets = np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5])

    n_bins = len(profile)
    if n_bins == 0:
        return {
            "center_offset": 0.0,
            "m_fitted": m_est,
            "peak_positions": expected_offsets * m_est,
            "score": 0.0,
            "profile_values": np.zeros(len(expected_offsets)),
        }

    mean_profile = float(np.mean(profile)) if n_bins > 0 else 0.0

    best_score = -np.inf
    best_du = 0.0
    best_m = m_est

    du_vals = np.linspace(-m_est / 2.0, m_est / 2.0, n_center)
    m_vals = np.linspace(m_est * 0.5, m_est * 1.8, n_m)

    for du in du_vals:
        for m in m_vals:
            peaks = expected_offsets * m + du
            vals = [_interp_profile(positions, profile, p) for p in peaks]
            hit_score = sum(vals)
            score = hit_score - miss_weight * mean_profile * len(expected_offsets)

            if score > best_score:
                best_score = score
                best_du = du
                best_m = m

    best_peaks = expected_offsets * best_m + best_du
    best_vals = [_interp_profile(positions, profile, p) for p in best_peaks]

    return {
        "center_offset": float(best_du),
        "m_fitted": float(best_m),
        "peak_positions": best_peaks,
        "score": float(best_score),
        "profile_values": np.array(best_vals, dtype=np.float64),
    }


def refine_outer_line(
    nms: np.ndarray,
    angle: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    position: float,
    distance_thresh: float = 3.0,
    angle_gate_deg: float = 22.5,
    fix_normal: bool = True,
) -> tuple[np.ndarray, float]:
    """Refine an outer finder line on nearby NMS pixels.

    The line passes through ``center_xy + position * axis`` with normal
    parallel to *axis*.  Nearby edge pixels with consistent gradient
    angles are collected.

    When *fix_normal* is ``True`` (default), the normal is kept fixed to
    *axis* and only *rho* is refined via a weighted-mean 1-D fit — robust
    when support pixels are sparse.  When ``False``, a full weighted TLS
    fit refines both direction and position.

    Parameters
    ----------
    nms : ndarray (H, W)
        NMS edge magnitudes.
    angle : ndarray (H, W)
        Edge-normal angles in [-π, π].
    center_xy : ndarray (2,)
        Finder center (x, y).
    axis : ndarray (2,)
        Unit vector — the line's normal direction.
    position : float
        Position along *axis* where the line passes through centre.
    distance_thresh : float
        Max perpendicular distance for support pixels.  Default 3.0 px.
    angle_gate_deg : float
        Max angular deviation of edge-normal from *axis* (mod π).  Default 22.5°.
    fix_normal : bool
        If True, keep normal == axis and only fit rho (default).  If False,
        do full TLS fit on support pixels.

    Returns
    -------
    normal : ndarray (2,)
        Refined unit normal vector (canonicalised so rho ≥ 0).
    rho : float
        Refined signed distance from origin.
    """
    approx_rho = float(axis @ (center_xy + position * axis))

    ys, xs = np.nonzero(nms)
    if len(ys) < 2:
        normal = axis.copy()
        return normal, approx_rho

    strengths = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    dists = np.abs(points @ axis - approx_rho)
    mask = dists < distance_thresh

    axis_angle = float(np.arctan2(axis[1], axis[0])) % np.pi
    alpha = np.fmod(angle[ys, xs], np.pi)
    alpha = np.where(alpha < 0, alpha + np.pi, alpha)
    angle_diff = np.abs(alpha - axis_angle)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    mask &= angle_diff < np.deg2rad(angle_gate_deg)

    if np.sum(mask) < 2:
        normal = axis.copy()
        return normal, approx_rho

    support_pts = points[mask]
    support_w = strengths[mask]

    if fix_normal:
        normal = axis.copy()
        w = support_w / support_w.sum()
        rho = float((support_pts @ normal) @ w)
        return normal, rho

    w = support_w / support_w.sum()
    c = (support_pts * w[:, None]).sum(axis=0)
    X = support_pts - c
    Xw = X * np.sqrt(w[:, None])
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)

    refined_normal = vt[1]
    refined_rho = float(refined_normal @ c)

    if refined_rho < 0:
        refined_normal = -refined_normal
        refined_rho = -refined_rho

    return refined_normal, refined_rho


def intersect_lines(
    n1: np.ndarray, rho1: float, n2: np.ndarray, rho2: float
) -> np.ndarray | None:
    """Intersect two lines ``n1·p = ρ1`` and ``n2·p = ρ2``.

    Returns the intersection point (x, y), or None if lines are parallel.
    """
    A = np.column_stack([n1, n2]).T  # 2x2
    b = np.array([rho1, rho2])
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if abs(det) < 1e-12:
        return None
    return np.linalg.solve(A, b)


def _expected_polarity_sign(t: float, k: int) -> float:
    """Expected gradient sign at transition offset ``t * k * m``.

    *t* is ±1 (direction along local axis), *k* ∈ {1, 2, 3}.
    Returns +1 or -1.
    """
    return float(t * ((-1) ** (k + 1)))


def _sample_1d_cross_section(
    roi: np.ndarray,
    center_xy: np.ndarray,
    axis: np.ndarray,
    pos: float,
    num_samples: int,
) -> np.ndarray:
    """Sample ROI intensity along a line segment through *center_xy*.

    Parameters
    ----------
    roi : ndarray (H, W)
        Grayscale ROI.
    center_xy : ndarray (2,)
        Center point (x, y).
    axis : ndarray (2,)
        Unit direction vector.
    pos : float
        Signed extent along *axis* (range [-pos, +pos]).
    num_samples : int
        Number of samples along the segment.

    Returns
    -------
    samples : ndarray (num_samples,)
        Interpolated intensity values.
    """
    H, W = roi.shape
    ts = np.linspace(-pos, pos, num_samples)
    xs = center_xy[0] + ts * axis[0]
    ys = center_xy[1] + ts * axis[1]

    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    fx = xs - x0.astype(np.float64)
    fy = ys - y0.astype(np.float64)

    return (
        (1.0 - fy) * ((1.0 - fx) * roi[y0, x0] + fx * roi[y0, x1])
        + fy * ((1.0 - fx) * roi[y1, x0] + fx * roi[y1, x1])
    )


def _cross_section_ncc(
    samples: np.ndarray, template: np.ndarray
) -> float:
    """Normalised cross-correlation between *samples* and *template*."""
    s = samples.astype(np.float64)
    t = template.astype(np.float64)
    s_mean = s.mean()
    t_mean = t.mean()
    s_std = s.std()
    t_std = t.std()
    if s_std < 1e-12 or t_std < 1e-12:
        return 0.0
    return float(((s - s_mean) * (t - t_mean)).mean() / (s_std * t_std))


def fit_finder_template(
    roi_gray: np.ndarray,
    nms: np.ndarray,
    angle: np.ndarray,
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    m_est: float,
    w_edge: float = 0.25,
    w_polarity: float = 0.35,
    w_contrast: float = 0.25,
    w_quiet: float = 0.15,
    n_center: int = 9,
    n_m: int = 11,
    angle_gate_deg: float = 22.5,
) -> FinderFit:
    """Fit a finder template with polarity + contrast + quiet-zone scoring.

    Extends the 1D profile approach by incorporating:
    - Polarity scoring: gradient direction consistency with 1:1:3:1:1
      intensity profile
    - Cross-section contrast: NCC between sampled intensity and ideal template
    - Quiet-zone brightness: must be bright outside ±3.5m

    Searches over ``(du, dv, m)`` in a local neighbourhood around the
    Phase 2 estimates.

    Parameters
    ----------
    roi_gray : ndarray (H, W)
        Grayscale ROI.
    nms : ndarray (H, W)
        NMS edge magnitudes.
    angle : ndarray (H, W)
        Edge-normal angles in [-π, π].
    center_xy : ndarray (2,)
        Finder centre estimate (x, y).
    e1 : ndarray (2,)
        First local axis unit vector.
    e2 : ndarray (2,)
        Second local axis unit vector.
    m_est : float
        Estimated module pitch.
    w_edge : float
        Weight for edge-response score.
    w_polarity : float
        Weight for polarity-consistency score.
    w_contrast : float
        Weight for contrast NCC score.
    w_quiet : float
        Weight for quiet-zone brightness score.
    n_center : int
        Number of search points per dimension for centre offset.
    n_m : int
        Number of search points for module pitch.
    angle_gate_deg : float
        Angle gate for edge-polarity computation.

    Returns
    -------
    FinderFit
        Fitted finder geometry with combined score.
    """
    H, W = roi_gray.shape
    expected_offsets = np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5])

    ys, xs = np.nonzero(nms)
    if len(ys) == 0:
        return FinderFit(
            center=center_xy.copy(),
            e1=e1.copy(),
            e2=e2.copy(),
            m=m_est,
            phi=float(np.arctan2(e1[1], e1[0])),
        )

    strengths = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    alpha = np.fmod(angle[ys, xs], np.pi)
    alpha = np.where(alpha < 0, alpha + np.pi, alpha)

    e1_angle = float(np.arctan2(e1[1], e1[0])) % np.pi
    e2_angle = float(np.arctan2(e2[1], e2[0])) % np.pi

    # Gate pixels by axis alignment
    diff1 = np.abs(alpha - e1_angle)
    diff1 = np.minimum(diff1, np.pi - diff1)
    gate1 = diff1 < np.deg2rad(angle_gate_deg)

    diff2 = np.abs(alpha - e2_angle)
    diff2 = np.minimum(diff2, np.pi - diff2)
    gate2 = diff2 < np.deg2rad(angle_gate_deg)

    ideal_template = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    du_vals = np.linspace(-m_est / 4.0, m_est / 4.0, n_center)
    dv_vals = np.linspace(-m_est / 4.0, m_est / 4.0, n_center)
    m_vals = np.linspace(m_est * 0.85, m_est * 1.15, n_m)

    mean_strength = float(strengths.mean()) if len(strengths) > 0 else 1.0
    max_strength = float(strengths.max()) if len(strengths) > 0 else 1.0

    best_score = -np.inf
    best_du, best_dv, best_m = 0.0, 0.0, m_est

    for du in du_vals:
        for dv in dv_vals:
            for m in m_vals:
                center = center_xy + du * e1 + dv * e2
                score = 0.0

                # --- edge response ---
                for axis, gate, e_angle in [
                    (e1, gate1, e1_angle),
                    (e2, gate2, e2_angle),
                ]:
                    if np.any(gate):
                        proj = (points[gate] - center) @ axis
                        for offset in expected_offsets:
                            pos = offset * m
                            dist = np.abs(proj - pos)
                            nearby = dist < (m / 2.0)
                            if np.any(nearby):
                                score += w_edge * float(strengths[gate][nearby].sum()) / max_strength

                # --- polarity scoring ---
                polarity_score = 0.0
                n_total = 0
                for sign, axis, gate in [(1, e1, gate1), (-1, e1, gate1),
                                          (1, e2, gate2), (-1, e2, gate2)]:
                    if not np.any(gate):
                        continue
                    proj = (points[gate] - center) @ axis
                    gated_strengths = strengths[gate]
                    gated_alpha = alpha[gate]
                    axis_angle_g = e1_angle if axis is e1 else e2_angle

                    for k in [1, 2, 3]:
                        pos = sign * k * m
                        dist = np.abs(proj - pos)
                        nearby = dist < (m / 2.0)
                        if not np.any(nearby):
                            continue
                        n_total += 1
                        expected_sign = _expected_polarity_sign(sign, k)
                        dot_prods = np.cos(gated_alpha[nearby] - axis_angle_g)
                        actual_sign = np.sign(dot_prods.mean()) if dot_prods.size > 0 else 0.0
                        if np.sign(actual_sign) == expected_sign:
                            polarity_score += 1.0

                if n_total > 0:
                    score += w_polarity * polarity_score / n_total

                # --- cross-section contrast ---
                ncc_sum = 0.0
                for axis in [e1, e2]:
                    samples = _sample_1d_cross_section(
                        roi_gray, center, axis, 4.5 * m, len(ideal_template) * 10
                    )
                    if len(samples) < len(ideal_template):
                        continue
                    samples_norm = (255.0 - samples) / 255.0
                    n_seg = len(ideal_template)
                    seg_len = len(samples_norm) // n_seg
                    if seg_len == 0:
                        continue
                    binned = np.array([
                        samples_norm[i * seg_len:(i + 1) * seg_len].mean()
                        for i in range(n_seg)
                    ])
                    ncc_sum += _cross_section_ncc(binned, ideal_template)
                score += w_contrast * ncc_sum / 2.0

                # --- quiet-zone scoring ---
                quiet_sum = 0.0
                quiet_n = 0
                for axis in [e1, e2]:
                    for sign in [-1, 1]:
                        qx = center[0] + sign * 4.0 * m * axis[0]
                        qy = center[1] + sign * 4.0 * m * axis[1]
                        qi = int(round(qy))
                        qj = int(round(qx))
                        if 0 <= qi < H and 0 <= qj < W:
                            quiet_sum += float(roi_gray[qi, qj]) / 255.0
                            quiet_n += 1
                if quiet_n > 0:
                    score += w_quiet * quiet_sum / quiet_n

                if score > best_score:
                    best_score = score
                    best_du = du
                    best_dv = dv
                    best_m = m

    fitted_center = center_xy + best_du * e1 + best_dv * e2

    # Compute outer lines
    outer_lines: dict[str, tuple[np.ndarray, float]] = {}
    for label, axis, pos in [
        ("u+", e1, +3.5 * best_m),
        ("u-", e1, -3.5 * best_m),
        ("v+", e2, +3.5 * best_m),
        ("v-", e2, -3.5 * best_m),
    ]:
        normal, rho = refine_outer_line(
            nms, angle, fitted_center, axis, pos,
            distance_thresh=3.0, angle_gate_deg=angle_gate_deg,
        )
        outer_lines[label] = (normal, rho)

    # Intersect to get corners: [(-,-), (+,-), (+,+), (-,+)]
    c00 = intersect_lines(*outer_lines["u-"], *outer_lines["v-"])
    c10 = intersect_lines(*outer_lines["u+"], *outer_lines["v-"])
    c11 = intersect_lines(*outer_lines["u+"], *outer_lines["v+"])
    c01 = intersect_lines(*outer_lines["u-"], *outer_lines["v+"])

    corners = np.array([
        c00 if c00 is not None else [0.0, 0.0],
        c10 if c10 is not None else [0.0, 0.0],
        c11 if c11 is not None else [0.0, 0.0],
        c01 if c01 is not None else [0.0, 0.0],
    ], dtype=np.float64)

    return FinderFit(
        center=fitted_center,
        e1=e1.copy(),
        e2=e2.copy(),
        m=float(best_m),
        outer_lines=outer_lines,
        corners=corners,
        score=float(best_score),
        phi=float(np.arctan2(e1[1], e1[0])),
    )


def extract_finder_corners(
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    m: float,
) -> np.ndarray:
    """Compute the 4 outer corners from finder pose parameters.

    Corners are returned in order [(-,-), (+,-), (+,+), (-,+)]
    relative to the e1/e2 axes.
    """
    c00 = center_xy - 3.5 * m * e1 - 3.5 * m * e2
    c10 = center_xy + 3.5 * m * e1 - 3.5 * m * e2
    c11 = center_xy + 3.5 * m * e1 + 3.5 * m * e2
    c01 = center_xy - 3.5 * m * e1 + 3.5 * m * e2
    return np.array([c00, c10, c11, c01], dtype=np.float64)


def fit_finder_full(
    nms: np.ndarray,
    angle: np.ndarray,
    roi_gray: np.ndarray,
    center_xy: np.ndarray,
    m_est: float,
    use_template: bool = False,
    angle_gate_deg: float = 22.5,
    estimate_anisotropic_pitch: bool = False,
) -> FinderFit:
    """Fit a finder pattern from NMS edges and ROI image (Phases 1–3, optionally 4).

    Production API: estimates orientation, builds 1-D projection profiles,
    fits transition positions, and computes outer corners via refined
    line positions.

    Parameters
    ----------
    nms : ndarray (H, W)
        NMS edge magnitudes from ``extract_thin_edges``.
    angle : ndarray (H, W)
        Edge-normal angles in [-π, π].
    roi_gray : ndarray (H, W)
        Grayscale ROI (used only for Phase 4 when *use_template*).
    center_xy : ndarray (2,)
        Approximate finder centre (x, y) — e.g. from ``CandidateCluster``.
    m_est : float
        Estimated module pitch — e.g. cluster width / 7.
    use_template : bool
        If True, run Phase 4 template fitting and keep the best result.
    angle_gate_deg : float
        Angle gating threshold in degrees.  Default 22.5°.
    estimate_anisotropic_pitch : bool
        If True, store the per-axis fitted module pitches in ``m_u`` and
        ``m_v``.  Corners still use the shared ``m`` to avoid changing the
        existing geometric path.

    Returns
    -------
    FinderFit
        Fitted finder geometry with corners.
    """
    phi, e1, e2 = estimate_orientation(nms, angle, center_xy)

    m_edge = estimate_m_from_edges(nms, angle, center_xy, e1, e2, angle_gate_deg)
    m_init = max(m_est, m_edge)

    pos_u, prof_u = build_projection_profile(nms, angle, center_xy, e1, m_init, angle_gate_deg)
    pos_v, prof_v = build_projection_profile(nms, angle, center_xy, e2, m_init, angle_gate_deg)

    fit_u = fit_finder_1d(prof_u, pos_u, m_init)
    fit_v = fit_finder_1d(prof_v, pos_v, m_init)

    m_fit = (fit_u["m_fitted"] + fit_v["m_fitted"]) / 2.0
    du = fit_u["center_offset"]
    dv = fit_v["center_offset"]
    fitted_center = center_xy + du * e1 + dv * e2

    n_um, um = refine_outer_line(nms, angle, fitted_center, e1, -3.5 * m_fit, angle_gate_deg=angle_gate_deg)
    n_up, up = refine_outer_line(nms, angle, fitted_center, e1, +3.5 * m_fit, angle_gate_deg=angle_gate_deg)
    n_vm, vm = refine_outer_line(nms, angle, fitted_center, e2, -3.5 * m_fit, angle_gate_deg=angle_gate_deg)
    n_vp, vp = refine_outer_line(nms, angle, fitted_center, e2, +3.5 * m_fit, angle_gate_deg=angle_gate_deg)

    corners = extract_finder_corners_from_rho(um, up, vm, vp, n_um, n_vm)

    outer_lines = {
        "u+": (n_up.copy(), up),
        "u-": (n_um.copy(), um),
        "v+": (n_vp.copy(), vp),
        "v-": (n_vm.copy(), vm),
    }

    result = FinderFit(
        center=fitted_center,
        e1=e1.copy(),
        e2=e2.copy(),
        m=float(m_fit),
        outer_lines=outer_lines,
        corners=corners,
        phi=float(phi),
    )

    if estimate_anisotropic_pitch:
        result.m_u = float(fit_u["m_fitted"])
        result.m_v = float(fit_v["m_fitted"])

    if use_template:
        tmpl = fit_finder_template(roi_gray, nms, angle, center_xy, e1, e2, m_fit,
                                   angle_gate_deg=angle_gate_deg)
        if tmpl.score > 0:
            result = tmpl
            # Carry forward per-axis diagnostics from Phase 3 when template is used.
            if estimate_anisotropic_pitch:
                result.m_u = float(fit_u["m_fitted"])
                result.m_v = float(fit_v["m_fitted"])

    return result


def _corners_from_rho(um, up, vm, vp, n_u, n_v):
    """Compute corners from line normals and signed distances.

    ``n_u`` and ``n_v`` are the (possibly canonicalised) unit normals of the
    u- and v-edge families.  The corner is ``rho_u * n_u + rho_v * n_v``,
    which is the intersection point of the two corresponding lines.
    """
    c00 = um * n_u + vm * n_v
    c10 = up * n_u + vm * n_v
    c11 = up * n_u + vp * n_v
    c01 = um * n_u + vp * n_v
    return np.array([c00, c10, c11, c01], dtype=np.float64)


def extract_finder_corners_from_rho(
    um: float, up: float, vm: float, vp: float,
    e1: np.ndarray, e2: np.ndarray,
) -> np.ndarray:
    """Compute 4 outer corners from refined rho values.

    Uses the decomposition p = u*e1 + v*e2 (valid because e1⊥e2).
    """
    return _corners_from_rho(um, up, vm, vp, e1, e2)


def estimate_m_from_edges(
    nms: np.ndarray,
    angle: np.ndarray,
    center_xy: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    angle_gate_deg: float = 22.5,
) -> float:
    """Estimate module pitch from the span of angle-gated NMS edge pixels.

    Projects gated edge pixels onto e1 and e2, computes the 5%-95%
    percentile span, and divides by 7 (expected finder width in modules).
    Uses the larger of the two axis estimates (to avoid the
    systematically-too-small cluster-width problem).

    Parameters
    ----------
    nms : ndarray (H, W)
    angle : ndarray (H, W)
    center_xy : ndarray (2,)
    e1, e2 : ndarray (2,)
        Orientation axes.
    angle_gate_deg : float

    Returns
    -------
    m : float
        Estimated module pitch.  Falls back to 5.0 px if no suitable
        edge pixels are found.
    """
    ys, xs = np.nonzero(nms)
    if len(ys) < 4:
        return 5.0

    w = nms[ys, xs].astype(np.float64)
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    alpha = np.fmod(angle[ys, xs], np.pi)
    alpha = np.where(alpha < 0, alpha + np.pi, alpha)

    m_estimates = []
    for axis in [e1, e2]:
        axis_angle = float(np.arctan2(axis[1], axis[0])) % np.pi
        diff = np.abs(alpha - axis_angle)
        diff = np.minimum(diff, np.pi - diff)
        gate = diff < np.deg2rad(angle_gate_deg)
        if np.sum(gate) < 2:
            continue
        proj = (points[gate] - center_xy) @ axis
        lo = float(np.percentile(proj, 5))
        hi = float(np.percentile(proj, 95))
        span = hi - lo
        if span > 0:
            m_estimates.append(span / 7.0)

    if not m_estimates:
        return 5.0
    return float(max(m_estimates))
