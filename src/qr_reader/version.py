"""Version estimation: cross-ratio measurement, expected cross-ratios,
constraint building/filtering, and version inference."""

from dataclasses import dataclass
from typing import List

import numpy as np

from qr_reader.landmarks import NamedLandmarks, Quadruple, get_colinear_quadruples

# ---------------------------------------------------------------------------
# Cross-ratio measurement
# ---------------------------------------------------------------------------


def measured_cross_ratio(
    points4: np.ndarray,
) -> tuple[float, float, float]:
    """Compute the cross-ratio of 4 points via SVD line-fit.

    points4: (4, 2) array in arbitrary coordinates.

    Returns (r, line_error, span) where:
      r          — cross-ratio (u2-u0)*(u3-u1) / ((u3-u0)*(u2-u1))
      line_error — sigma2/sigma1 (0 = perfectly colinear)
      span       — sigma1 (scale of the spread along the line)
    """
    center = points4.mean(axis=0)
    centered = points4 - center

    # SVD of the 4×2 centered matrix
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    sigma1, sigma2 = S[0], S[1] if len(S) > 1 else 0.0
    line_error = sigma2 / (sigma1 + 1e-12)
    span = sigma1

    # Project onto the principal axis
    direction = Vt[0]  # row of Vt = principal direction
    u = centered @ direction  # 1D coordinates

    # Ensure consistent orientation: u[3] >= u[0]
    if u[3] < u[0]:
        u = -u

    u0, u1, u2, u3 = u[0], u[1], u[2], u[3]

    # Monotonicity sanity check
    if not (u0 < u1 < u2 < u3):
        # Points are not in order along the line — still compute but flag
        pass

    # Cross-ratio:  r = (u2-u0)*(u3-u1) / ((u3-u0)*(u2-u1))
    denom = (u3 - u0) * (u2 - u1)
    if abs(denom) < 1e-12:
        return 0.0, line_error, span

    r = ((u2 - u0) * (u3 - u1)) / denom
    return r, line_error, span


# ---------------------------------------------------------------------------
# Expected cross-ratio helper
# ---------------------------------------------------------------------------


def expected_cross_ratio(x0: float, x1: float, x2: float, x3: float) -> float:
    """Cross-ratio of 4 colinear points from their 1D positions.

    r = (x2 - x0)*(x3 - x1) / ((x3 - x0)*(x2 - x1))
    """
    denom = (x3 - x0) * (x2 - x1)
    if abs(denom) < 1e-12:
        return 0.0
    return ((x2 - x0) * (x3 - x1)) / denom


# ---------------------------------------------------------------------------
# Constraint dataclass and builders
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """A single cross-ratio measurement from a colinear quadruple."""

    type: str  # "outer" or "inner"
    label: str  # e.g. "A0-A1-E0-E1"
    r_measured: float
    line_error: float
    span: float


def build_constraints(image_landmarks: NamedLandmarks) -> list[Constraint]:
    """Build cross-ratio constraints from image NamedLandmarks.

    Takes the detected image landmarks, extracts colinear quadruples, and
    computes measured cross-ratios for each.
    """
    quads = get_colinear_quadruples(image_landmarks)
    constraints = []
    for q in quads:
        r, line_error, span = measured_cross_ratio(q.points)
        constraints.append(
            Constraint(
                type=q.type,
                label=q.label,
                r_measured=r,
                line_error=line_error,
                span=span,
            )
        )
    return constraints


def filter_constraints(
    constraints: list[Constraint],
    k: int = 3,
    eps: float = 1e-2,
    min_span: float = 1.0,
    max_error_cap: float = 0.05,
) -> list[Constraint]:
    """Filter constraints by span, then keep the best by line_error.

    - Drop constraints with span < min_span (points are nearly coincident).
    - Sort remaining by line_error ascending.
    - Take the best k; compute reference_error = max(line_error among best k).
    - Keep constraints with line_error <= min(reference_error + eps, max_error_cap).

    min_span should be tied to the expected finder size in pixels. A few pixels
    is a sensible default; cross-ratio is unstable when the 4 points are nearly
    coincident because the denominator in the ratio is small relative to noise.
    """
    # Drop by span
    usable = [c for c in constraints if c.span >= min_span]
    if not usable:
        return []

    # Sort by line_error (lower = more colinear = better)
    usable.sort(key=lambda c: c.line_error)

    if len(usable) <= k:
        return usable

    best_k = usable[:k]
    reference_error = max(c.line_error for c in best_k)

    threshold = min(reference_error + eps, max_error_cap)
    return [c for c in usable if c.line_error <= threshold]


# ---------------------------------------------------------------------------
# Version estimation
# ---------------------------------------------------------------------------


def estimate_version(
    constraints: list[Constraint],
    v_range: range = range(1, 41),
) -> tuple[int, np.ndarray]:
    """Estimate the QR version from a list of filtered constraints.

    For each candidate V (N = 4V + 17), compute the per-constraint error as
    |log(r_measured / r_expected)|; score[V] = median(errors); return argmin.

    Returns (V_best, scores) where scores is an array of length len(v_range).
    """
    if not constraints:
        return 1, np.array([])

    scores = np.full(len(v_range), np.inf)
    Vs = list(v_range)

    for idx, V in enumerate(Vs):
        N = 4 * V + 17
        errors = []
        for c in constraints:
            r_expected = _expected_for_constraint(c, N)
            if r_expected is None or abs(r_expected) < 1e-12:
                continue
            err = abs(np.log(c.r_measured / r_expected))
            errors.append(err)

        if errors:
            scores[idx] = float(np.median(errors))

    if np.all(np.isinf(scores)):
        return Vs[0], scores

    best_idx = int(np.argmin(scores))
    return Vs[best_idx], scores


def _expected_for_constraint(c: Constraint, N: int) -> float | None:
    """Compute the expected cross-ratio for a constraint at a given N.

    Outer quadruples use positions 0, 7, N-7, N.
    Inner quadruples use positions 1, 6, N-6, N-1 (white ring at offset 1).
    """
    if c.type == "outer":
        return expected_cross_ratio(0.0, 7.0, float(N - 7), float(N))
    elif c.type == "inner":
        return expected_cross_ratio(1.0, 6.0, float(N - 6), float(N - 1))
    return None


# ---------------------------------------------------------------------------
# Debug / visualization helpers
# ---------------------------------------------------------------------------


def expected_cross_ratio_by_N(N: int) -> tuple[float, float]:
    """Convenience: return (outer, inner) expected cross-ratios for a given N."""
    outer = expected_cross_ratio(0.0, 7.0, float(N - 7), float(N))
    inner = expected_cross_ratio(1.0, 6.0, float(N - 6), float(N - 1))
    return outer, inner
