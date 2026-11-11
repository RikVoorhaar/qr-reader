"""End-to-end QR decoding with multi-triplet / multi-version attempts.

``decode_image`` runs the finder/triplet stage once, ranks every triplet by
its cross-ratio version score, and tries a bounded set of (triplet, version)
candidates.  Each candidate is rectified with the 12-corner DLT + LM
homography, sampled, and decoded.  The first payload that passes the optional
``expected_payload`` check wins; otherwise the result carries every attempt
and its failure reason.

The single-candidate detector API (``detect_corners``, ``detect_homography``,
``detect_sample``) is unchanged and remains the cheap path when only geometry
is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from qr_reader.decoder.decoder import decode as decode_bits
from qr_reader.detector.detector import (
    DetectionResult,
    _fit_candidate_homography,
    _prepare_detection,
    _rank_triplets,
)
from qr_reader.detector.sample import sample_qr_bits


@dataclass
class DecodeAttempt:
    """One (triplet, version) trial and why it did or did not produce a payload."""

    triplet_rank: int
    top_left_idx: int
    top_right_idx: int
    bottom_left_idx: int
    version: int
    cross_ratio_score: float
    homography_ok: bool = False
    homography_error_px: float = -1.0
    fit_time_s: float = 0.0
    sample_time_s: float = 0.0
    decode_time_s: float = 0.0
    decoded_text: str | None = None
    success: bool = False
    failure_reason: str = ""


@dataclass
class DecodeResult:
    """Outcome of :func:`decode_image`.

    ``success`` is true only when a decoded payload matched
    ``expected_payload`` (or any payload when it was ``None``).  ``H`` and
    ``version`` describe the successful candidate, or the first candidate that
    produced a homography when no payload decoded.  ``attempts`` retains the
    full search trace.
    """

    success: bool = False
    text: str | None = None
    version: int | None = None
    H: np.ndarray | None = None
    error: Exception | None = None
    attempts: list[DecodeAttempt] = field(default_factory=list)
    n_candidates: int = 0
    n_finder_patterns: int = 0
    n_triplets: int = 0
    detection_ok: bool = False
    detected_version: int | None = None
    prep_time_s: float = 0.0
    total_time_s: float = 0.0
    time_budget_exhausted: bool = False


def decode_image(
    image: np.ndarray,
    expected_payload: str | None = None,
    max_triplets: int = 5,
    max_versions_per_triplet: int = 2,
    time_budget_s: float = 15.0,
) -> DecodeResult:
    """Detect and decode a single-QR image, trying several triplets/versions.

    Args:
        image: Grayscale or colour image containing one QR code.
        expected_payload: If given, only an exact match counts as success.  A
            non-matching decode is recorded in ``attempts`` and the search
            continues.
        max_triplets: Maximum number of ranked triplets to try.
        max_versions_per_triplet: Maximum cross-ratio-ranked versions to try
            per triplet.
        time_budget_s: Wall-clock budget for the whole call.  The pipeline
            checks the budget before starting each candidate; an in-flight
            sample/decode is allowed to finish.

    Returns:
        A :class:`DecodeResult`.  Structural failures (no alignment patterns,
        no triplet, empty cross-ratio constraints, etc.) are reported in
        ``error`` rather than raised.
    """
    t_total = perf_counter()
    result = DecodeResult()

    detection = DetectionResult(success=False)
    t0 = perf_counter()
    try:
        context = _prepare_detection(image, detection)
    except Exception as exc:
        result.error = exc
        result.n_candidates = detection.n_candidates
        result.n_finder_patterns = detection.n_finder_patterns
        result.n_triplets = detection.n_triplets
        result.prep_time_s = perf_counter() - t0
        result.total_time_s = perf_counter() - t_total
        return result

    result.n_candidates = detection.n_candidates
    result.n_finder_patterns = detection.n_finder_patterns
    result.n_triplets = detection.n_triplets
    result.prep_time_s = perf_counter() - t0

    t0 = perf_counter()
    try:
        ranked = _rank_triplets(context)
    except Exception as exc:
        result.error = exc
        result.prep_time_s += perf_counter() - t0
        result.total_time_s = perf_counter() - t_total
        return result
    result.prep_time_s += perf_counter() - t0

    budget_exhausted = False
    for triplet_rank, candidate in enumerate(ranked[:max_triplets]):
        for version in candidate.version_candidates[:max_versions_per_triplet]:
            if perf_counter() - t_total >= time_budget_s:
                budget_exhausted = True
                break

            attempt = DecodeAttempt(
                triplet_rank=triplet_rank,
                top_left_idx=candidate.triplet.top_left_idx,
                top_right_idx=candidate.triplet.top_right_idx,
                bottom_left_idx=candidate.triplet.bottom_left_idx,
                version=version,
                cross_ratio_score=candidate.cross_ratio_score,
            )
            N = 4 * version + 17

            t0 = perf_counter()
            try:
                H, err = _fit_candidate_homography(context, candidate.triplet, N)
            except Exception as exc:
                attempt.fit_time_s = perf_counter() - t0
                attempt.failure_reason = (
                    f"homography failed: {type(exc).__name__}: {exc}"
                )
                result.attempts.append(attempt)
                continue

            attempt.fit_time_s = perf_counter() - t0
            attempt.homography_ok = True
            attempt.homography_error_px = err

            if not result.detection_ok:
                result.detection_ok = True
                result.H = H
                result.detected_version = version

            t0 = perf_counter()
            try:
                bits = sample_qr_bits(context.img_gray, H, N)
            except Exception as exc:
                attempt.sample_time_s = perf_counter() - t0
                attempt.failure_reason = (
                    f"sampling failed: {type(exc).__name__}: {exc}"
                )
                result.attempts.append(attempt)
                continue

            attempt.sample_time_s = perf_counter() - t0

            t0 = perf_counter()
            try:
                text = decode_bits(bits)
            except Exception as exc:
                attempt.decode_time_s = perf_counter() - t0
                attempt.failure_reason = (
                    f"decode failed: {type(exc).__name__}: {exc}"
                )
                result.attempts.append(attempt)
                continue

            attempt.decode_time_s = perf_counter() - t0
            attempt.decoded_text = text

            if expected_payload is not None and text != expected_payload:
                attempt.failure_reason = (
                    f"payload mismatch: decoded {text!r}, "
                    f"expected {expected_payload!r}"
                )
                result.attempts.append(attempt)
                continue

            attempt.success = True
            result.attempts.append(attempt)
            result.success = True
            result.text = text
            result.version = version
            result.H = H
            result.detected_version = version
            result.total_time_s = perf_counter() - t_total
            return result

        if budget_exhausted:
            break

    result.time_budget_exhausted = budget_exhausted
    result.total_time_s = perf_counter() - t_total
    return result
