"""End-to-end QR decoding with multi-triplet / multi-version attempts.

``decode_image`` runs the finder/triplet stage once, ranks every triplet by
its cross-ratio version score, and tries a bounded set of (triplet, version)
candidates. Failed first passes use a gated resize or base/refined-homography
sampling fallback without widening the candidate search. The first payload
that passes the optional ``expected_payload`` check wins; otherwise the result
carries every attempt and its failure reason.

The single-candidate detector API (``detect_corners``, ``detect_homography``,
``detect_sample``) is unchanged and remains the cheap path when only geometry
is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.ndimage import zoom

from qr_reader.decoder.decoder import decode as decode_bits
from qr_reader.detector.detector import (
    DetectionContext,
    DetectionResult,
    TripletCandidate,
    _fit_candidate_homography,
    _prepare_detection,
    _rank_triplets,
)
from qr_reader.detector.landmarks_refined import build_refined_homography
from qr_reader.detector.sample import (
    sample_qr_bits,
    sample_qr_bits_spatial_threshold,
)
from qr_reader.preprocess import to_gray


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
    provenance: str = "production"


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
    fallback_time_s: float = 0.0
    scale_retry_attempted: bool = False
    scale_retry_resize_time_s: float = 0.0
    retry_n_finder_patterns: int = 0
    retry_n_triplets: int = 0


def _decode_standard_pass(
    image: np.ndarray,
    expected_payload: str | None = None,
    max_triplets: int = 5,
    max_versions_per_triplet: int = 2,
    *,
    deadline: float,
    provenance: str = "production",
) -> tuple[
    DecodeResult,
    DetectionContext | None,
    list[TripletCandidate] | None,
    dict[tuple[int, int], tuple[np.ndarray, float]],
]:
    t_total = perf_counter()
    result = DecodeResult()
    context = None
    ranked = None
    base_homographies: dict[tuple[int, int], tuple[np.ndarray, float]] = {}

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
        return result, context, ranked, base_homographies

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
        return result, context, ranked, base_homographies
    result.prep_time_s += perf_counter() - t0

    budget_exhausted = False
    for triplet_rank, candidate in enumerate(ranked[:max_triplets]):
        for version in candidate.version_candidates[:max_versions_per_triplet]:
            if perf_counter() >= deadline:
                budget_exhausted = True
                break

            attempt = DecodeAttempt(
                triplet_rank=triplet_rank,
                top_left_idx=candidate.triplet.top_left_idx,
                top_right_idx=candidate.triplet.top_right_idx,
                bottom_left_idx=candidate.triplet.bottom_left_idx,
                version=version,
                cross_ratio_score=candidate.cross_ratio_score,
                provenance=provenance,
            )
            N = 4 * version + 17

            t0 = perf_counter()
            try:
                H, err = _fit_candidate_homography(context, candidate.triplet, N)
                base_homographies[(triplet_rank, version)] = (H, err)
                # Refine with interior alignment patterns (versions >= 2).
                # A refinement failure is allowed to fail this candidate; the
                # outer handler records it and the search moves on.
                H, _ref_stats = build_refined_homography(
                    context.img_gray, H, candidate.triplet,
                    context.fps, version,
                )
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
            result.error = None
            result.version = version
            result.H = H
            result.detected_version = version
            result.total_time_s = perf_counter() - t_total
            return result, context, ranked, base_homographies

        if budget_exhausted:
            break

    result.time_budget_exhausted = budget_exhausted
    result.total_time_s = perf_counter() - t_total
    return result, context, ranked, base_homographies


def _homography_to_original(
    H_scaled: np.ndarray,
    original_shape: tuple[int, ...],
    scaled_shape: tuple[int, ...],
) -> np.ndarray:
    """Map a QR-grid → zoomed-image homography back to original pixels."""
    original_h, original_w = original_shape[:2]
    scaled_h, scaled_w = scaled_shape[:2]
    sx, sy = scaled_w / original_w, scaled_h / original_h
    # grid_mode=True maps pixel centres by x' = scale * (x + .5) - .5.
    inverse_pixel_map = np.array(
        [
            [1.0 / sx, 0.0, (1.0 / sx - 1.0) / 2.0],
            [0.0, 1.0 / sy, (1.0 / sy - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return inverse_pixel_map @ H_scaled


def _run_sampling_fallback(
    result: DecodeResult,
    image: np.ndarray,
    context: DetectionContext,
    ranked: list[TripletCandidate],
    base_homographies: dict[tuple[int, int], tuple[np.ndarray, float]],
    expected_payload: str | None,
    max_triplets: int,
    max_versions_per_triplet: int,
    deadline: float,
) -> None:
    """Try base-H/global-threshold, then refined-H/spatial-threshold sampling."""
    candidates = [
        (rank, candidate, version)
        for rank, candidate in enumerate(ranked[:max_triplets])
        for version in candidate.version_candidates[:max_versions_per_triplet]
    ]

    for provenance in ("base_global_retry", "refined_spatial_retry"):
        for triplet_rank, candidate, version in candidates:
            if perf_counter() >= deadline:
                result.time_budget_exhausted = True
                return
            attempt = DecodeAttempt(
                triplet_rank=triplet_rank,
                top_left_idx=candidate.triplet.top_left_idx,
                top_right_idx=candidate.triplet.top_right_idx,
                bottom_left_idx=candidate.triplet.bottom_left_idx,
                version=version,
                cross_ratio_score=candidate.cross_ratio_score,
                provenance=provenance,
            )
            N = 4 * version + 17
            key = (triplet_rank, version)
            t0 = perf_counter()
            try:
                if key in base_homographies:
                    H_base, err = base_homographies[key]
                else:
                    H_base, err = _fit_candidate_homography(
                        context, candidate.triplet, N
                    )
                    base_homographies[key] = (H_base, err)
                H = H_base
                if provenance == "refined_spatial_retry":
                    H, _ = build_refined_homography(
                        image, H_base, candidate.triplet, context.fps, version
                    )
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
                if provenance == "base_global_retry":
                    bits = sample_qr_bits(image, H, N)
                else:
                    bits = sample_qr_bits_spatial_threshold(image, H, N)
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
            return


def decode_image(
    image: np.ndarray,
    expected_payload: str | None = None,
    max_triplets: int = 5,
    max_versions_per_triplet: int = 2,
    time_budget_s: float = 15.0,
) -> DecodeResult:
    """Decode an image with bounded candidate search and staged fallbacks.

    The normal production pass runs first. A failed image with exactly two
    fitted finder patterns gets one 1.25× linear SciPy zoom retry. Other
    ranked-triplet misses get base-H/global-threshold then refined-H/spatial-
    threshold sampling over the same bounded candidates and versions. All
    passes share one wall-clock budget.
    """
    t_total = perf_counter()
    deadline = t_total + max(0.0, time_budget_s)
    result, context, ranked, base_homographies = _decode_standard_pass(
        image,
        expected_payload,
        max_triplets,
        max_versions_per_triplet,
        deadline=deadline,
    )

    if result.success or perf_counter() >= deadline:
        result.time_budget_exhausted |= (
            not result.success and perf_counter() >= deadline
        )
        result.total_time_s = perf_counter() - t_total
        return result

    if result.n_finder_patterns == 2:
        result.scale_retry_attempted = True
        gray = context.img_gray if context is not None else to_gray(image)
        t0 = perf_counter()
        scaled = zoom(
            gray, 1.25, order=1, mode="reflect", grid_mode=True, prefilter=False
        )
        result.scale_retry_resize_time_s = perf_counter() - t0
        result.fallback_time_s += result.scale_retry_resize_time_s
        if perf_counter() < deadline:
            scaled_result, _, _, _ = _decode_standard_pass(
                scaled,
                expected_payload,
                max_triplets,
                max_versions_per_triplet,
                deadline=deadline,
                provenance="scale_retry",
            )
            result.retry_n_finder_patterns = scaled_result.n_finder_patterns
            result.retry_n_triplets = scaled_result.n_triplets
            result.prep_time_s += scaled_result.prep_time_s
            result.attempts.extend(scaled_result.attempts)
            result.fallback_time_s += scaled_result.total_time_s
            result.time_budget_exhausted |= scaled_result.time_budget_exhausted
            if scaled_result.success:
                result.success = True
                result.text = scaled_result.text
                result.error = None
                result.version = scaled_result.version
                result.H = _homography_to_original(
                    scaled_result.H, gray.shape, scaled.shape
                )
                result.detection_ok = scaled_result.detection_ok
                result.detected_version = scaled_result.detected_version
        else:
            result.time_budget_exhausted = True
    elif (
        result.n_triplets > 0
        and context is not None
        and ranked is not None
    ):
        t0 = perf_counter()
        _run_sampling_fallback(
            result,
            context.img_gray,
            context,
            ranked,
            base_homographies,
            expected_payload,
            max_triplets,
            max_versions_per_triplet,
            deadline,
        )
        result.fallback_time_s += perf_counter() - t0

    result.total_time_s = perf_counter() - t_total
    return result
