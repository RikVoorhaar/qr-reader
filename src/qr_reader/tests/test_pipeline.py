"""Tests for the end-to-end multi-triplet decode pipeline."""

from __future__ import annotations

import numpy as np

import qr_reader.pipeline as pipeline_module
from qr_reader.decoder.decoder import decode as decode_bits
from qr_reader.pipeline import DecodeAttempt, DecodeResult, decode_image
from qr_reader.qr_gen import generate_test_image, make_qr_image
from qr_reader.detector.sample import sample_qr_bits_spatial_threshold


def _two_qr_image(
    payload_left: str = "first payload",
    payload_right: str = "second payload",
    version_left: int = 2,
    version_right: int = 5,
) -> np.ndarray:
    """Composite two QR codes onto one canvas.

    The right-hand (higher-version) QR ranks first by cross-ratio score, so
    decoding the left payload forces the search past the first triplet.
    """
    left = make_qr_image(
        content=payload_left, version=version_left, box_size=10, border=4
    )
    right = make_qr_image(
        content=payload_right, version=version_right, box_size=10, border=4
    )
    height = max(left.shape[0], right.shape[0]) + 80
    width = left.shape[1] + right.shape[1] + 120
    canvas = np.full((height, width), 255, dtype=np.uint8)
    canvas[40 : 40 + left.shape[0], 40 : 40 + left.shape[1]] = left
    x_right = 80 + left.shape[1]
    canvas[40 : 40 + right.shape[0], x_right : x_right + right.shape[1]] = right
    return canvas


class TestDecodeImage:
    def test_clean_image_decodes(self):
        img = make_qr_image(content="hello pipeline", version=2, box_size=10, border=4)
        result = decode_image(img, expected_payload="hello pipeline")

        assert result.success
        assert result.text == "hello pipeline"
        assert result.version == 2
        assert result.detected_version == 2
        assert len(result.attempts) >= 1
        assert result.attempts[0].success

    def test_distorted_image_decodes(self):
        img = generate_test_image(
            version=1,
            content="distorted payload",
            rotation_angle_deg=20.0,
            perspective_max_shift=30.0,
            noise_std=40.0,
            seed=1,
        )
        result = decode_image(img, expected_payload="distorted payload")

        assert result.success
        assert result.text == "distorted payload"
        assert result.version == 1

    def test_expected_payload_mismatch_is_reported(self):
        img = make_qr_image(content="actual payload", version=2, box_size=10, border=4)
        result = decode_image(img, expected_payload="different payload")

        assert not result.success
        assert result.text is None
        assert result.attempts
        decoded = [a.decoded_text for a in result.attempts]
        assert "actual payload" in decoded
        mismatch = next(
            a for a in result.attempts if a.decoded_text == "actual payload"
        )
        assert "payload mismatch" in mismatch.failure_reason
        assert not mismatch.success

    def test_multi_triplet_decode(self):
        """The expected payload is found after rejecting a higher-ranked triplet."""
        img = _two_qr_image()
        result = decode_image(img, expected_payload="first payload")

        assert result.success
        assert result.text == "first payload"
        assert result.version == 2
        assert len(result.attempts) >= 3
        assert result.attempts[0].triplet_rank == 0
        assert result.attempts[0].failure_reason.startswith("payload mismatch")
        assert result.attempts[-1].success
        assert result.attempts[-1].triplet_rank >= 1

    def test_time_budget_zero_skips_all_attempts(self):
        img = make_qr_image(content="budget", version=1, box_size=10, border=4)
        result = decode_image(img, time_budget_s=0.0)

        assert not result.success
        assert result.attempts == []
        assert result.time_budget_exhausted
        assert result.n_candidates > 0
        assert not result.scale_retry_attempted

    def test_two_finder_failure_runs_one_scaled_standard_pass(self, monkeypatch):
        image = np.zeros((8, 10), dtype=np.uint8)
        scaled = np.zeros((10, 13), dtype=np.uint8)
        context = type("Context", (), {"img_gray": image})()
        calls = []
        scaled_h = np.array(
            [[2.0, 0.0, 4.0], [0.0, 3.0, 6.0], [0.0, 0.0, 1.0]]
        )

        def standard_pass(
            current, expected, max_triplets, max_versions, *, deadline, provenance="production"
        ):
            calls.append((current.shape, provenance))
            if len(calls) == 1:
                return (
                    DecodeResult(n_finder_patterns=2),
                    context,
                    [],
                    {},
                )
            retry = DecodeResult(
                success=True,
                text="payload",
                version=1,
                H=scaled_h,
                detection_ok=True,
                detected_version=1,
                attempts=[
                    DecodeAttempt(
                        triplet_rank=0,
                        top_left_idx=0,
                        top_right_idx=1,
                        bottom_left_idx=2,
                        version=1,
                        cross_ratio_score=0.0,
                        success=True,
                        decoded_text="payload",
                        provenance=provenance,
                    )
                ],
            )
            return retry, context, [], {}

        monkeypatch.setattr(pipeline_module, "_decode_standard_pass", standard_pass)
        monkeypatch.setattr(pipeline_module, "zoom", lambda *a, **k: scaled)

        result = decode_image(image, expected_payload="payload")

        assert result.success
        assert result.scale_retry_attempted
        assert result.attempts[0].provenance == "scale_retry"
        assert result.retry_n_finder_patterns == 0
        assert calls == [((8, 10), "production"), ((10, 13), "scale_retry")]
        sx, sy = 13 / 10, 10 / 8
        expected_map = np.array(
            [
                [1 / sx, 0, (1 / sx - 1) / 2],
                [0, 1 / sy, (1 / sy - 1) / 2],
                [0, 0, 1],
            ]
        )
        np.testing.assert_allclose(result.H, expected_map @ scaled_h)

    def test_non_two_finder_failure_does_not_scale(self, monkeypatch):
        image = np.zeros((8, 10), dtype=np.uint8)
        result_without_retry = DecodeResult(n_finder_patterns=1)

        def standard_pass(*args, **kwargs):
            return result_without_retry, None, None, {}

        monkeypatch.setattr(pipeline_module, "_decode_standard_pass", standard_pass)
        monkeypatch.setattr(
            pipeline_module,
            "zoom",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected zoom")),
        )

        result = decode_image(image)

        assert not result.success
        assert not result.scale_retry_attempted

    def test_spatial_threshold_sampler_decodes_clean_qr(self):
        version = 2
        modules = 17 + 4 * version
        image = make_qr_image(
            content="spatial threshold",
            version=version,
            box_size=10,
            border=4,
        )
        H = np.array(
            [[10.0, 0.0, 40.0], [0.0, 10.0, 40.0], [0.0, 0.0, 1.0]]
        )

        bits = sample_qr_bits_spatial_threshold(image, H, modules)

        assert decode_bits(bits) == "spatial threshold"

    def test_alignment_refinement_failure_is_traced_before_base_h_fallback(
        self, monkeypatch
    ):
        """The failed refined attempt remains visible if base-H sampling succeeds."""
        import qr_reader.pipeline as pipeline_module

        img = make_qr_image(
            content="refine failure", version=2, box_size=10, border=4
        )

        def fail_refinement(*args, **kwargs):
            raise RuntimeError("alignment refinement exploded")

        monkeypatch.setattr(
            pipeline_module, "build_refined_homography", fail_refinement
        )

        result = decode_image(img, expected_payload="refine failure")

        assert result.success
        assert result.text == "refine failure"
        assert result.attempts
        first_pass_failures = [
            attempt
            for attempt in result.attempts
            if attempt.provenance == "production"
            and "alignment refinement exploded" in attempt.failure_reason
        ]
        assert first_pass_failures
        fallback_success = next(
            attempt for attempt in result.attempts if attempt.success
        )
        assert fallback_success.provenance == "base_global_retry"
