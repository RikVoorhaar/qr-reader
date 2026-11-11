"""Tests for the end-to-end multi-triplet decode pipeline."""

from __future__ import annotations

import numpy as np

from qr_reader.pipeline import decode_image
from qr_reader.qr_gen import generate_test_image, make_qr_image


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
