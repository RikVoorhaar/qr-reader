"""Tests for decode module: OpenCV QR decode wrapper."""

import numpy as np
from qr_reader.decode import decode_qr

from qr_reader.qr_gen import make_qr_image


def test_decode_clean_qr_no_warp():
    """Decode a clean QR image with known content using true corners."""
    content = "Hello, QR!"
    img = make_qr_image(content=content, version=2, box_size=10, border=4)

    # For a clean QR with box_size=10 and border=4:
    # The QR region starts at (border * box_size, border * box_size) = (40, 40)
    # N = 4*V+17 = 25 modules, each box_size=10 pixels → 250 pixels wide
    # So corners in image: (40,40), (290,40), (290,290), (40,290)
    V = 2
    N = 4 * V + 17  # 25
    border_px = 4 * 10  # 40
    size_px = N * 10  # 250

    corners = np.array(
        [
            [border_px, border_px],  # TL
            [border_px + size_px, border_px],  # TR
            [border_px + size_px, border_px + size_px],  # BR
            [border_px, border_px + size_px],  # BL
        ],
        dtype=np.float32,
    )

    text, ok = decode_qr(img, corners)
    assert ok, f"Decode failed with corners: {corners}"
    assert text == content, f"Expected '{content}', got '{text}'"


def test_decode_returns_false_on_bad_corners():
    """Decode should return ok=False for bad corners."""
    img = make_qr_image(content="test", version=1, box_size=10, border=4)

    # Bad corners (tiny, wrong position)
    bad_corners = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
        ],
        dtype=np.float32,
    )

    text, ok = decode_qr(img, bad_corners)
    assert not ok, "Should fail with bad corners"
