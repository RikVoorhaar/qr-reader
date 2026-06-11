"""Thin wrapper around OpenCV QRCodeDetector for decoding.

Corners are expected in (x, y) order as [TL, TR, BR, BL].
"""

import cv2
import numpy as np


def decode_qr(image: np.ndarray, corners_xy: np.ndarray) -> tuple[str, bool]:
    """Decode a QR code using OpenCV's QRCodeDetector.

    ``image``: grayscale uint8 image (0–255). Do NOT pass a boolean binary.
    ``corners_xy``: (4, 2) float32 in [TL, TR, BR, BL] order, (x, y).

    Returns (decoded_text, ok) where ok is True if decode succeeded.
    """
    detector = cv2.QRCodeDetector()
    # OpenCV expects (1, 4, 2) or (4, 1, 2) shape
    points = corners_xy.astype(np.float32).reshape(1, 4, 2)
    text, straight_qrcode = detector.decode(image, points)
    ok = text != ""
    return text, ok
