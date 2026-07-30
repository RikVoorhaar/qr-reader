"""Shared test helpers for QR decoder unit tests."""

from __future__ import annotations

import numpy as np
import qrcode


def make_qr_bitmatrix(
    content: str,
    version: int | None = None,
    ecl: str = "L",
    *,
    mask: int | None = None,
) -> np.ndarray:
    """Return a bool 2D array of QR modules (True = dark).

    Args:
        content: The data to encode.
        version: QR version 1–40, or None for auto-fit.
        ecl: Error correction level — 'L', 'M', 'Q', or 'H'.
        mask: Specific mask pattern 0–7, or None to let qrcode pick best.

    Returns:
        2D numpy bool array, shape (size, size), borderless (no quiet zone).
    """
    ecl_map = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }

    extra = {}
    if version is not None:
        extra["version"] = version
    if mask is not None:
        extra["mask_pattern"] = mask

    qr = qrcode.QRCode(
        error_correction=ecl_map[ecl],
        border=0,
        **extra,
    )
    qr.add_data(content)
    qr.make(fit=False)
    return np.array(qr.modules, dtype=bool).T
