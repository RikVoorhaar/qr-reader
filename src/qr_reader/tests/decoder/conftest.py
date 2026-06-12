"""Shared test fixtures and helpers for decoder tests."""

from __future__ import annotations

import numpy as np
import qrcode


def make_qr_bitmatrix(
    content: str,
    version: int = 1,
    ecl: str = "L",
    mask: int | None = None,
) -> np.ndarray:
    """Return a bool 2D array of QR code modules (True = dark).

    Args:
        content: Text to encode.
        version: QR version (1–40).
        ecl: Error correction level — "L", "M", "Q", or "H".
        mask: Force a specific mask pattern (0–7).  If None, the library
              chooses the best mask automatically.

    Returns:
        2-D numpy bool array of shape ``(size, size)`` where black modules
        are True.
    """
    ecl_map = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    qr = qrcode.QRCode(
        version=version,
        error_correction=ecl_map[ecl],
        box_size=1,
        border=0,
    )
    qr.add_data(content)
    if mask is None:
        qr.make()
    else:
        qr.make(mask_pattern=mask)
    return np.array(qr.modules, dtype=bool)
