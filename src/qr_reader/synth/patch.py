"""Phase 1 — QR patch and mask generation.

Functions
---------
generate_qr_patch
    Produce a clean binary QR code image and an all-ones mask.
compute_qr_corners_patch_space
    Compute the four corners of the QR code proper (excluding quiet zone) in
    patch-space coordinates.
"""

from __future__ import annotations

import numpy as np
import qrcode
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

__all__ = [
    "generate_qr_patch",
    "compute_qr_corners_patch_space",
]

# ---------------------------------------------------------------------------
# ECL string → qrcode constant
# ---------------------------------------------------------------------------

_ECL_MAP: dict[str, int] = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}

VALID_ECL = frozenset(_ECL_MAP)


def _resolve_ecl(ecl_str: str) -> int:
    """Convert a one-letter ECL string to the corresponding ``qrcode`` constant."""
    try:
        return _ECL_MAP[ecl_str]
    except KeyError:
        msg = f"Unknown ECL {ecl_str!r}; expected one of {sorted(VALID_ECL)}"
        raise ValueError(msg) from None


# ---------------------------------------------------------------------------
# 1.1  generate_qr_patch
# ---------------------------------------------------------------------------


def generate_qr_patch(
    version: int,
    content: str,
    ecl_str: str,
    ppm: int,
    quiet_zone_modules: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean binary QR code patch with quiet zone and an all-ones mask.

    Parameters
    ----------
    version : int
        QR code version (1–40).  The caller must ensure *content* fits in this
        version at the given ECL.
    content : str
        Text payload to encode.
    ecl_str : str
        Error correction level — one of ``"L"``, ``"M"``, ``"Q"``, ``"H"``.
    ppm : int
        Pixels per module (passed as ``box_size`` to the ``qrcode`` library).
        Must be a positive integer.
    quiet_zone_modules : int
        Width of the quiet zone in modules (default ``4``, the QR spec minimum).

    Returns
    -------
    patch_rgb : np.ndarray, shape ``(H, H, 3)``, dtype ``uint8``
        3-channel RGB image where black modules are 0 and white modules are 255.
        ``H = (N + 2 * quiet_zone_modules) * ppm``.
    mask : np.ndarray, shape ``(H, H)``, dtype ``float32``
        All-ones mask (every pixel = 1.0) matching the patch spatial dimensions.
    """
    ecl = _resolve_ecl(ecl_str)

    qr = qrcode.QRCode(
        version=version,
        error_correction=ecl,
        box_size=ppm,
        border=quiet_zone_modules,
    )
    qr.add_data(content)
    qr.make(fit=False)  # use the specified version exactly

    # Render to uint8 PIL → numpy
    pil_img = qr.make_image()
    binary: np.ndarray = np.array(pil_img, dtype=np.uint8) * 255

    # Promote grayscale (H, W) to 3-channel RGB
    patch_rgb = np.stack([binary, binary, binary], axis=-1)

    # Solid white mask
    H, W = patch_rgb.shape[:2]
    mask = np.ones((H, W), dtype=np.float32)

    return patch_rgb, mask


# ---------------------------------------------------------------------------
# 1.2  compute_qr_corners_patch_space
# ---------------------------------------------------------------------------


def compute_qr_corners_patch_space(
    quiet_zone_modules: int,
    N: int,
    ppm: int,
) -> np.ndarray:
    """Compute the four corners of the QR code proper in patch-space coordinates.

    The "QR code proper" excludes the quiet zone.  Corners are returned in
    TL, TR, BR, BL order and use **(x, y)** convention (matching OpenCV).

    Parameters
    ----------
    quiet_zone_modules : int
        Width of the quiet zone in modules.
    N : int
        Number of modules along one side of the QR code (``N = 17 + 4 * version``).
    ppm : int
        Pixels per module (same value passed to ``generate_qr_patch``).

    Returns
    -------
    corners : np.ndarray, shape ``(4, 2)``, dtype ``float64``
        ``[[TL_x, TL_y], [TR_x, TR_y], [BR_x, BR_y], [BL_x, BL_y]]`` in
        patch-space (x, y) coordinates.
    """
    qz = quiet_zone_modules
    inner = N * ppm
    offset = qz * ppm

    corners = np.array(
        [
            [offset, offset],  # TL
            [offset + inner, offset],  # TR
            [offset + inner, offset + inner],  # BR
            [offset, offset + inner],  # BL
        ],
        dtype=np.float64,
    )
    return corners
