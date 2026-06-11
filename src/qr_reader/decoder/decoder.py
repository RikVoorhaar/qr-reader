"""Top-level QR Code decoder — orchestrates all phases.

Given a QR code bit matrix (True = dark module), extracts and decodes the
text content using the full pipeline: format info, version info, codeword
extraction, de-interleaving, Reed-Solomon error correction, and bit-stream
decoding.

Reference: docs/decoder-plan.md Phase 8
"""

from __future__ import annotations

import numpy as np

from qr_reader.decoder.bitstream import decode_bitstream
from qr_reader.decoder.codeword_extractor import extract_codewords
from qr_reader.decoder.data_block import deinterleave
from qr_reader.decoder.format_info import FormatInfoDecodeError, decode_format_info
from qr_reader.decoder.rs import rs_decode
from qr_reader.decoder.tables import ECL_NAMES
from qr_reader.decoder.version_info import decode_version_info


class DecodeError(Exception):
    """Top-level QR code decoding failure."""


def decode(matrix: np.ndarray) -> str:
    """Decode a QR code from its bit matrix.

    Args:
        matrix: 2D bool numpy array (True = dark module, False = light),
                shape (size, size) where size = 17 + 4*version.

    Returns:
        The decoded text string.

    Raises:
        ValueError: if the matrix is not square or version is out of range.
        DecodeError: if any decoding step fails.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square 2D array, got shape {matrix.shape}")

    size = matrix.shape[0]
    version = (size - 17) // 4
    if version < 1 or version > 40 or (size - 17) % 4 != 0:
        raise ValueError(
            f"Invalid QR symbol size {size} (version would be {version}, must be 1–40)"
        )

    # ── 1. Format information → ECL, mask ─────────────────────
    try:
        ecl_idx, mask_idx = decode_format_info(matrix, version)
    except FormatInfoDecodeError as e:
        raise DecodeError(f"Format info decode failed: {e}") from e

    ecl = ECL_NAMES[ecl_idx]

    # ── 2. Version information (v ≥ 7) — cross-check ──────────
    if version >= 7:
        version2 = decode_version_info(matrix)
        if version2 is not None and version2 != version:
            raise DecodeError(
                f"Version mismatch: symbol size implies v{version}, "
                f"but version info decodes as v{version2}"
            )

    # ── 3. Extract codewords (unmask + zigzag) ────────────────
    try:
        raw = extract_codewords(matrix, version, mask_idx)
    except ValueError as e:
        raise DecodeError(f"Codeword extraction failed: {e}") from e

    # ── 4. De-interleave into data blocks ─────────────────────
    try:
        blocks = deinterleave(raw, version, ecl)
    except ValueError as e:
        raise DecodeError(f"Data-block de-interleaving failed: {e}") from e

    # ── 5. Reed-Solomon error correction ──────────────────────
    corrected_data: list[int] = []
    for blk in blocks:
        combined = list(blk.data) + list(blk.ec)
        num_ec = len(blk.ec)
        corrected = rs_decode(combined, num_ec)
        if corrected is None:
            raise DecodeError(
                f"RS error correction failed for version {version} ECL {ecl} "
                f"(block with {len(blk.data)} data + {num_ec} EC bytes)"
            )
        # Append only the corrected data bytes (discard EC bytes).
        corrected_data.extend(corrected[: len(blk.data)])

    # ── 6. Decode bit stream → text ───────────────────────────
    try:
        text = decode_bitstream(corrected_data, version)
    except ValueError as e:
        raise DecodeError(f"Bit-stream decoding failed: {e}") from e

    return text
