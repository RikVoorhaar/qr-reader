"""Data Block De-interleaving for QR Code Model 2.

Splits raw codeword bytes into error-correction blocks as defined
by the version + ECL table, then separates data from EC codewords.

In QR Code Model 2, the raw codewords are interleaved: for N blocks,
the stream is byte_0 of block_0, byte_0 of block_1, …, byte_{K-1} of
block_{N-1}, followed by the extra data byte for longer blocks,
followed by the EC codewords in the same interleaved order.

References:
- zxing-cpp QRDataBlock.cpp
- zxing-cpp QRECB.h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataBlock:
    """One error-correction block: data and EC codewords separated."""

    data: bytes
    ec: bytes

    def __repr__(self) -> str:
        return f"DataBlock(data={len(self.data)}B, ec={len(self.ec)}B)"


def deinterleave(
    raw: bytes,
    version: int,
    ecl: str,
) -> List[DataBlock]:
    """De-interleave raw codewords into EC blocks for a given version and ECL.

    Args:
        raw: Raw codeword bytes as they appear in the QR code.
        version: QR code version number (1–40).
        ecl: Error correction level as "L", "M", "Q", or "H".

    Returns:
        List of DataBlock instances, each containing separate data and EC bytes.

    Raises:
        ValueError: If the raw codeword count doesn't match the expected total,
                    or if the version is invalid.
    """
    from qr_reader.decoder.tables import VERSIONS, total_codewords

    if version < 1 or version > 40:
        raise ValueError(f"Invalid version: {version}")

    expected_total = total_codewords(version, ecl)
    if len(raw) != expected_total:
        raise ValueError(
            f"Raw codeword length {len(raw)} != expected {expected_total} "
            f"for version {version} ECL {ecl}"
        )

    ec_per_block, groups = VERSIONS[version].ec_info[ecl]

    # Flatten groups into individual block sizes (data_bytes per block).
    # Blocks may have one of two sizes (shorter / longer).
    block_data_sizes: List[int] = []
    for data_bytes, num_blocks in groups:
        block_data_sizes.extend([data_bytes] * num_blocks)

    total_blocks = len(block_data_sizes)

    if total_blocks == 0:
        return []

    # Determine which blocks are "shorter" vs "longer".
    min_data = min(block_data_sizes)
    longer_indices = [i for i, ds in enumerate(block_data_sizes) if ds > min_data]

    # Allocate codeword arrays for each block (data + EC interleaved form).
    codewords: List[List[int]] = []
    for ds in block_data_sizes:
        codewords.append([0] * (ds + ec_per_block))

    # --- De-interleave data bytes ---
    offset = 0

    # First pass: interleave byte[i] across all blocks, for i in 0..min_data-1.
    for i in range(min_data):
        for b in range(total_blocks):
            codewords[b][i] = raw[offset]
            offset += 1

    # Second pass: for longer blocks, fill the extra data byte at position min_data.
    for b in longer_indices:
        codewords[b][min_data] = raw[offset]
        offset += 1

    # --- De-interleave EC bytes ---
    shorter_total = min_data + ec_per_block  # total codewords in a short block

    for i in range(min_data, shorter_total):
        for b in range(total_blocks):
            if b in longer_indices:
                i_offset = i + 1  # skip extra data byte
            else:
                i_offset = i
            codewords[b][i_offset] = raw[offset]
            offset += 1

    # --- Split into data / EC ---
    result: List[DataBlock] = []
    for b in range(total_blocks):
        ds = block_data_sizes[b]
        result.append(
            DataBlock(
                data=bytes(codewords[b][:ds]),
                ec=bytes(codewords[b][ds:]),
            )
        )

    return result
