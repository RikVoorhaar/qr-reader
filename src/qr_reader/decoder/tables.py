"""QR Code spec tables — version info, EC blocks, format/version BCH patterns,
masks, character counts, and alignment positions.

All data is hard-coded from ISO 18004 and cross-referenced against
zxing-cpp QRVersion.cpp, nayuki qrcodegen.py, and OpenCV encoder tables.

No logic here — pure data tables used by the rest of the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────
# ECL enum (matching the 2-bit format info encoding)
# ──────────────────────────────────────────────────────────────

ECL_L = 0  # 0b01 in format info
ECL_M = 1  # 0b00
ECL_Q = 2  # 0b11
ECL_H = 3  # 0b10

# The actual 2-bit encoding in the format info (low 2 bits after unmasking)
ECL_TO_FORMAT_BITS = {ECL_L: 1, ECL_M: 0, ECL_Q: 3, ECL_H: 2}
FORMAT_BITS_TO_ECL = {1: ECL_L, 0: ECL_M, 3: ECL_Q, 2: ECL_H}

ECL_NAMES = {ECL_L: "L", ECL_M: "M", ECL_Q: "Q", ECL_H: "H"}


# ──────────────────────────────────────────────────────────────
# 1. Version info table
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VersionInfo:
    """Static data for one QR code version (1–40)."""

    version: int  # 1–40
    symbol_size: int  # modules per side = 17 + 4*version
    alignment_positions: list[int] = field(default_factory=list)
    # EC block layout per ECL: (ec_codewords_per_block, [(data_bytes, num_blocks), ...])
    ec_info: dict[str, tuple[int, list[tuple[int, int]]]] = field(default_factory=dict)


def _ec_info(raw: list[int]) -> tuple[int, list[tuple[int, int]]]:
    """Parse the zxing-style 5-int-per-ECL encoding.
    raw = [ec_per_block, g1_blocks, g1_data, g2_blocks, g2_data]
    """
    ec = raw[0]
    groups = []
    for i in range(1, len(raw), 2):
        if raw[i] > 0:
            groups.append((raw[i + 1], raw[i]))
    return ec, groups


_VERSIONS_RAW: list[tuple[int, list[int], list[int]]] = [
    (1, [], [7, 1, 19, 0, 0, 10, 1, 16, 0, 0, 13, 1, 13, 0, 0, 17, 1, 9, 0, 0]),
    (2, [6, 18], [10, 1, 34, 0, 0, 16, 1, 28, 0, 0, 22, 1, 22, 0, 0, 28, 1, 16, 0, 0]),
    (3, [6, 22], [15, 1, 55, 0, 0, 26, 1, 44, 0, 0, 18, 2, 17, 0, 0, 22, 2, 13, 0, 0]),
    (4, [6, 26], [20, 1, 80, 0, 0, 18, 2, 32, 0, 0, 26, 2, 24, 0, 0, 16, 4, 9, 0, 0]),
    (
        5,
        [6, 30],
        [26, 1, 108, 0, 0, 24, 2, 43, 0, 0, 18, 2, 15, 2, 16, 22, 2, 11, 2, 12],
    ),
    (6, [6, 34], [18, 2, 68, 0, 0, 16, 4, 27, 0, 0, 24, 4, 19, 0, 0, 28, 4, 15, 0, 0]),
    (
        7,
        [6, 22, 38],
        [20, 2, 78, 0, 0, 18, 4, 31, 0, 0, 18, 2, 14, 4, 15, 26, 4, 13, 1, 14],
    ),
    (
        8,
        [6, 24, 42],
        [24, 2, 97, 0, 0, 22, 2, 38, 2, 39, 22, 4, 18, 2, 19, 26, 4, 14, 2, 15],
    ),
    (
        9,
        [6, 26, 46],
        [30, 2, 116, 0, 0, 22, 3, 36, 2, 37, 20, 4, 16, 4, 17, 24, 4, 12, 4, 13],
    ),
    (
        10,
        [6, 28, 50],
        [18, 2, 68, 2, 69, 26, 4, 43, 1, 44, 24, 6, 19, 2, 20, 28, 6, 15, 2, 16],
    ),
    (
        11,
        [6, 30, 54],
        [20, 4, 81, 0, 0, 30, 1, 50, 4, 51, 28, 4, 22, 4, 23, 24, 3, 12, 8, 13],
    ),
    (
        12,
        [6, 32, 58],
        [24, 2, 92, 2, 93, 22, 6, 36, 2, 37, 26, 4, 20, 6, 21, 28, 7, 14, 4, 15],
    ),
    (
        13,
        [6, 34, 62],
        [26, 4, 107, 0, 0, 22, 8, 37, 1, 38, 24, 8, 20, 4, 21, 22, 12, 11, 4, 12],
    ),
    (
        14,
        [6, 26, 46, 66],
        [30, 3, 115, 1, 116, 24, 4, 40, 5, 41, 20, 11, 16, 5, 17, 24, 11, 12, 5, 13],
    ),
    (
        15,
        [6, 26, 48, 70],
        [22, 5, 87, 1, 88, 24, 5, 41, 5, 42, 30, 5, 24, 7, 25, 24, 11, 12, 7, 13],
    ),
    (
        16,
        [6, 26, 50, 74],
        [24, 5, 98, 1, 99, 28, 7, 45, 3, 46, 24, 15, 19, 2, 20, 30, 3, 15, 13, 16],
    ),
    (
        17,
        [6, 30, 54, 78],
        [28, 1, 107, 5, 108, 28, 10, 46, 1, 47, 28, 1, 22, 15, 23, 28, 2, 14, 17, 15],
    ),
    (
        18,
        [6, 30, 56, 82],
        [30, 5, 120, 1, 121, 26, 9, 43, 4, 44, 28, 17, 22, 1, 23, 28, 2, 14, 19, 15],
    ),
    (
        19,
        [6, 30, 58, 86],
        [28, 3, 113, 4, 114, 26, 3, 44, 11, 45, 26, 17, 21, 4, 22, 26, 9, 13, 16, 14],
    ),
    (
        20,
        [6, 34, 62, 90],
        [28, 3, 107, 5, 108, 26, 3, 41, 13, 42, 30, 15, 24, 5, 25, 28, 15, 15, 10, 16],
    ),
    (
        21,
        [6, 28, 50, 72, 94],
        [28, 4, 116, 4, 117, 26, 17, 42, 0, 0, 28, 17, 22, 6, 23, 30, 19, 16, 6, 17],
    ),
    (
        22,
        [6, 26, 50, 74, 98],
        [28, 2, 111, 7, 112, 28, 17, 46, 0, 0, 30, 7, 24, 16, 25, 24, 34, 13, 0, 0],
    ),
    (
        23,
        [6, 30, 54, 78, 102],
        [30, 4, 121, 5, 122, 28, 4, 47, 14, 48, 30, 11, 24, 14, 25, 30, 16, 15, 14, 16],
    ),
    (
        24,
        [6, 28, 54, 80, 106],
        [30, 6, 117, 4, 118, 28, 6, 45, 14, 46, 30, 11, 24, 16, 25, 30, 30, 16, 2, 17],
    ),
    (
        25,
        [6, 32, 58, 84, 110],
        [26, 8, 106, 4, 107, 28, 8, 47, 13, 48, 30, 7, 24, 22, 25, 30, 22, 15, 13, 16],
    ),
    (
        26,
        [6, 30, 58, 86, 114],
        [28, 10, 114, 2, 115, 28, 19, 46, 4, 47, 28, 28, 22, 6, 23, 30, 33, 16, 4, 17],
    ),
    (
        27,
        [6, 34, 62, 90, 118],
        [30, 8, 122, 4, 123, 28, 22, 45, 3, 46, 30, 8, 23, 26, 24, 30, 12, 15, 28, 16],
    ),
    (
        28,
        [6, 26, 50, 74, 98, 122],
        [30, 3, 117, 10, 118, 28, 3, 45, 23, 46, 30, 4, 24, 31, 25, 30, 11, 15, 31, 16],
    ),
    (
        29,
        [6, 30, 54, 78, 102, 126],
        [30, 7, 116, 7, 117, 28, 21, 45, 7, 46, 30, 1, 23, 37, 24, 30, 19, 15, 26, 16],
    ),
    (
        30,
        [6, 26, 52, 78, 104, 130],
        [
            30,
            5,
            115,
            10,
            116,
            28,
            19,
            47,
            10,
            48,
            30,
            15,
            24,
            25,
            25,
            30,
            23,
            15,
            25,
            16,
        ],
    ),
    (
        31,
        [6, 30, 56, 82, 108, 134],
        [30, 13, 115, 3, 116, 28, 2, 46, 29, 47, 30, 42, 24, 1, 25, 30, 23, 15, 28, 16],
    ),
    (
        32,
        [6, 34, 60, 86, 112, 138],
        [30, 17, 115, 0, 0, 28, 10, 46, 23, 47, 30, 10, 24, 35, 25, 30, 19, 15, 35, 16],
    ),
    (
        33,
        [6, 30, 58, 86, 114, 142],
        [
            30,
            17,
            115,
            1,
            116,
            28,
            14,
            46,
            21,
            47,
            30,
            29,
            24,
            19,
            25,
            30,
            11,
            15,
            46,
            16,
        ],
    ),
    (
        34,
        [6, 34, 62, 90, 118, 146],
        [30, 13, 115, 6, 116, 28, 14, 46, 23, 47, 30, 44, 24, 7, 25, 30, 59, 16, 1, 17],
    ),
    (
        35,
        [6, 30, 54, 78, 102, 126, 150],
        [
            30,
            12,
            121,
            7,
            122,
            28,
            12,
            47,
            26,
            48,
            30,
            39,
            24,
            14,
            25,
            30,
            22,
            15,
            41,
            16,
        ],
    ),
    (
        36,
        [6, 24, 50, 76, 102, 128, 154],
        [30, 6, 121, 14, 122, 28, 6, 47, 34, 48, 30, 46, 24, 10, 25, 30, 2, 15, 64, 16],
    ),
    (
        37,
        [6, 28, 54, 80, 106, 132, 158],
        [
            30,
            17,
            122,
            4,
            123,
            28,
            29,
            46,
            14,
            47,
            30,
            49,
            24,
            10,
            25,
            30,
            24,
            15,
            46,
            16,
        ],
    ),
    (
        38,
        [6, 32, 58, 84, 110, 136, 162],
        [
            30,
            4,
            122,
            18,
            123,
            28,
            13,
            46,
            32,
            47,
            30,
            48,
            24,
            14,
            25,
            30,
            42,
            15,
            32,
            16,
        ],
    ),
    (
        39,
        [6, 26, 54, 82, 110, 138, 166],
        [
            30,
            20,
            117,
            4,
            118,
            28,
            40,
            47,
            7,
            48,
            30,
            43,
            24,
            22,
            25,
            30,
            10,
            15,
            67,
            16,
        ],
    ),
    (
        40,
        [6, 30, 58, 86, 114, 142, 170],
        [
            30,
            19,
            118,
            6,
            119,
            28,
            18,
            47,
            31,
            48,
            30,
            34,
            24,
            34,
            25,
            30,
            20,
            15,
            61,
            16,
        ],
    ),
]

ECL_NAMES_LIST = ["L", "M", "Q", "H"]

VERSIONS: dict[int, VersionInfo] = {}
for v_num, align, blocks in _VERSIONS_RAW:
    ec_info = {}
    for ecl_idx, ecl_name in enumerate(ECL_NAMES_LIST):
        ec_info[ecl_name] = _ec_info(blocks[ecl_idx * 5 : (ecl_idx + 1) * 5])
    VERSIONS[v_num] = VersionInfo(
        version=v_num,
        symbol_size=17 + 4 * v_num,
        alignment_positions=list(align),
        ec_info=ec_info,
    )


def total_codewords(version: int, ecl: str) -> int:
    """Total number of data + EC codewords for a given version and ECL."""
    ec_per_block, groups = VERSIONS[version].ec_info[ecl]
    total = 0
    for data_bytes, num_blocks in groups:
        total += num_blocks * (data_bytes + ec_per_block)
    return total


# ──────────────────────────────────────────────────────────────
# 2. Format information table — 32 valid 15-bit patterns
# ──────────────────────────────────────────────────────────────

# Generated by BCH(15,5) with generator 0x537, XOR mask 0x5412.
# Key: (ecl_index, mask_index) → 15-bit pattern
# Reference: nayuki qrcodegen.py _draw_format_bits()

_FORMAT_INFO_RAW = {
    (0, 0): 0x77C4,  # L, mask 0
    (0, 1): 0x72F3,  # L, mask 1
    (0, 2): 0x7DAA,  # L, mask 2
    (0, 3): 0x789D,  # L, mask 3
    (0, 4): 0x662F,  # L, mask 4
    (0, 5): 0x6318,  # L, mask 5
    (0, 6): 0x6C41,  # L, mask 6
    (0, 7): 0x6976,  # L, mask 7
    (1, 0): 0x5412,  # M, mask 0
    (1, 1): 0x5125,  # M, mask 1
    (1, 2): 0x5E7C,  # M, mask 2
    (1, 3): 0x5B4B,  # M, mask 3
    (1, 4): 0x45F9,  # M, mask 4
    (1, 5): 0x40CE,  # M, mask 5
    (1, 6): 0x4F97,  # M, mask 6
    (1, 7): 0x4AA0,  # M, mask 7
    (2, 0): 0x355F,  # Q, mask 0
    (2, 1): 0x3068,  # Q, mask 1
    (2, 2): 0x3F31,  # Q, mask 2
    (2, 3): 0x3A06,  # Q, mask 3
    (2, 4): 0x24B4,  # Q, mask 4
    (2, 5): 0x2183,  # Q, mask 5
    (2, 6): 0x2EDA,  # Q, mask 6
    (2, 7): 0x2BED,  # Q, mask 7
    (3, 0): 0x1689,  # H, mask 0
    (3, 1): 0x13BE,  # H, mask 1
    (3, 2): 0x1CE7,  # H, mask 2
    (3, 3): 0x19D0,  # H, mask 3
    (3, 4): 0x0762,  # H, mask 4
    (3, 5): 0x0255,  # H, mask 5
    (3, 6): 0x0D0C,  # H, mask 6
    (3, 7): 0x083B,  # H, mask 7
}


# List of the 32 valid 15-bit format information patterns (as ints)
VALID_FORMAT_PATTERNS: list[int] = list(_FORMAT_INFO_RAW.values())

# Map: pattern → (ecl_index, mask_index)
FORMAT_PATTERN_TO_INFO: dict[int, tuple[int, int]] = {
    v: k for k, v in _FORMAT_INFO_RAW.items()
}


# ──────────────────────────────────────────────────────────────
# 3. Version information table — 34 valid 18-bit patterns (v7–v40)
# ──────────────────────────────────────────────────────────────

# Generated by BCH(18,6) with generator 0x1F25.
# Reference: zxing QRVersion.cpp VERSION_DECODE_INFO, nayuki qrcodegen.py _draw_version()

VERSION_INFO_PATTERNS: list[int] = [
    0x07C94,  # v7
    0x085BC,  # v8
    0x09A99,  # v9
    0x0A4D3,  # v10
    0x0BBF6,  # v11
    0x0C762,  # v12
    0x0D847,  # v13
    0x0E60D,  # v14
    0x0F928,  # v15
    0x10B78,  # v16
    0x1145D,  # v17
    0x12A17,  # v18
    0x13532,  # v19
    0x149A6,  # v20
    0x15683,  # v21
    0x168C9,  # v22
    0x177EC,  # v23
    0x18EC4,  # v24
    0x191E1,  # v25
    0x1AFAB,  # v26
    0x1B08E,  # v27
    0x1CC1A,  # v28
    0x1D33F,  # v29
    0x1ED75,  # v30
    0x1F250,  # v31
    0x209D5,  # v32
    0x216F0,  # v33
    0x228BA,  # v34
    0x2379F,  # v35
    0x24B0B,  # v36
    0x2542E,  # v37
    0x26A64,  # v38
    0x27541,  # v39
    0x28C69,  # v40
]

# Map: 18-bit pattern → version number (7–40)
VERSION_PATTERN_TO_VERSION: dict[int, int] = {
    p: v for v, p in enumerate(VERSION_INFO_PATTERNS, start=7)
}

# Set of valid patterns for quick membership testing
VALID_VERSION_PATTERNS: set[int] = set(VERSION_INFO_PATTERNS)


# ──────────────────────────────────────────────────────────────
# 4. Character count indicator bit lengths
# ──────────────────────────────────────────────────────────────

# Per mode per version range.
# Reference: ISO 18004 Table 3, zxing QRCodecMode.cpp

CHAR_COUNT_BITS = {
    "numeric": {1: 10, 2: 12, 3: 14},  # v 1-9, 10-26, 27-40
    "alphanumeric": {1: 9, 2: 11, 3: 13},
    "byte": {1: 8, 2: 16, 3: 16},
}


def char_count_version_range(version: int) -> int:
    """Return the version range category: 1 (v1–9), 2 (v10–26), 3 (v27–40)."""
    if version <= 9:
        return 1
    elif version <= 26:
        return 2
    else:
        return 3


# Mode indicator bits (4 bits)
MODE_INDICATORS = {
    "terminator": 0x0,
    "numeric": 0x1,
    "alphanumeric": 0x2,
    "byte": 0x4,
}

MODE_BY_INDICATOR = {v: k for k, v in MODE_INDICATORS.items()}


# ──────────────────────────────────────────────────────────────
# 5. Mask functions (0–7)
# ──────────────────────────────────────────────────────────────


def _mask_condition(mask_idx: int, row: int, col: int) -> bool:
    """Return True if the module at (row, col) should be inverted under this mask."""
    if mask_idx == 0:
        return (row + col) % 2 == 0
    elif mask_idx == 1:
        return row % 2 == 0
    elif mask_idx == 2:
        return col % 3 == 0
    elif mask_idx == 3:
        return (row + col) % 3 == 0
    elif mask_idx == 4:
        return ((row // 2) + (col // 3)) % 2 == 0
    elif mask_idx == 5:
        return (row * col) % 6 == 0
    elif mask_idx == 6:
        return ((row * col) % 6) < 3
    elif mask_idx == 7:
        return (row + col + ((row * col) % 3)) % 2 == 0
    else:
        raise ValueError(f"Invalid mask index: {mask_idx}")


def apply_mask(matrix, mask_idx: int):
    """Apply data mask in-place to a 2D numpy bool array (True = dark).
    Returns the modified matrix (same object).
    """
    import numpy as np

    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            if _mask_condition(mask_idx, r, c):
                matrix[r, c] ^= True
    return matrix


# ──────────────────────────────────────────────────────────────
# 6. Alphanumeric encoding table (spec Table 5)
# ──────────────────────────────────────────────────────────────

_ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

ALPHANUMERIC_TO_VALUE: dict[str, int] = {
    ch: i for i, ch in enumerate(_ALPHANUMERIC_CHARS)
}
ALPHANUMERIC_VALUE_TO_CHAR: dict[int, str] = {
    i: ch for i, ch in enumerate(_ALPHANUMERIC_CHARS)
}
