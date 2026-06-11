"""Tests for QR Code spec tables."""

import numpy as np
import pytest

from qr_reader.decoder.tables import (
    ALPHANUMERIC_TO_VALUE,
    ALPHANUMERIC_VALUE_TO_CHAR,
    CHAR_COUNT_BITS,
    ECL_H,
    ECL_L,
    ECL_M,
    ECL_NAMES,
    ECL_Q,
    FORMAT_PATTERN_TO_INFO,
    MODE_BY_INDICATOR,
    MODE_INDICATORS,
    VALID_FORMAT_PATTERNS,
    VALID_VERSION_PATTERNS,
    VERSION_INFO_PATTERNS,
    VERSION_PATTERN_TO_VERSION,
    VERSIONS,
    _mask_condition,
    apply_mask,
    char_count_version_range,
    total_codewords,
)


class TestVersionInfo:
    """Test version info tables."""

    def test_all_versions_exist(self):
        for v in range(1, 41):
            assert v in VERSIONS, f"Version {v} missing"

    def test_symbol_sizes(self):
        for v in range(1, 41):
            expected = 17 + 4 * v
            assert VERSIONS[v].symbol_size == expected

    def test_total_codewords_is_positive(self):
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                tc = total_codewords(v, ecl)
                assert tc > 0, f"v{v} {ecl}: total={tc}"

    def test_total_codewords_known_values(self):
        # Spec examples
        assert total_codewords(1, "L") == 26  # 19 data + 7 ec
        assert total_codewords(1, "M") == 26  # 16 + 10
        assert total_codewords(1, "Q") == 26  # 13 + 13
        assert total_codewords(1, "H") == 26  #  9 + 17
        assert total_codewords(40, "L") == 3706
        assert total_codewords(40, "H") == 3706

        # Version 3-L: 1 block, 55 data + 15 ec = 70
        assert total_codewords(3, "L") == 70

    def test_ec_info_format(self):
        """Each version+ECL has (ec_per_block, groups)."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                ec_per_block, groups = VERSIONS[v].ec_info[ecl]
                assert isinstance(ec_per_block, int) and ec_per_block > 0
                assert isinstance(groups, list)
                for data_bytes, num_blocks in groups:
                    assert isinstance(data_bytes, int) and data_bytes > 0
                    assert isinstance(num_blocks, int) and num_blocks > 0


class TestAlignmentPatterns:
    """Test alignment pattern positions."""

    def test_v1_has_no_alignments(self):
        assert VERSIONS[1].alignment_positions == []

    def test_v2_alignments(self):
        assert VERSIONS[2].alignment_positions == [6, 18]

    def test_v40_alignments(self):
        assert VERSIONS[40].alignment_positions == [6, 30, 58, 86, 114, 142, 170]

    def test_all_first_element_is_6(self):
        for v in range(2, 41):
            assert VERSIONS[v].alignment_positions[0] == 6

    def test_all_last_element_is_symbol_size_minus_7(self):
        for v in range(2, 41):
            sz = VERSIONS[v].symbol_size
            assert VERSIONS[v].alignment_positions[-1] == sz - 7


class TestFormatInfoTable:
    """Test the format information table."""

    def test_32_patterns(self):
        assert len(VALID_FORMAT_PATTERNS) == 32

    def test_all_15_bit(self):
        for p in VALID_FORMAT_PATTERNS:
            assert 0 <= p < 0x8000  # 15 bits

    def test_mapping_bijection(self):
        assert len(FORMAT_PATTERN_TO_INFO) == 32

    def test_known_pattern(self):
        # L, mask 0 → 0x77C4
        assert FORMAT_PATTERN_TO_INFO[0x77C4] == (ECL_L, 0)
        # H, mask 7 → 0x083B
        assert FORMAT_PATTERN_TO_INFO[0x083B] == (ECL_H, 7)


class TestVersionInfoTable:
    """Test the version information table (v7–v40)."""

    def test_34_patterns(self):
        assert len(VERSION_INFO_PATTERNS) == 34

    def test_all_18_bit(self):
        for p in VERSION_INFO_PATTERNS:
            assert 0 <= p < 0x40000  # 18 bits

    def test_mapping(self):
        assert VERSION_PATTERN_TO_VERSION[0x07C94] == 7
        assert VERSION_PATTERN_TO_VERSION[0x28C69] == 40

    def test_all_versions_7_to_40(self):
        for v in range(7, 41):
            assert v in VERSION_PATTERN_TO_VERSION.values()


class TestCharCountBits:
    """Test character count indicator bit lengths."""

    def test_version_ranges(self):
        assert char_count_version_range(1) == 1
        assert char_count_version_range(9) == 1
        assert char_count_version_range(10) == 2
        assert char_count_version_range(26) == 2
        assert char_count_version_range(27) == 3
        assert char_count_version_range(40) == 3

    def test_numeric_bits(self):
        assert CHAR_COUNT_BITS["numeric"][1] == 10
        assert CHAR_COUNT_BITS["numeric"][2] == 12
        assert CHAR_COUNT_BITS["numeric"][3] == 14

    def test_alphanumeric_bits(self):
        assert CHAR_COUNT_BITS["alphanumeric"][1] == 9
        assert CHAR_COUNT_BITS["alphanumeric"][2] == 11
        assert CHAR_COUNT_BITS["alphanumeric"][3] == 13

    def test_byte_bits(self):
        assert CHAR_COUNT_BITS["byte"][1] == 8
        assert CHAR_COUNT_BITS["byte"][2] == 16
        assert CHAR_COUNT_BITS["byte"][3] == 16


class TestModeIndicators:
    """Test mode indicator constants."""

    def test_all_modes_known(self):
        assert MODE_INDICATORS["numeric"] == 1
        assert MODE_INDICATORS["alphanumeric"] == 2
        assert MODE_INDICATORS["byte"] == 4
        assert MODE_INDICATORS["terminator"] == 0

    def test_reverse_roundtrip(self):
        for mode, bits in MODE_INDICATORS.items():
            assert MODE_BY_INDICATOR[bits] == mode


class TestMaskFunctions:
    """Test the 8 data mask functions."""

    def test_mask_0(self):
        # (row + col) % 2 == 0
        assert _mask_condition(0, 0, 0) is True  # 0 even
        assert _mask_condition(0, 0, 1) is False  # 1 odd
        assert _mask_condition(0, 1, 1) is True  # 2 even

    def test_mask_1(self):
        # row % 2 == 0
        assert _mask_condition(1, 0, 0) is True
        assert _mask_condition(1, 0, 1) is True
        assert _mask_condition(1, 1, 0) is False
        assert _mask_condition(1, 1, 1) is False

    def test_mask_2(self):
        # col % 3 == 0
        assert _mask_condition(2, 0, 0) is True
        assert _mask_condition(2, 0, 1) is False
        assert _mask_condition(2, 0, 3) is True

    def test_mask_3(self):
        # (row + col) % 3 == 0
        assert _mask_condition(3, 0, 0) is True
        assert _mask_condition(3, 0, 1) is False
        assert _mask_condition(3, 1, 2) is True

    def test_mask_4(self):
        # ((row // 2) + (col // 3)) % 2 == 0
        assert _mask_condition(4, 0, 0) is True
        assert _mask_condition(4, 0, 3) is False

    def test_mask_5(self):
        # (row * col) % 6 == 0
        assert _mask_condition(5, 0, 0) is True
        assert _mask_condition(5, 2, 3) is True  # 6 % 6 == 0
        assert _mask_condition(5, 1, 1) is False

    def test_mask_6(self):
        # ((row * col) % 6) < 3
        assert _mask_condition(6, 0, 0) is True  # 0 < 3
        assert _mask_condition(6, 1, 1) is True  # 1 < 3
        assert _mask_condition(6, 2, 2) is False  # 4 >= 3

    def test_mask_7(self):
        # (row + col + ((row * col) % 3)) % 2 == 0
        assert _mask_condition(7, 0, 0) is True
        assert _mask_condition(7, 1, 0) is False

    def test_invalid_mask_raises(self):
        with pytest.raises(ValueError):
            _mask_condition(8, 0, 0)
        with pytest.raises(ValueError):
            _mask_condition(-1, 0, 0)

    def test_apply_mask(self):
        """apply_mask inverts the correct positions."""
        import numpy as np

        m = np.array(
            [[False, False, False], [False, False, False], [False, False, False]],
            dtype=bool,
        )

        # Mask 0: (row+col)%2==0 → (0,0), (0,2), (1,1), (2,0), (2,2) toggled
        apply_mask(m, 0)
        expected = np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, False, True],
            ]
        )
        assert np.array_equal(m, expected)


class TestAlphanumericTable:
    """Test alphanumeric encoding table."""

    def test_digit_0_is_0(self):
        assert ALPHANUMERIC_TO_VALUE["0"] == 0
        assert ALPHANUMERIC_VALUE_TO_CHAR[0] == "0"

    def test_digit_9_is_9(self):
        assert ALPHANUMERIC_TO_VALUE["9"] == 9

    def test_A_is_10(self):
        assert ALPHANUMERIC_TO_VALUE["A"] == 10
        assert ALPHANUMERIC_VALUE_TO_CHAR[10] == "A"

    def test_Z_is_35(self):
        assert ALPHANUMERIC_TO_VALUE["Z"] == 35

    def test_space_is_36(self):
        assert ALPHANUMERIC_TO_VALUE[" "] == 36

    def test_colon_is_44(self):
        assert ALPHANUMERIC_TO_VALUE[":"] == 44

    def test_roundtrip(self):
        for ch in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:":
            assert ALPHANUMERIC_VALUE_TO_CHAR[ALPHANUMERIC_TO_VALUE[ch]] == ch

    def test_45_chars(self):
        assert len(ALPHANUMERIC_TO_VALUE) == 45


class TestECLConstants:
    """Test ECL naming and format bits."""

    def test_ecl_names(self):
        assert ECL_NAMES[ECL_L] == "L"
        assert ECL_NAMES[ECL_M] == "M"
        assert ECL_NAMES[ECL_Q] == "Q"
        assert ECL_NAMES[ECL_H] == "H"
