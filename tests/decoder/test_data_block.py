"""Tests for QR code data block de-interleaving."""

import pytest

from qr_reader.decoder.data_block import DataBlock, deinterleave
from qr_reader.decoder.tables import VERSIONS, total_codewords


def _mkbytes(n: int, offset: int = 0) -> bytes:
    """Return n bytes: offset, offset+1, ... mod 256."""
    return bytes((offset + i) % 256 for i in range(n))


def _interleave(blocks, ec_per_block):
    """Given a list of (data_bytes, ec_bytes) tuples, produce the interleaved raw
    codeword stream as it would appear in a QR code Model 2 symbol.

    This is the inverse of deinterleave: it's what an encoder does.
    """
    data_sizes = [len(d) for d, _ in blocks]
    min_data = min(data_sizes)
    total_blocks = len(blocks)
    longer_indices = [i for i, ds in enumerate(data_sizes) if ds > min_data]

    raw = bytearray()

    # Interleave data: byte_0 of block_0, byte_0 of block_1, ...
    for i in range(min_data):
        for b in range(total_blocks):
            raw.append(blocks[b][0][i])

    # Extra data byte for longer blocks
    for b in longer_indices:
        raw.append(blocks[b][0][min_data])

    # Interleave EC bytes
    for i in range(ec_per_block):
        for b in range(total_blocks):
            raw.append(blocks[b][1][i])

    return bytes(raw)


class TestDataBlockStruct:
    """Tests for DataBlock dataclass."""

    def test_creation(self):
        b = DataBlock(data=b"hello", ec=b"\x01\x02")
        assert b.data == b"hello"
        assert b.ec == b"\x01\x02"

    def test_repr(self):
        b = DataBlock(data=b"abc", ec=b"\x00" * 7)
        assert "3B" in repr(b)
        assert "7B" in repr(b)


class TestDeinterleaveBasic:
    """Basic de-interleaving tests with known patterns."""

    def test_single_block(self):
        """Version 1-L has 1 block: 19 data + 7 EC = 26 bytes. No interleaving."""
        raw = _mkbytes(26)
        blocks = deinterleave(raw, version=1, ecl="L")
        assert len(blocks) == 1
        assert blocks[0].data == _mkbytes(19)
        assert blocks[0].ec == _mkbytes(7, offset=19)

    def test_version_1_all_ecls(self):
        """All v1 ECLs have a single block."""
        mappings = {"L": 19, "M": 16, "Q": 13, "H": 9}
        for ecl, data_len in mappings.items():
            raw = _mkbytes(26)
            blocks = deinterleave(raw, version=1, ecl=ecl)
            assert len(blocks) == 1
            assert len(blocks[0].data) == data_len
            assert len(blocks[0].ec) == 26 - data_len
            assert blocks[0].data + blocks[0].ec == raw

    def test_roundtrip_two_equal_blocks(self):
        """Two equal-sized blocks: interleave then de-interleave, using a version
        that matches the block layout.
        v2-L: ec_per_block=10, groups=[(34, 1)] → 1 block only.

        Let's use v1 as a single block sanity check. For two equal blocks,
        we use v3-M: ec_per_block=26, groups=[(44, 1)] → 1 block only...

        Actually, let's find a version with exactly 2 equal blocks.
        v6-L: ec_per_block=18, groups=[(68, 2)] → 2 blocks of 68 data, 18 EC each!
        """
        version, ecl = 6, "L"
        ec_per_block, groups = VERSIONS[version].ec_info[ecl]
        assert groups == [(68, 2)]

        # Create two blocks of 68 data + 18 EC each
        blocks_data = [
            (_mkbytes(68, offset=0), _mkbytes(18, offset=200)),
            (_mkbytes(68, offset=100), _mkbytes(18, offset=220)),
        ]

        raw = _interleave(blocks_data, ec_per_block)
        assert len(raw) == total_codewords(version, ecl)

        recovered = deinterleave(raw, version=version, ecl=ecl)
        assert len(recovered) == 2
        assert recovered[0].data == blocks_data[0][0]
        assert recovered[0].ec == blocks_data[0][1]
        assert recovered[1].data == blocks_data[1][0]
        assert recovered[1].ec == blocks_data[1][1]

    def test_roundtrip_unequal_blocks(self):
        """Two blocks where one is longer by 1 byte.
        v5-H: ec_per_block=22, groups=[(11, 2), (12, 2)] → 2 short + 2 long.
        We need exactly 2 blocks with unequal sizes.
        v2-H: ec_per_block=28, groups=[(16, 1)] → 1 block only.

        Let's use v3-H which has ec_per_block=22, groups=[(13, 2)] → 2 equal.

        Actually, the QR spec doesn't have a "2 blocks, one short, one long" case.
        The group structure always pairs equal blocks. But the interleaving
        algorithm handles it. Let's test with a version that has group1=(n,1), group2=(n+1,1).
        v11-M: ec_per_block=30, groups=[(50, 1), (51, 4)] → that's 1 + 4 = 5 total.

        v3-H: ec_per_block=22, groups=[(13, 2)] → 2 equal blocks of 13.

        For the unequal test, let's pick v8-M: ec_per_block=22, groups=[(38, 2), (39, 2)].
        That's 4 blocks total (2 short + 2 long). Close enough for testing.
        """
        version, ecl = 8, "M"
        ec_per_block, groups = VERSIONS[version].ec_info[ecl]
        assert groups == [(38, 2), (39, 2)]

        blocks_data = [
            (_mkbytes(38, offset=0), _mkbytes(22, offset=200)),
            (_mkbytes(38, offset=50), _mkbytes(22, offset=230)),
            (_mkbytes(39, offset=100), _mkbytes(22, offset=10)),
            (_mkbytes(39, offset=150), _mkbytes(22, offset=40)),
        ]

        raw = _interleave(blocks_data, ec_per_block)
        assert len(raw) == total_codewords(version, ecl)

        recovered = deinterleave(raw, version=version, ecl=ecl)
        assert len(recovered) == 4
        for i in range(4):
            assert recovered[i].data == blocks_data[i][0], f"block {i} data"
            assert recovered[i].ec == blocks_data[i][1], f"block {i} EC"

    def test_many_blocks(self):
        """Test with 4 equal-sized blocks. v6-Q: ec_per_block=24, groups=[(19, 4)]."""
        version, ecl = 6, "Q"
        ec_per_block, groups = VERSIONS[version].ec_info[ecl]
        assert groups == [(19, 4)]  # 4 blocks of 19 data

        blocks_data = [
            (_mkbytes(19, offset=i * 30), _mkbytes(24, offset=200 + i * 30))
            for i in range(4)
        ]

        raw = _interleave(blocks_data, ec_per_block)
        assert len(raw) == total_codewords(version, ecl)

        recovered = deinterleave(raw, version=version, ecl=ecl)
        assert len(recovered) == 4
        for i in range(4):
            assert recovered[i].data == blocks_data[i][0], f"block {i} data mismatch"
            assert recovered[i].ec == blocks_data[i][1], f"block {i} EC mismatch"

    def test_v5_h_two_groups(self):
        """v5-H has 2 groups: (11, 2) and (12, 2) with ec_per_block=22."""
        ec_per_block = VERSIONS[5].ec_info["H"][0]  # 22
        groups = VERSIONS[5].ec_info["H"][1]  # [(11, 2), (12, 2)]

        blocks_data = [
            (_mkbytes(11, offset=0), _mkbytes(22, offset=50)),
            (_mkbytes(11, offset=100), _mkbytes(22, offset=150)),
            (_mkbytes(12, offset=200), _mkbytes(22, offset=0)),
            (_mkbytes(12, offset=220), _mkbytes(22, offset=30)),
        ]

        raw = _interleave(blocks_data, ec_per_block)

        total = total_codewords(5, "H")
        assert len(raw) == total

        blocks = deinterleave(raw, version=5, ecl="H")
        assert len(blocks) == 4

        data_lengths = [len(b.data) for b in blocks]
        assert sorted(data_lengths) == [11, 11, 12, 12]

        for b in blocks:
            assert len(b.ec) == 22

        assert blocks[0].data == blocks_data[0][0]
        assert blocks[0].ec == blocks_data[0][1]
        assert blocks[1].data == blocks_data[1][0]
        assert blocks[1].ec == blocks_data[1][1]
        assert blocks[2].data == blocks_data[2][0]
        assert blocks[2].ec == blocks_data[2][1]
        assert blocks[3].data == blocks_data[3][0]
        assert blocks[3].ec == blocks_data[3][1]


class TestDeinterleaveTableMatch:
    """Verify that de-interleaving output matches the spec table dimensions."""

    def test_block_counts_all_versions(self):
        """For every version + ECL, verify the number of blocks matches."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                ec_per_block, groups = VERSIONS[v].ec_info[ecl]
                total_blocks = sum(num for _, num in groups)
                total = total_codewords(v, ecl)

                raw = _mkbytes(total)
                blocks = deinterleave(raw, version=v, ecl=ecl)

                assert len(blocks) == total_blocks, (
                    f"v{v} {ecl}: {len(blocks)} != {total_blocks}"
                )

    def test_data_lengths_all_versions(self):
        """For every version + ECL, verify data lengths match the spec table."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                ec_per_block, groups = VERSIONS[v].ec_info[ecl]
                total = total_codewords(v, ecl)

                raw = _mkbytes(total)
                blocks = deinterleave(raw, version=v, ecl=ecl)

                expected_data_sizes = []
                for data_bytes, num_blocks in groups:
                    expected_data_sizes.extend([data_bytes] * num_blocks)

                actual_data_sizes = [len(b.data) for b in blocks]
                assert sorted(actual_data_sizes) == sorted(expected_data_sizes), (
                    f"v{v} {ecl}: {actual_data_sizes} != {expected_data_sizes}"
                )

    def test_ec_lengths_all_versions(self):
        """For every version + ECL, verify EC codeword counts are correct."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                ec_per_block, groups = VERSIONS[v].ec_info[ecl]
                total = total_codewords(v, ecl)

                raw = _mkbytes(total)
                blocks = deinterleave(raw, version=v, ecl=ecl)

                for b in blocks:
                    assert len(b.ec) == ec_per_block, (
                        f"v{v} {ecl}: ec_len={len(b.ec)} != {ec_per_block}"
                    )

    def test_total_data_preserved(self):
        """The total number of data + EC bytes de-interleaved equals input."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                total = total_codewords(v, ecl)
                raw = _mkbytes(total)
                blocks = deinterleave(raw, version=v, ecl=ecl)

                recovered_total = sum(len(b.data) + len(b.ec) for b in blocks)
                assert recovered_total == total, (
                    f"v{v} {ecl}: {recovered_total} != {total}"
                )


class TestDeinterleaveMultipleGroups:
    """Specific tests for versions with multiple block groups (unequal block sizes)."""

    def test_v5_h_two_groups(self):
        """v5-H: ec_per_block=22, groups=[(11, 2), (12, 2)] → total_blocks=4."""
        ec_per_block, groups = VERSIONS[5].ec_info["H"]
        assert ec_per_block == 22
        assert groups == [(11, 2), (12, 2)]

        total = total_codewords(5, "H")
        raw = _mkbytes(total)
        blocks = deinterleave(raw, version=5, ecl="H")

        assert len(blocks) == 4
        assert len(blocks[0].data) == 11
        assert len(blocks[1].data) == 11
        assert len(blocks[2].data) == 12
        assert len(blocks[3].data) == 12

    def test_v8_m_two_groups(self):
        """v8-M: ec_per_block=22, groups=[(38, 2), (39, 2)]."""
        ec_per_block, groups = VERSIONS[8].ec_info["M"]
        assert ec_per_block == 22
        assert groups == [(38, 2), (39, 2)]

        total = total_codewords(8, "M")
        raw = _mkbytes(total)
        blocks = deinterleave(raw, version=8, ecl="M")

        assert len(blocks) == 4
        assert len(blocks[0].data) == 38
        assert len(blocks[1].data) == 38
        assert len(blocks[2].data) == 39
        assert len(blocks[3].data) == 39

    def test_v10_l_two_groups(self):
        """v10-L: ec_per_block=18, groups=[(68, 2), (69, 2)]."""
        ec_per_block, groups = VERSIONS[10].ec_info["L"]
        assert ec_per_block == 18
        assert groups == [(68, 2), (69, 2)]

        total = total_codewords(10, "L")
        raw = _mkbytes(total)
        blocks = deinterleave(raw, version=10, ecl="L")

        assert len(blocks) == 4

    def test_v40_h_two_groups(self):
        """v40-H: ec_per_block=30, groups=[(15, 20), (16, 61)]."""
        ec_per_block, groups = VERSIONS[40].ec_info["H"]
        assert ec_per_block == 30
        assert groups == [(15, 20), (16, 61)]

        total = total_codewords(40, "H")
        raw = _mkbytes(total)
        blocks = deinterleave(raw, version=40, ecl="H")

        assert len(blocks) == 81  # 20 + 61

        for i in range(20):
            assert len(blocks[i].data) == 15

        for i in range(20, 81):
            assert len(blocks[i].data) == 16


class TestDeinterleaveErrors:
    """Test error conditions."""

    def test_invalid_version(self):
        with pytest.raises(ValueError, match="Invalid version"):
            deinterleave(b"", version=0, ecl="L")
        with pytest.raises(ValueError, match="Invalid version"):
            deinterleave(b"", version=41, ecl="L")

    def test_wrong_raw_length(self):
        with pytest.raises(ValueError, match="Raw codeword length"):
            deinterleave(b"\x00" * 10, version=1, ecl="L")  # expect 26

    def test_invalid_ecl(self):
        with pytest.raises(KeyError):
            deinterleave(_mkbytes(26), version=1, ecl="X")


class TestDeinterleaveRoundTrip:
    """Verify interleave + deinterleave round-trips for all version/ECL combos."""

    def _make_test_blocks(self, version, ecl):
        """Create synthetic identifiable blocks matching the version+ECL layout."""
        ec_per_block, groups = VERSIONS[version].ec_info[ecl]
        blocks = []
        byte_counter = 0
        for data_bytes, num_blocks in groups:
            for _ in range(num_blocks):
                data = _mkbytes(data_bytes, offset=byte_counter)
                byte_counter += data_bytes
                ec = _mkbytes(ec_per_block, offset=byte_counter)
                byte_counter += ec_per_block
                blocks.append((data, ec))
        return blocks, ec_per_block

    def test_roundtrip_all_recoverable(self):
        """For all version/ECL combos, interleaving then deinterleaving recovers
        the original data in the correct order."""
        for v in range(1, 41):
            for ecl in ["L", "M", "Q", "H"]:
                blocks, ec_per_block = self._make_test_blocks(v, ecl)
                raw = _interleave(blocks, ec_per_block)

                recovered = deinterleave(raw, version=v, ecl=ecl)

                assert len(recovered) == len(blocks), f"v{v} {ecl}: count mismatch"
                for i, (rec, (orig_data, orig_ec)) in enumerate(zip(recovered, blocks)):
                    assert rec.data == orig_data, f"v{v} {ecl} block {i}: data mismatch"
                    assert rec.ec == orig_ec, f"v{v} {ecl} block {i}: EC mismatch"
