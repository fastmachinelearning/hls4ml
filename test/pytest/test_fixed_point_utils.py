import pytest

from hls4ml.utils.fixed_point_utils import FixedPointEmulator, uint_to_binary


@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize('N,I', [(8, 3), (8, 0), (8, 8), (8, -2), (4, 6), (1, 1), (1, 3), (6, -1), (5, 7)])
def test_to_float_matches_twos_complement(N, I, signed):
    """The N stored bits, read as an integer (two's complement if signed), scaled by 2^(I - N)."""
    for i in range(2**N):
        f = FixedPointEmulator(N, I, signed=signed)
        f.set_msb_bits(uint_to_binary(i, N))
        expected = (i - 2**N if signed and i >= 2 ** (N - 1) else i) * 2.0 ** (I - N)
        assert f.to_float() == expected, f'bits={i:0{N}b}'
