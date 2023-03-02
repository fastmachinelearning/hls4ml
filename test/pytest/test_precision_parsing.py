import pytest

import hls4ml


@pytest.mark.parametrize(
    'prec_pair',
    [
        ('ap_fixed<3, 2>', True),
        ('ap_ufixed<3, 2>', False),
        ('ac_fixed<3, 2, true>', True),
        ('ac_fixed<3, 2, false>', False),
        ('ac_fixed<3, 2, 1>', True),
        ('ac_fixed<3, 2, 0>', False),
        ('ap_int<3, 2>', True),
        ('ap_uint<3>', False),
        ('ac_int<3, TRue>', True),
        ('ac_int<3, FALse>', False),
        ('ac_int<3, 1>', True),
        ('ac_int<3, 0>', False),
    ],
)
def test_sign_parsing(prec_pair):
    '''Test that convert_precions_string determines the signedness correctly'''
    strprec = prec_pair[0]
    signed = prec_pair[1]

    evalprec = hls4ml.backends.fpga.fpga_backend.FPGABackend.convert_precision_string(strprec)
    assert evalprec.signed == signed
