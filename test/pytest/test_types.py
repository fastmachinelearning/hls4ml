import pytest

from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.backends.fpga.fpga_types import ACFixedPrecisionDefinition, APFixedPrecisionDefinition
from hls4ml.model.types import (
    ExponentPrecisionType,
    FixedPrecisionType,
    IntegerPrecisionType,
    RoundingMode,
    SaturationMode,
    XnorPrecisionType,
)


def test_precision_type_creation(capsys):
    int_type = IntegerPrecisionType(width=1, signed=False)
    xnr_type = XnorPrecisionType()

    assert int_type != xnr_type  # Must ensure that similar types are not matched

    int_type = IntegerPrecisionType(width=8, signed=True)
    exp_type = ExponentPrecisionType(width=8, signed=True)

    assert int_type != exp_type  # Must ensure that similar types are not matched

    fp_type = FixedPrecisionType(12, 6)
    fp_type.integer += 2
    fp_type.rounding_mode = None
    fp_type.saturation_mode = 'SAT'

    assert fp_type.integer == 8
    assert fp_type.fractional == 4  # Should be automatically updated
    assert fp_type.rounding_mode == RoundingMode.TRN  # None should be changed to default
    assert fp_type.saturation_mode == SaturationMode.SAT  # Strings should parse correctly

    # Setting saturation mode but not rounding mode should still result in correct type being written out
    fp_type = FixedPrecisionType(12, 6, rounding_mode=None, saturation_mode=SaturationMode.SAT_SYM, saturation_bits=1)
    # Circumvent the type wrapping that happens in the backend
    fp_type.__class__ = type('APFixedPrecisionType', (type(fp_type), APFixedPrecisionDefinition), {})
    fp_cpp = fp_type.definition_cpp()
    assert fp_cpp == 'ap_fixed<12,6,AP_TRN,AP_SAT_SYM,1>'  # Should include the whole type definition, including rounding
    # Reset to default
    fp_type.saturation_mode = 'WRAP'
    fp_type.saturation_bits = 0
    fp_cpp = fp_type.definition_cpp()
    assert fp_cpp == 'ap_fixed<12,6>'  # Should not include defaults

    # Same test for AC types
    fp_type = FixedPrecisionType(12, 6, rounding_mode=None, saturation_mode=SaturationMode.SAT_SYM, saturation_bits=1)
    # Circumvent the type wrapping that happens in the backend
    fp_type.__class__ = type('ACFixedPrecisionType', (type(fp_type), ACFixedPrecisionDefinition), {})
    fp_cpp = fp_type.definition_cpp()
    assert fp_cpp == 'ac_fixed<12,6,true,AC_TRN,AC_SAT_SYM>'  # Should include the whole type definition, including rounding
    # The invalid saturation bit setting should produce a warning
    captured = capsys.readouterr()
    assert 'WARNING: Invalid setting of saturation bits' in captured.out
    # Reset to default
    fp_type.saturation_mode = 'WRAP'
    fp_type.saturation_bits = 0
    fp_cpp = fp_type.definition_cpp()
    assert fp_cpp == 'ac_fixed<12,6,true>'  # Should not include defaults


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
    '''Test that convert_precisions_string determines the signedness correctly'''
    strprec = prec_pair[0]
    signed = prec_pair[1]

    evalprec = FPGABackend.convert_precision_string(strprec)
    assert evalprec.signed == signed
