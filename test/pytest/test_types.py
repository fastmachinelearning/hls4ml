import pytest

from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.backends.fpga.fpga_types import ACFixedPrecisionDefinition, APFixedPrecisionDefinition
from hls4ml.model.types import (
    ExponentPrecisionType,
    FixedPrecisionType,
    FloatPrecisionType,
    IntegerPrecisionType,
    RoundingMode,
    SaturationMode,
    StandardFloatPrecisionType,
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
    strprec, signed = prec_pair

    evalprec = FPGABackend.convert_precision_string(strprec)
    assert evalprec.signed == signed


@pytest.mark.parametrize(
    'prec_tuple',
    [
        # Notation without the prefix
        ('fixed<16,6>', 16, 6, True, RoundingMode.TRN, SaturationMode.WRAP),
        ('ufixed<18,7>', 18, 7, False, RoundingMode.TRN, SaturationMode.WRAP),
        ('fixed<14, 5, RND>', 14, 5, True, RoundingMode.RND, SaturationMode.WRAP),
        ('ufixed<13, 8, RND, SAT>', 13, 8, False, RoundingMode.RND, SaturationMode.SAT),
        # Prefixed notation
        ('ap_ufixed<17,6>', 17, 6, False, RoundingMode.TRN, SaturationMode.WRAP),
        ('ac_fixed<15, 4, false, RND, SAT>', 15, 4, False, RoundingMode.RND, SaturationMode.SAT),
    ],
)
def test_fixed_type_parsing(prec_tuple):
    '''Test that convert_precision_string correctly parses specified fixed-point types'''
    prec_str, width, integer, signed, round_mode, saturation_mode = prec_tuple

    evalprec = FPGABackend.convert_precision_string(prec_str)

    assert isinstance(evalprec, FixedPrecisionType)
    assert evalprec.width == width
    assert evalprec.integer == integer
    assert evalprec.signed == signed
    assert evalprec.rounding_mode == round_mode
    assert evalprec.saturation_mode == saturation_mode


@pytest.mark.parametrize(
    'prec_tuple',
    [
        # Notation without the prefix
        ('int<16>', 16, True),
        ('uint<18>', 18, False),
        # Prefixed notation
        ('ap_uint<8>', 8, False),
        ('ac_int<14, false>', 14, False),
    ],
)
def test_int_type_parsing(prec_tuple):
    '''Test that convert_precision_string correctly parses specified fixed-point types'''
    prec_str, width, signed = prec_tuple

    evalprec = FPGABackend.convert_precision_string(prec_str)

    assert isinstance(evalprec, IntegerPrecisionType)
    assert evalprec.width == width
    assert evalprec.signed == signed


@pytest.mark.parametrize(
    'prec_pair',
    [
        # Standard floating-point types, should be parsed as C++ types
        ('float', True),
        ('double', True),
        ('half', True),
        ('bfloat16', True),
        # Standard bitwidths, but should result in ap_float or ac_std_float, not standard C++ types
        ('std_float<32,8>', False),
        ('std_float<64,11>', False),
        ('std_float<16,5>', False),
        ('std_float<16,8>', False),
        # Non-standard bitwidths, should not be parsed as C++ types
        ('std_float<16,6>', False),
        ('std_float<64,10>', False),
    ],
)
def test_float_cpp_parsing(prec_pair):
    '''Test that convert_precision_string correctly parses C++ types'''
    prec_str, is_cpp = prec_pair

    evalprec = FPGABackend.convert_precision_string(prec_str)
    assert isinstance(evalprec, StandardFloatPrecisionType)
    assert evalprec.use_cpp_type == is_cpp and prec_str in str(evalprec)


@pytest.mark.parametrize(
    'prec_tuple',
    [
        # Should result in ac_float
        ('ac_float<25,2, 8>', 33, 2, 8, RoundingMode.TRN),
        ('ac_float<54,2,11, AC_RND>', 65, 2, 11, RoundingMode.RND),
        ('ac_float<25,4, 8>', 33, 4, 8, RoundingMode.TRN),
    ],
)
def test_ac_float_parsing(prec_tuple):
    '''Test that convert_precision_string correctly parses ac_float types'''
    prec_str, width, integer, exponent, round_mode = prec_tuple

    evalprec = FPGABackend.convert_precision_string(prec_str)

    assert isinstance(evalprec, FloatPrecisionType)
    assert evalprec.width == width
    assert evalprec.integer == integer
    assert evalprec.exponent == exponent
    assert evalprec.rounding_mode == round_mode
