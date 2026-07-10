from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from quantizers import get_fixed_quantizer_np

from hls4ml.backends.fpga.fpga_types import FPGAPrecisionConverter
from hls4ml.model.layers import Layer
from hls4ml.model.types import (
    ExponentPrecisionType,
    FixedPrecisionType,
    IntegerPrecisionType,
    PrecisionType,
    RoundingMode,
    SaturationMode,
    XnorPrecisionType,
)

# region Precision types


class XLSDefinitionBase:
    def xls_definition(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.xls_definition()


def xls_str(x) -> str:
    return x.xls_definition() if hasattr(x, 'xls_definition') else str(x)


class XLSIntegerPrecisionDefinition(XLSDefinitionBase):
    def __init__(self, width, signed: bool):
        self.width = width
        self.signed = signed

    def xls_definition(self) -> str:
        prefix = 's' if self.signed else 'u'
        if isinstance(self.width, int) and 1 <= self.width <= 64:
            # u32
            return f'{prefix}{self.width}'
        # uN[NUM_BITS]
        return f'{prefix}N[{self.width}]'


def sN(width):
    return XLSIntegerPrecisionDefinition(width, signed=True)


def uN(width):
    return XLSIntegerPrecisionDefinition(width, signed=False)


s32 = sN(32)
u32 = uN(32)


class XLSFixedPointPrecisionDefinitionBase(XLSDefinitionBase, PrecisionType):
    def xls_definition(self) -> str:
        return f'FixedPoint<{self.xls_num_bits}, {self.xls_binary_exponent}>'

    @property
    def xls_num_bits(self) -> int:
        """Number of bits for DSLX FixedPoint representation for the given precision.
        Note that FixedPoint is always a signed type,
        so we add an extra bit when converting from unsigned."""
        return self.width + 1 if not self.signed else self.width

    @property
    def xls_binary_exponent(self) -> int:
        raise NotImplementedError

    @property
    def xls_rounding_mode(self) -> RoundingMode:
        return getattr(self, 'rounding_mode', RoundingMode.TRN)

    @property
    def xls_saturation_mode(self) -> SaturationMode:
        return getattr(self, 'saturation_mode', SaturationMode.WRAP)

    @property
    def significand_type(self) -> XLSIntegerPrecisionDefinition:
        return XLSIntegerPrecisionDefinition(width=self.xls_num_bits, signed=True)


class XLSFixedPointFixedPrecisionDefinition(XLSFixedPointPrecisionDefinitionBase):
    @property
    def xls_binary_exponent(self) -> int:
        return -self.fractional


class XLSFixedPointIntegerPrecisionDefinition(XLSFixedPointPrecisionDefinitionBase):
    @property
    def xls_binary_exponent(self) -> int:
        return 0


class XLSFixedPointExponentPrecisionDefinition(XLSFixedPointPrecisionDefinitionBase):
    @property
    def xls_binary_exponent(self) -> int:
        return 1 - self.width


class XLSFixedPointXnorPrecisionDefinition(XLSFixedPointPrecisionDefinitionBase):
    @property
    def xls_binary_exponent(self) -> int:
        return 0


class XLSPrecisionConverter(FPGAPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: XLSFixedPointFixedPrecisionDefinition,
                IntegerPrecisionType: XLSFixedPointIntegerPrecisionDefinition,
                ExponentPrecisionType: XLSFixedPointExponentPrecisionDefinition,
                XnorPrecisionType: XLSFixedPointXnorPrecisionDefinition,
            },
            prefix='XLS',
        )


# endregion

# region Helper functions


def float_to_significand(
    x: np.floating[Any] | NDArray[np.floating[Any]], precision: XLSFixedPointPrecisionDefinitionBase
) -> np.integer[Any] | NDArray[np.integer[Any]]:
    """Convert floating point value to fixed point significand.

    Returns: x * 2^precision.fractional
    """
    assert isinstance(precision, XLSFixedPointPrecisionDefinitionBase), (
        f'precision must be XLSFixedPointDefinitionBase, got {type(precision)}'
    )
    assert precision.xls_num_bits <= 64, f'precision.xls_num_bits must be <=64, got {precision.xls_num_bits}'

    if not np.isscalar(x):
        if not isinstance(x, np.ndarray) or x.dtype.kind != 'f':
            x = np.asarray(x, dtype=np.float64)

    if isinstance(precision, XLSFixedPointXnorPrecisionDefinition):
        # hls4ml stores XNOR weights as bits {0,1};
        # We convert it to XLS FixedPoint {-1, 1}
        assert np.all((x == 0) | (x == 1)), 'XNOR weights must be 0 or 1'
        x = np.where(x == 0, -1, 1)
    quantizer = get_fixed_quantizer_np(
        round_mode=str(precision.xls_rounding_mode), overflow_mode=str(precision.xls_saturation_mode)
    )
    scale = 2 ** (-precision.xls_binary_exponent)
    return quantizer(x * scale, k=1, i=precision.xls_num_bits - 1, f=0).astype(np.int64)


def shape_tuple(shape):
    if np.isscalar(shape):
        return (shape,)
    return tuple(shape)


def xls_shape_str(shape):
    return ''.join(f'[{dim}]' for dim in reversed(shape_tuple(shape)))


# endregion


# region Data types


class XLSQualifiedName(XLSDefinitionBase):
    def __init__(self, name: str, module_name: str | None = None):
        self.name = name
        self.module_name = module_name

    def xls_definition(self) -> str:
        return f'{self.module_name}::{self.name}' if self.module_name else self.name


class XLSConstDefinition(XLSDefinitionBase):
    def __init__(self, name, value, type=None):
        self.name = name
        self.value = value
        self.type = type

    def xls_definition(self) -> str:
        type_str = f': {xls_str(self.type)}' if self.type else ''
        return f'pub const {self.name}{type_str} = {xls_str(self.value)};'


class XLSFixedPointDefinition(XLSDefinitionBase):
    def __init__(self, significand, precision: XLSFixedPointPrecisionDefinitionBase):
        assert isinstance(precision, XLSFixedPointPrecisionDefinitionBase)
        self.significand = significand
        self.precision = precision

    def xls_definition(self) -> str:
        return f'{xls_str(self.precision)}{{ significand: {self.significand} }}'

    @classmethod
    def from_float(cls, value, precision: XLSFixedPointPrecisionDefinitionBase):
        return cls(float_to_significand(value, precision), precision)

    @classmethod
    def min_value(cls, precision):
        significand = -(2 ** (precision.xls_num_bits - 1))
        if precision.xls_saturation_mode == SaturationMode.SAT_SYM:
            significand += 1
        return cls(significand, precision)

    @classmethod
    def max_value(cls, precision):
        significand = 2 ** (precision.xls_num_bits - 1) - 1
        return cls(significand, precision)


# endregion


class XLSArrayTypeDefinition(XLSDefinitionBase):
    def __init__(self, element_type, shape):
        self.element_type = element_type
        self.shape = shape_tuple(shape)

    @property
    def rank(self):
        return len(self.shape)

    def inner_type(self):
        """Returns: type of arr[0]"""
        if self.rank == 1:
            return self.element_type
        return XLSArrayTypeDefinition(self.element_type, self.shape[1:])

    def xls_definition(self):
        return f'{xls_str(self.element_type)}{xls_shape_str(self.shape)}'


class XLSTypeConverter:
    def __init__(self, precision_converter):
        self.precision_converter = precision_converter

    def convert(self, atype):
        atype.precision = self.precision_converter.convert(atype.precision)
        return atype


class XLSVarConverter:
    def __init__(self, type_converter):
        self.type_converter = type_converter

    def convert(self, var):
        var.type = self.type_converter.convert(var.type)
        return var


class XLSInterfaceVarConverter(XLSVarConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter)

    def convert(self, var):
        var = super().convert(var)
        type_cls_name = type(var).__name__
        cls_fqn = var.__class__.__module__ + '.' + var.__class__.__qualname__

        var.__class__ = type('XLS' + type_cls_name, (type(var), XLSTensorVariableDefinition), {'_wrapped': cls_fqn})
        return var


class XLSTypeAliasDefinition(XLSDefinitionBase):
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def xls_definition(self):
        return f'pub type {self.name} = {xls_str(self.type)};'


class XLSTensorVariableDefinition(XLSDefinitionBase):
    @property
    def xls_name(self):
        return ''.join(filter(lambda s: s.isalnum() or s == '_', self.name)).lower()

    @property
    def xls_name_upper(self):
        return self.xls_name.upper()

    @property
    def xls_name_camel(self):
        words = self.xls_name.split('_')
        return '_'.join(word[:1].upper() + word[1:].lower() for word in words)

    @property
    def xls_num_bits(self):
        return XLSConstDefinition(f'{self.xls_name_upper}_NUM_BITS', self.type.precision.xls_num_bits, type=u32)

    @property
    def xls_binary_exponent(self):
        return XLSConstDefinition(
            f'{self.xls_name_upper}_BINARY_EXPONENT', self.type.precision.xls_binary_exponent, type=s32
        )

    @property
    def xls_rounding_mode(self):
        return XLSConstDefinition(
            f'{self.xls_name_upper}_ROUNDING_MODE',
            f'RoundingMode::{self.type.precision.xls_rounding_mode}',
            type='RoundingMode',
        )

    @property
    def xls_overflow_mode(self):
        return XLSConstDefinition(
            f'{self.xls_name_upper}_OVERFLOW_MODE',
            f'OverflowMode::{self.type.precision.xls_saturation_mode}',
            type='OverflowMode',
        )

    @property
    def xls_dims(self):
        return tuple(
            XLSConstDefinition(f'{self.xls_name_upper}_DIM_{i}', dim, type=u32)
            for i, dim in enumerate(shape_tuple(self.shape))
        )

    def xls_type_alias(self, rank=None):
        shape = [dim.name for dim in self.xls_dims]
        name = f'{self.xls_name_camel}_Type'
        if rank is not None:
            shape = shape[len(shape) - rank :]
            name += f'_{rank}d'
        return XLSTypeAliasDefinition(name=name, type=XLSArrayTypeDefinition(self.type.precision, shape))

    @property
    def xls_type_alias_bits(self):
        return XLSTypeAliasDefinition(
            name=f'{self.xls_name_camel}_Type_Bits',
            type=XLSArrayTypeDefinition(self.type.precision.significand_type, self.shape),
        )

    def xls_definitions(self) -> list:
        return [
            self.xls_num_bits,
            self.xls_binary_exponent,
            self.xls_rounding_mode,
            self.xls_overflow_mode,
            *self.xls_dims,
            self.xls_type_alias(),
            self.xls_type_alias_bits,
        ] + [self.xls_type_alias(rank=rank) for rank in range(1, len(self.shape) + 1)]

    def xls_definition(self) -> str:
        return '\n'.join(x.xls_definition() for x in self.xls_definitions())


class XLSArrayDefinition(XLSDefinitionBase):
    def __init__(self, element_type, shape, data):
        self.element_type = element_type
        self.shape = shape_tuple(shape)
        self.data = data

    def xls_definition(self) -> str:
        array_type = f'{xls_str(self.element_type)}{xls_shape_str(self.shape)}'
        data = self.data
        if not isinstance(data, str):
            if len(self.shape) > 1:
                data = ', '.join(
                    XLSArrayDefinition(element_type=self.element_type, shape=self.shape[1:], data=x).xls_definition()
                    for x in self.data
                )
            else:
                data = ', '.join(map(str, data))
        return f'{array_type}:[{data}]'


class XLSFunctionCallDefinition(XLSDefinitionBase):
    def __init__(self, name, params=None, args=None):
        self.name = name
        self.params = params or []
        self.args = args or []
        if isinstance(self.params, str):
            self.params = [self.params]
        if isinstance(self.args, str):
            self.args = [self.args]

    @property
    def namespace(self):
        parts = self.name.split('::')
        match len(parts):
            case 1:
                return None
            case 2:
                return parts[0]
            case _:
                raise ValueError(f'Cannot extract namespace from function name: {self.name}')

    def xls_definition(self):
        params = ', '.join(map(str, self.params))
        if params:
            params = f'<{params}>'
        args = ', '.join(map(str, self.args))
        return f'{self.name}{params}({args})'


class XLSFixedPointArrayDefinition(XLSDefinitionBase):
    def __init__(self, precision: XLSFixedPointPrecisionDefinitionBase, shape, data):
        precision = XLSPrecisionConverter().convert(precision)
        shape = shape_tuple(shape)

        self.precision = precision
        self.shape = shape
        self.data = data

    def xls_definition(self) -> str:
        raw_data = float_to_significand(self.data, self.precision)
        raw_element_type = self.precision.significand_type
        raw_array = XLSArrayDefinition(element_type=raw_element_type, shape=self.shape, data=raw_data)
        return XLSFunctionCallDefinition(
            name=f'fixed_point_util::make_fixed_points_{len(self.shape)}d',
            params=[self.precision.xls_binary_exponent],
            args=[raw_array.xls_definition()],
        ).xls_definition()


class XLSWeightVariableDefinition:
    @property
    def xls_name(self):
        return self.name.upper()

    def get_xls_const_definition(self, node: Layer, weights_key: str) -> XLSConstDefinition:
        class_name = node.class_name
        if class_name == 'ApplyAlpha':
            class_name = 'BatchNormalization'

        input_var = node.get_input_variable()
        output_var = node.get_output_variable()

        precision = None
        data = self.data
        expected_shape = None
        if weights_key == 'bias':
            expected_shape = (output_var.shape[-1],)
        else:
            if class_name == 'PReLU':
                assert weights_key == 'param', (
                    f'Unexpected weights key {weights_key} for PReLU node {node.name}, expected "param"'
                )
                precision = node.get_attr('param_t').precision
            elif class_name == 'BatchNormalization':
                assert weights_key == 'scale', (
                    f'Unexpected weights key {weights_key} for BatchNormalization node {node.name}, expected "scale"'
                )
            else:
                assert weights_key == 'weight', (
                    f'Unexpected weights key {weights_key} for node {node.name}, expected "weight"'
                )

            match class_name:
                case 'BatchNormalization':
                    # NB: we need flattening because sometimes the weights can be e.g.
                    # (1,1,1,n_filt) instead of (n_filt,)
                    # We'll throw an error if there are several dimensions larger than 1.
                    data = data.flatten()
                    n_filt = node.get_attr('n_filt')
                    if n_filt == -1:
                        n_filt = input_var.shape[-1]
                    expected_shape = (n_filt,)
                case 'Conv1D':
                    expected_shape = tuple(node.get_attr(x) for x in ['filt_width', 'n_chan', 'n_filt'])
                case 'DepthwiseConv1D':
                    expected_shape = tuple(node.get_attr(x) for x in ['filt_width', 'n_chan', 'depth_multiplier'])
                case 'Conv2D':
                    expected_shape = tuple(node.get_attr(x) for x in ['filt_height', 'filt_width', 'n_chan', 'n_filt'])
                case 'DepthwiseConv2D':
                    expected_shape = tuple(
                        node.get_attr(x) for x in ['filt_height', 'filt_width', 'n_chan', 'depth_multiplier']
                    )
                case 'Dense':
                    # Transpose the weights so that we can call dot_prod(x, w[i]) in nnet_utils/dense.x
                    data = data.T
                    expected_shape = (output_var.shape[0], input_var.shape[0])
                case 'PReLU':
                    expected_shape = (input_var.shape[0],)
                case _:
                    raise ValueError(f'Unsupported weights for layer {node.class_name}')

        if expected_shape is not None:
            assert shape_tuple(data.shape) == expected_shape, (
                f'Weights shape mismatch: expected {expected_shape}, got {data.shape}'
            )

        precision = precision or self.type.precision

        return XLSConstDefinition(
            name=self.xls_name,
            type=precision.xls_definition() + xls_shape_str(data.shape),
            value=XLSFixedPointArrayDefinition(precision=precision, shape=data.shape, data=data),
        )


class XLSWeightVarConverter:
    def __init__(self, type_converter):
        self.type_converter = type_converter

    def convert(self, weight_var, node: Layer, weights_key: str):
        if isinstance(weight_var, XLSWeightVariableDefinition):  # Already converted
            return weight_var
        weight_var.type = self.type_converter.convert(weight_var.type)
        weight_cls_fqn = weight_var.__class__.__module__ + '.' + weight_var.__class__.__qualname__
        weight_var.__class__ = type(
            'XLSWeightVariable',
            (type(weight_var), XLSWeightVariableDefinition),
            {'_wrapped': weight_cls_fqn},
        )
        return weight_var


class XLSLookupTableDefinition(XLSDefinitionBase):
    def __init__(
        self,
        name: str,
        input_precision: XLSFixedPointPrecisionDefinitionBase,
        output_precision: XLSFixedPointPrecisionDefinitionBase,
        x_min,
        log2_step,
        raw_table,
    ) -> None:
        assert isinstance(input_precision, XLSFixedPointPrecisionDefinitionBase)
        assert isinstance(output_precision, XLSFixedPointPrecisionDefinitionBase)

        self.input_num_bits = XLSConstDefinition(f'{name}_INPUT_NUM_BITS', input_precision.xls_num_bits, u32)
        self.input_binary_exponent = XLSConstDefinition(
            f'{name}_INPUT_BINARY_EXPONENT', input_precision.xls_binary_exponent, s32
        )
        self.output_num_bits = XLSConstDefinition(f'{name}_OUTPUT_NUM_BITS', output_precision.xls_num_bits, u32)
        self.output_binary_exponent = XLSConstDefinition(
            f'{name}_OUTPUT_BINARY_EXPONENT', output_precision.xls_binary_exponent, s32
        )
        self.size = XLSConstDefinition(f'{name}_SIZE', len(raw_table), u32)
        self.log2_step = XLSConstDefinition(f'{name}_LOG2_STEP', log2_step, s32)
        self.x_min = XLSConstDefinition(f'{name}_X_MIN', x_min, input_precision)
        int_table = XLSArrayDefinition(element_type=sN(f'{name}_OUTPUT_NUM_BITS'), shape=f'{name}_SIZE', data=raw_table)
        fixed_point_table = XLSFunctionCallDefinition(
            name='fixed_point_util::make_fixed_points_1d', params=[self.output_binary_exponent.name], args=[int_table]
        )
        self.lookup_table = XLSConstDefinition(
            name=name,
            value=XLSFunctionCallDefinition(
                name='lookup_table::create', params=[self.log2_step.name], args=[self.x_min.name, fixed_point_table]
            ),
        )

    def xls_definitions(self) -> list[XLSConstDefinition]:
        return [
            self.input_num_bits,
            self.input_binary_exponent,
            self.output_num_bits,
            self.output_binary_exponent,
            self.size,
            self.log2_step,
            self.x_min,
            self.lookup_table,
        ]

    def xls_definition(self, indent=None):
        indent = indent or ''
        return '\n'.join([f'{indent}{d}' for d in self.xls_definitions()])
