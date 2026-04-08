from __future__ import annotations

import builtins
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hls4ml.model.types import FixedPrecisionType, TensorVariable, PrecisionType, IntegerPrecisionType


def float_to_significand(x: np.floating[Any] | NDArray[np.floating[Any]],
                         precision: FixedPrecisionType) -> int:
    """Convert floating point value to fixed point significand.

    Returns: x * 2^precision.fractional
    """
    assert precision.signed, 'Only signed types are supported'

    width = precision.width
    frac = precision.fractional
    scale = 2 ** frac
    if not np.isscalar(x):
        if not isinstance(x, np.ndarray) or x.dtype.kind != 'f':
            x = np.asarray(x, dtype=np.float64)
    # TODO support different saturation and rounding modes
    significand = np.round(x * scale).astype(np.int64)
    n = 2 ** width
    shift = 2 ** (width - 1)
    return (significand + shift) % n - shift


def to_signed_fixed_precision(precision: PrecisionType) -> FixedPrecisionType:
    """Convert precision to a signed FixedPrecisionType used by XLS."""
    assert isinstance(precision, IntegerPrecisionType) or \
           isinstance(precision, FixedPrecisionType), \
        f'Unknown precision type: {type(precision)}'
    fixed_precision = FixedPrecisionType(
        width=precision.width,
        integer=precision.integer,
        signed=precision.signed,
        rounding_mode=precision.rounding_mode,
        saturation_mode=precision.saturation_mode,
    )
    # Only signed types are supported in XLS
    if not fixed_precision.signed:
        fixed_precision.signed = True
        fixed_precision.width += 1
        fixed_precision.integer += 1

    return fixed_precision


# XLS types

class XLSIntegerType:
    def __init__(self, width, signed: bool):
        self.width = width
        self.signed = signed

    def __str__(self):
        prefix = 's' if self.signed else 'u'
        if isinstance(self.width, int) and 1 <= self.width <= 64:
            # u32
            return f'{prefix}{self.width}'
        # uN[NUM_BITS]
        return f'{prefix}N[{self.width}]'

    @staticmethod
    def u32():
        return XLSIntegerType(width=32, signed=False)

    @staticmethod
    def s32():
        return XLSIntegerType(width=32, signed=True)


class XLSFixedPointType:
    def __init__(self, num_bits, binary_exponent):
        self.num_bits = num_bits
        self.binary_exponent = binary_exponent

    @classmethod
    def from_precision(cls, precision: FixedPrecisionType):
        assert precision.signed == True, "XLS FixedPoint is always a signed type"
        num_bits = precision.width
        binary_exponent = -precision.fractional
        return cls(num_bits=num_bits, binary_exponent=binary_exponent)

    @property
    def precision(self):
        return FixedPrecisionType(
            width=self.num_bits,
            integer=self.num_bits + self.binary_exponent,
            signed=True)

    def __str__(self):
        return f'FixedPoint<{self.num_bits}, {self.binary_exponent}>'


def as_xls_fixed_point_type(type: XLSFixedPointType | FixedPrecisionType) -> XLSFixedPointType:
    if isinstance(type, XLSFixedPointType):
        return type
    return XLSFixedPointType.from_precision(type)


# 1d array type. TODO make it explicitly multidimensional?
class XLSArrayType:
    def __init__(self, element_type, shape: int | str | tuple[int | str, ...] | list[int | str]):
        if isinstance(element_type, FixedPrecisionType):
            element_type = XLSFixedPointType.from_precision(element_type)

        if isinstance(shape, str) or isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        assert len(shape) > 0, 'Zero-dimensional arrays are not supported'
        if len(shape) == 1:
            self.element_type = element_type
        else:
            self.element_type = XLSArrayType(element_type, shape[1:])
        self.size = shape[0]

    def as_multidimensional(self) -> tuple[Any, tuple[int | str, ...]]:
        """ Returns: (inner element type, shape)

        >>> element_type = XLSFixedPointType(num_bits=16, binary_exponent=-10)
        >>> array_2d = XLSArrayType(element_type=element_type, shape=(2,3))
        >>> elt, shape = array_2d.as_multidimensional()
        >>> str(elt)
        'FixedPoint<16, -10>'
        >>> shape
        (2, 3)

        """
        if isinstance(self.element_type, XLSArrayType):
            elt, shape = self.element_type.as_multidimensional()
            shape = (self.size,) + shape
        else:
            elt = self.element_type
            shape = (self.size,)
        return elt, shape

    @property
    def rank(self):
        _, shape = self.as_multidimensional()
        return len(shape)

    @property
    def innermost_element_type(self):
        """Returns: inner element type, for example:

        >>> element_type = XLSFixedPointType(num_bits=16, binary_exponent=-10)
        >>> array_2d = XLSArrayType(element_type=element_type, shape=(2,3))
        >>> str(array_2d.innermost_element_type)
        'FixedPoint<16, -10>'
        >>> str(array_2d.element_type)
        'FixedPoint<16, -10>[3]'
        """
        elt, shape = self.as_multidimensional()
        return elt

    def __str__(self):
        return f'{self.element_type}[{self.size}]'


# XLS values

class XLSInteger:
    def __init__(self, type: XLSIntegerType | str, value: int | str):
        self.type = type
        self.value = value

    @classmethod
    def u32(cls, value: int | str):
        if isinstance(value, int):
            assert value >= 0, f'value={value} is not an unsigned integer'
        return cls(XLSIntegerType.u32(), value)

    @classmethod
    def s32(cls, value: int | str):
        return cls(XLSIntegerType.s32(), value)

    def __str__(self):
        return f'{self.type}:{self.value}'


class XLSFixedPoint:
    def __init__(self, type: XLSFixedPointType | FixedPrecisionType,
                 significand: XLSInteger | int | np.integer[Any] | str):
        if isinstance(type, FixedPrecisionType):
            type = XLSFixedPointType.from_precision(type)

        if np.issubdtype(builtins.type(significand), np.integer):
            significand = XLSInteger(type=XLSIntegerType(width=type.num_bits, signed=True), value=significand)
        elif isinstance(significand, XLSInteger):
            assert significand.type.width == type.num_bits
            assert significand.type.signed == True, 'FixedPoint is always a signed type'

        self.type = type
        self.significand = significand

    @classmethod
    def from_float(cls, x: np.floating[Any], precision: FixedPrecisionType):
        fp_type = XLSFixedPointType.from_precision(precision)
        return cls(type=fp_type, significand=float_to_significand(x, precision))

    @classmethod
    def min_value(cls, type: XLSFixedPointType | FixedPrecisionType):
        type = as_xls_fixed_point_type(type)
        return cls(type=type, significand=-2 ** (type.num_bits - 1))

    @classmethod
    def max_value(cls, type: XLSFixedPointType | FixedPrecisionType):
        type = as_xls_fixed_point_type(type)
        return cls(type=type, significand=2 ** (type.num_bits - 1) - 1)

    @classmethod
    def zero(cls, type: XLSFixedPointType | FixedPrecisionType):
        type = as_xls_fixed_point_type(type)
        return cls(type=type, significand=0)

    def __str__(self):
        # return f'fp_util::make_fixed_point<{self.type.binary_exponent}>:<{self.significand}>'
        return f'{self.type}{{ significand: {self.significand} }}'


# 1d array. TODO make it explicitly multidimensional?
class XLSArray:
    def __init__(self, array_type: XLSArrayType, array):
        self.array_type = array_type

        if not isinstance(array, str):
            if isinstance(array_type.element_type, XLSArrayType):
                array = [XLSArray(array_type=array_type.element_type, array=inner_array)
                         for inner_array in array]
            if not isinstance(array_type.size, str):
                assert len(array) == array_type.size, \
                    f'Array size mismatch: expected {array_type.size}, got {len(array)}'
        self.array = array

    def __str__(self):
        # TODO make it less verbose, e.g. replace:
        #   FixedPoint<16,-6>[2]:[FixedPoint<16,-6>{ significand = sN[16]:-1}, FixedPoint<16,-6>{ significand = sN[16]:235} ]
        # with
        #   fp_util::make_fixed_points_1d<-6>(sN[6][2]:[-1, 235])
        # NB: this works only when self.array contains explicit values, not string(s)!
        if isinstance(self.array, str):
            return f'{self.array_type}:[{self.array}]'
        elements = ', '.join(map(str, self.array))
        return f'{self.array_type}:[{elements}]'


class XLSFunctionCall:
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

    def __str__(self):
        params = ', '.join(map(str, self.params))
        if params:
            params = f'<{params}>'
        args = ', '.join(map(str, self.args))
        return f'{self.name}{params}({args})'


class XLSConst:
    def __init__(self, name, value, type=None):
        self.name = name
        self.value = value
        self.type = type

    def __str__(self):
        type = f': {self.type}' if self.type else ''
        return f'pub const {self.name}{type} = {self.value};'


class XLSTypeAlias:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return f'pub type {self.name} = {self.type};'


class XLSImport:
    def __init__(self, name, alias=None):
        self.name = name
        self.alias = alias

    def __str__(self):
        as_alias = f' as {self.alias}' if self.alias else ''
        return f'import {self.name}{as_alias};'


class XLSVariableDefinition:
    def __init__(self, name, value, type=None):
        self.name = name
        self.type = type
        self.value = value

    def __str__(self):
        type = f': {self.type}' if self.type else ''
        return f'let {self.name}{type} = {self.value};'


class XLSFunctionDefinition:
    def __init__(self, name, params, args, output_type, body):
        self.name = name
        self.params = params or []
        self.args = args or []
        self.output_type = output_type or '()'
        self.body = body or ''

    def __str__(self):
        if isinstance(self.params, str):
            params = self.params
        else:
            params = ', '.join(map(str, self.params))
        if params:
            params = f'<{params}>'
        if isinstance(self.args, str):
            args = self.args
        else:
            args = ', '.join(map(str, self.args))
        return f'''pub fn {self.name}{params}({args})
    -> {self.output_type} {{
    {self.body}
}}'''


class XLSTensorVariable:
    """Helper class to generate XLS constants for tensor variables.
    """

    def __init__(self, name: str, num_bits, binary_exponent, shape) -> None:
        if isinstance(shape, int) or isinstance(shape, str):
            shape = (shape,)
        name = name.upper()
        self.num_bits = XLSConst(f'{name}_NUM_BITS', num_bits, type='u32')
        self.binary_exponent = XLSConst(f'{name}_BINARY_EXPONENT', binary_exponent, type='s32')
        self.shape = tuple(
            XLSConst(f'{name}_DIM_{i}', dim, type='u32')
            for i, dim in enumerate(shape)
        )
        name = name[0].upper() + name[1:].lower()
        self.type_alias = XLSTypeAlias(name=f'{name}Type', type=self.to_array_type())
        self.type_alias_bits = XLSTypeAlias(name=f'{name}TypeBits', type=self.to_array_type_bits())

    @classmethod
    def from_tensor_variable(cls, name: str, var: TensorVariable) -> XLSTensorVariable:
        assert isinstance(var.type.precision, FixedPrecisionType), \
            f'Precision {var.__class__.__name__} must have FixedPrecisionType'
        element_type = XLSFixedPointType.from_precision(var.type.precision)
        return cls(
            name=name,
            num_bits=element_type.num_bits,
            binary_exponent=element_type.binary_exponent,
            shape=var.shape)

    def definitions(self) -> list[XLSConst | XLSTypeAlias]:
        return [self.num_bits, self.binary_exponent] + list(self.shape) + [self.type_alias, self.type_alias_bits]

    def to_array_type(self) -> XLSArrayType:
        return XLSArrayType(
            element_type=XLSFixedPointType(
                self.num_bits.name,
                binary_exponent=self.binary_exponent.name),
            shape=tuple(dim.name for dim in self.shape)
        )

    def to_array_type_bits(self) -> XLSArrayType:
        return XLSArrayType(
            element_type=XLSIntegerType(width=self.num_bits.name, signed=True),
            shape=tuple(dim.name for dim in self.shape)
        )


class XLSLookupTable:
    def __init__(
            self,
            name : str,
            input_precision: XLSFixedPointType | FixedPrecisionType,
            output_precision: XLSFixedPointType | FixedPrecisionType,
            x_min,
            log2_step,
            raw_table
    ) -> None:
        input_precision = as_xls_fixed_point_type(input_precision)
        output_precision = as_xls_fixed_point_type(output_precision)
        self.input_num_bits = XLSConst(f'{name}_INPUT_NUM_BITS', input_precision.num_bits, 'u32')
        self.input_binary_exponent = XLSConst(f'{name}_INPUT_BINARY_EXPONENT', input_precision.binary_exponent, 's32')
        self.output_num_bits = XLSConst(f'{name}_OUTPUT_NUM_BITS', output_precision.num_bits, 'u32')
        self.output_binary_exponent = XLSConst(f'{name}_OUTPUT_BINARY_EXPONENT', output_precision.binary_exponent,
                                               's32')
        self.size = XLSConst(f'{name}_SIZE', len(raw_table), 'u32')
        self.log2_step = XLSConst(f'{name}_LOG2_STEP', log2_step, 's32')
        self.x_min = XLSConst(f'{name}_X_MIN',
                              x_min,
                              XLSFixedPointType(
                                  num_bits=f'{name}_INPUT_NUM_BITS',
                                  binary_exponent=f'{name}_INPUT_BINARY_EXPONENT')
                              )
        int_table = XLSArray(
            array_type=XLSArrayType(
                element_type=XLSIntegerType(width=f'{name}_OUTPUT_NUM_BITS', signed=True),
                shape=f'{name}_SIZE'
            ),
            array=raw_table
        )
        fixed_point_table = XLSFunctionCall(name='fixed_point_util::make_fixed_points_1d',
                                            params=[self.output_binary_exponent.name],
                                            args=[int_table])
        self.lookup_table = XLSConst(name=name,
                                     value=XLSFunctionCall(name='lookup_table::create',
                                                           params=[self.log2_step.name],
                                                           args=[x_min, fixed_point_table]))

    def definitions(self) -> list[XLSConst]:
        return [self.input_num_bits, self.input_binary_exponent, self.output_num_bits, self.output_binary_exponent,
                self.size, self.log2_step, self.x_min, self.lookup_table]

    def __str__(self):
        return '\n'.join(map(str, self.definitions()))
