"""
This module contains the definitions of classes hls4ml uses to represent data types. The data types are equivalents of
C++/HLS data types. The basic type(``PrecisionType``) is defined as having a specified width in bits (it's 'precision').
The Precision types are given names for convenience (``NamedType``). Named types are the building blocks of
higher-dimensional tensors, which are defined as arrays or FIFO streams in the generated code.
"""

import re
from enum import Enum

import numpy as np
import tensorflow as tf
from qkeras.quantizers import get_quantizer

# region Quantizer definition


class Quantizer:
    """
    Base class for representing quantizers in hls4ml.

    Subclasses of ``Quantizer`` are expected to wrap the quantizers of upstream tools (e.g., QKeras).

    Args:
        bits (int): Total number of bits used by the quantizer.
        hls_type (NamedType): The hls4ml type used by the quantizer.
    """

    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type

    def __call__(self, data):
        raise NotImplementedError


class BinaryQuantizer(Quantizer):
    """Quantizer that quantizes to 0 and 1 (``bits=1``) or -1 and 1 (``bits==2``).

    Args:
        bits (int, optional): Number of bits used by the quantizer. Defaults to 2.

    Raises:
        Exception: Raised if ``bits>2``
    """

    def __init__(self, bits=2):
        if bits == 1:
            hls_type = IntegerPrecisionType(width=1, signed=False)
        elif bits == 2:
            hls_type = IntegerPrecisionType(width=2)
        else:
            raise Exception(f'BinaryQuantizer suppots 1 or 2 bits, but called with bits={bits}')
        super().__init__(bits, hls_type)

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        quant_data = data
        if self.bits == 1:
            quant_data = np.where(data > 0, ones, zeros).astype('int')
        if self.bits == 2:
            quant_data = np.where(data > 0, ones, -ones)
        return quant_data


class TernaryQuantizer(Quantizer):
    """Quantizer that quantizes to -1, 0 and 1."""

    def __init__(self):
        super().__init__(2, IntegerPrecisionType(width=2))

    def __call__(self, data):
        zeros = np.zeros_like(data)
        ones = np.ones_like(data)
        return np.where(data > 0.5, ones, np.where(data <= -0.5, -ones, zeros))


class QKerasQuantizer(Quantizer):
    """Wrapper around QKeras quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        self.quantizer_fn = get_quantizer(config)
        self.alpha = config['config'].get('alpha', None)
        if config['class_name'] == 'quantized_bits':
            self.bits = config['config']['bits']
            self.hls_type = self._get_type(config)
        # ! includes stochastic_ternary
        elif 'ternary' in config['class_name']:
            self.bits = 2
            self.hls_type = IntegerPrecisionType(width=2, signed=True)
        # ! includes stochastic_binary
        elif 'binary' in config['class_name']:
            self.bits = 1
            self.hls_type = XnorPrecisionType()
        else:
            print("Unsupported quantizer: " + config['class_name'])
            self.bits = 16
            self.hls_type = FixedPrecisionType(width=16, integer=6, signed=True)

    def __call__(self, data):
        tf_data = tf.convert_to_tensor(data)
        return self.quantizer_fn(tf_data).numpy()
        # return self.quantizer_fn(data)

    def _get_type(self, quantizer_config):
        width = quantizer_config['config']['bits']
        integer = quantizer_config['config'].get('integer', 0)
        if quantizer_config['class_name'] == 'quantized_po2':
            return ExponentPrecisionType(width=width, signed=True)
        if width == integer:
            if width == 1:
                return XnorPrecisionType()
            else:
                return IntegerPrecisionType(width=width, signed=True)
        else:
            return FixedPrecisionType(width=width, integer=integer + 1, signed=True)


class QKerasBinaryQuantizer(Quantizer):
    """Wrapper around QKeras binary quantizer.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config, xnor=False):
        self.bits = 1 if xnor else 2
        self.hls_type = XnorPrecisionType() if xnor else IntegerPrecisionType(width=2, signed=True)
        self.alpha = config['config']['alpha']
        # Use the QKeras quantizer to handle any stochastic / alpha stuff
        self.quantizer_fn = get_quantizer(config)
        # Then we use our BinaryQuantizer to convert to '0,1' format
        self.binary_quantizer = BinaryQuantizer(1) if xnor else BinaryQuantizer(2)

    def __call__(self, data):
        x = tf.convert_to_tensor(data)
        y = self.quantizer_fn(x).numpy()
        return self.binary_quantizer(y)


class QKerasPO2Quantizer(Quantizer):
    """Wrapper around QKeras power-of-2 quantizers.

    Args:
        config (dict): Config of the QKeras quantizer to wrap.
    """

    def __init__(self, config):
        self.bits = config['config']['bits']
        self.quantizer_fn = get_quantizer(config)
        self.hls_type = ExponentPrecisionType(width=self.bits, signed=True)

    def __call__(self, data):
        # Weights are quantized to nearest power of two
        x = tf.convert_to_tensor(data)
        y = self.quantizer_fn(x)
        if hasattr(y, 'numpy'):
            y = y.numpy()
        return y


# endregion

# region Precision types


class RoundingMode(Enum):
    TRN = 1
    TRN_ZERO = 2
    RND = 3
    RND_ZERO = 4
    RND_INF = 5
    RND_MIN_INF = 6
    RND_CONV = 7

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, mode):
        mode = mode.strip().replace('AP_', '').upper()
        mode = mode.strip().replace('AC_', '').upper()

        return cls[mode]


class SaturationMode(Enum):
    WRAP = 1
    SAT = 2
    SAT_ZERO = 3
    SAT_SYM = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, mode):
        mode = mode.strip().replace('AP_', '').upper()
        mode = mode.strip().replace('AC_', '').upper()

        return cls[mode]


class PrecisionType:
    """
    Base class representing a precision type of specified width.

    Subclasses of this provide concrete implementations of arbitrary precision integer and fixed-point types.

    Args:
        width (int): Number of bits used by the precision type.
        signed (bool): Signed or unsigned type.
    """

    def __init__(self, width, signed):
        self.width = width
        self.signed = signed


class IntegerPrecisionType(PrecisionType):
    """Arbitrary precision integer  data type.

    This type is equivalent to ap_(u)int and ac_int HLS types.

    Args:
        width (int, optional): Number of bits used. Defaults to 16.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True``.
    """

    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)
        self.integer = width
        self.fractional = 0

    def __str__(self):
        typestring = '{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring

    def __eq__(self, other):
        eq = self.width == other.width
        eq = eq and self.signed == other.signed
        # These are probably unnecessary
        eq = eq and self.integer == other.integer
        eq = eq and self.fractional == other.fractional
        return eq


class FixedPrecisionType(PrecisionType):
    """Arbitrary precision fixed-point data type.

    This type is equivalent to ap_(u)fixed and ac_fixed HLS types.

    Args:
        width (int, optional): Total number of bits used. Defaults to 16.
        integer (int, optional): Number of integer bits left of the decimal point. Defaults to 6.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True``.
        rounding_mode (RoundingMode, optional): Quantization mode. Defaults to ``None`` (TRN).
        saturation_mode (SaturationMode, optional): Overflow mode. Defaults to ``None`` (WRAP).
        saturation_bits (int, optional): The number of saturation bits. Defaults to ``None``.
    """

    def __init__(self, width=16, integer=6, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        super().__init__(width=width, signed=signed)
        self.integer = integer
        self.fractional = width - integer
        self.rounding_mode = rounding_mode
        self.saturation_mode = saturation_mode
        self.saturation_bits = saturation_bits

    @property
    def rounding_mode(self):
        return self._rounding_mode

    @rounding_mode.setter
    def rounding_mode(self, mode):
        if isinstance(mode, str):
            self._rounding_mode = RoundingMode.from_string(mode)
        else:
            self._rounding_mode = mode

    @property
    def saturation_mode(self):
        return self._saturation_mode

    @saturation_mode.setter
    def saturation_mode(self, mode):
        if isinstance(mode, str):
            self._saturation_mode = SaturationMode.from_string(mode)
        else:
            self._saturation_mode = mode

    def __str__(self):
        args = [self.width, self.integer, self.rounding_mode, self.saturation_mode, self.saturation_bits]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = '{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring

    def __eq__(self, other):
        eq = self.width == other.width
        eq = eq and self.integer == other.integer
        eq = eq and self.fractional == other.fractional
        eq = eq and self.signed == other.signed
        eq = eq and self.rounding_mode == other.rounding_mode
        eq = eq and self.saturation_mode == other.saturation_mode
        eq = eq and self.saturation_bits == other.saturation_bits
        return eq


class XnorPrecisionType(IntegerPrecisionType):
    """
    Convenience class to differentiate 'regular' integers from BNN Xnor ones
    """

    def __init__(self):
        super().__init__(width=1, signed=False)


class ExponentPrecisionType(IntegerPrecisionType):
    """
    Convenience class to differentiate 'regular' integers from those which represent exponents,
    for QKeras po2 quantizers, for example.
    """

    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)


def find_minimum_width(data, signed=True):
    """
    Helper function to find the minimum integer width to express all entries in the data array
    without saturation / overflow.

    Args:
        data (ndarray): Data array.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True.``

    Returns:
        int: Minimum integer width required.
    """
    maxdata = np.amax(np.abs(data))
    if maxdata == 0.0:
        # fringe case (amax(abs(data)) == 0 -> data is uniformly zero)
        return 1

    log2max = np.log2(maxdata)

    iwidth = max(0, int(np.ceil(log2max)))
    if iwidth == int(np.floor(log2max)):  # is a power-of-two integer -> need one extra bit
        iwidth += 1

    if signed:
        # add the sign bit
        iwidth += 1

    return iwidth


# endregion

# region Data type definitions


class NamedType:
    """Class representing a named type.

    For convenience, hls4ml gives names to data types used in the generated HLS. This is equivalent to defining types
    in C/C++ like::
        typedef precision name;

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
    """

    def __init__(self, name, precision, **kwargs):
        self.name = name.format(**kwargs)
        self.precision = precision


class CompressedType(NamedType):
    """Class representing a compressed type in COO format.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
        index_precision (PrecisionType): Precision of the index of COO format.
    """

    def __init__(self, name, precision, index_precision, **kwargs):
        if not name.startswith('compressed_'):
            name = 'compressed_' + name
        super().__init__(name, precision, **kwargs)
        self.index_precision = index_precision


class ExponentType(NamedType):
    """Special type used to mark an exponent type, used by the power-of-2 quantizers.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
    """

    def __init__(self, name, precision, **kwargs):

        if not name.startswith('exponent_'):
            name = 'exponent_' + name
        super().__init__(name, precision, **kwargs)
        self.sign = XnorPrecisionType()


class PackedType(NamedType):
    """A type where multiple elements of the tensor are concatenated and stored as a single element, used by the streaming
    implementations to store elements of the last dimension of a tensor as a single element.

    The tensor of shape ``(H, W, C)`` will be represented as a FIFO stream having ``H * W / n_pack`` elements where each
    element will be a concatenation of ``n_elem * n_pack`` elements of the original tensor.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
        n_elem (int): Number of packed elements.
        n_pack (int): _description_
    """

    def __init__(self, name, precision, n_elem, n_pack, **kwargs):
        super().__init__(name, precision, **kwargs)
        self.n_elem = n_elem
        if n_pack < 0:
            self.n_pack = -n_pack
            self.unpack = True
        else:
            self.n_pack = n_pack
            self.unpack = False


# endregion

# region Variables


class Variable:
    """Base class representing a named multidimensional tensor.

    Args:
        var_name (str): Name of the variable in the generated C++/HLS.
        atype (NamedType): Data type used by the tensor.
    """

    def __init__(self, var_name, atype, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = atype


class TensorVariable(Variable):
    """Class representing the output of a layer (like an activation tensor).

    Args:
        shape (list, tuple): Shape of the tensor.
        dim_names (list, tuple): Names given to the dimensions of the tensor.
        var_name (str, optional): Name of the variable in the generated C++/HLS. Defaults to ``layer{index}``.
        type_name (str, optional): Name of the data type used (in NamedType). Defaults to ``layer{index}_t``.
        precision (PrecisionType, optional): Precision data type. Defaults to ``None``.
    """

    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, **kwargs):
        super().__init__(var_name, NamedType(type_name, precision, **kwargs), **kwargs)
        self.shape = shape
        self.dim_names = dim_names

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

    def size_cpp(self):
        # TODO get rid of size_cpp() (and dim_names)
        return '*'.join([str(k) for k in self.dim_names])


class InplaceTensorVariable(TensorVariable):
    """A ``TensorVariable`` that is just a link to another ``TensorVariable``.

    Args:
        tv (TensorVariable): The tensor variable to link.
        input_var (_type_): The input variable that should be should link to.
    """

    def __init__(self, tv, input_var):
        self.__dict__.update(tv.__dict__)
        self.type = input_var.type
        self.input_var = input_var


class WeightVariable(Variable):
    """Class representing a tensor containing the weights of a layer.

    Precision type of the instance can be modified with the ``update_precision`` method.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super().__init__(var_name, NamedType(type_name, precision, **kwargs), **kwargs)
        self.data = data
        self.nzeros = -1
        self.shape = list(self.data.shape)
        self.data_length = np.prod(self.data.shape)
        self.nonzeros = np.count_nonzero(self.data)
        self.nzeros = self.data_length - self.nonzeros
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self._iterator = None
        self.update_precision(precision)
        self.quantizer = quantizer

    def __iter__(self):
        self._iterator = np.nditer(self.data, order='C')
        return self

    def __next__(self):
        if not self._iterator.finished:
            value = self._iterator[0]
            self._iterator.iternext()
            return self.precision_fmt % value
        else:
            raise StopIteration

    next = __next__

    def update_precision(self, new_precision):
        self.type.precision = new_precision
        precision_str = str(self.type.precision)
        if 'int' in precision_str:
            self.precision_fmt = '%d'
        else:
            match = re.search('.+<(.+?)>', precision_str)
            if match is not None:
                precision_bits = match.group(1).split(',')
                width_bits = int(precision_bits[0])
                integer_bits = int(precision_bits[1])
                fractional_bits = integer_bits - width_bits
                lsb = 2**fractional_bits
                if lsb < 1:
                    # Use str to represent the float with digits, get the length
                    # to right of decimal point
                    decimal_spaces = len(str(lsb).split('.')[1])
                else:
                    decimal_spaces = len(str(2**integer_bits))
                self.precision_fmt = f'%.{decimal_spaces}f'
            else:
                self.precision_fmt = '%f'


class CompressedWeightVariable(WeightVariable):
    """Class representing a tensor containing the weights of a layer represented in the COO format.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        reuse_factor (_type_): The reuse factor used to pad the data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, reuse_factor, quantizer=None, **kwargs):
        super().__init__(var_name, type_name, precision, data, quantizer=quantizer, **kwargs)
        self.extra_zeros = 0
        self.data_length = np.prod(data.shape) - self.nzeros
        while self.data_length % reuse_factor != 0:
            self.extra_zeros += 1
            self.data_length += 1
        self.nonzeros = np.prod(data.shape) - self.nzeros + self.extra_zeros

        # Compress the array
        weights = []
        extra_nzero_cnt = self.extra_zeros
        it = np.nditer(data, order='C', flags=['multi_index'])
        max_idx = 0
        while not it.finished:
            val = it[0]
            if not (val == 0 and extra_nzero_cnt < 1):
                if val == 0:
                    extra_nzero_cnt -= 1
                if it.multi_index[0] > max_idx:
                    max_idx = it.multi_index[0]
                if it.multi_index[1] > max_idx:
                    max_idx = it.multi_index[1]
                weights.append([it.multi_index[1], it.multi_index[0], val])
            it.iternext()
        weights.sort()

        index_precision = 32
        if max_idx > 0:
            index_precision = int(np.log2(max_idx) + 1)
        self.type = CompressedType(type_name, precision, IntegerPrecisionType(width=index_precision, signed=False), **kwargs)

        self.data = weights

    def __iter__(self):
        self._iterator = iter(self.data)
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt % value[2]
        return '{ %u, %u, %s }' % (value[1], value[0], value_fmt)

    next = __next__


class ExponentWeightVariable(WeightVariable):
    """WeightVariable for Exponent aka power-of-2 data. The data should already by quantized by the quantizer.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super().__init__(var_name, type_name, precision, data, quantizer, **kwargs)
        self.type = ExponentType(type_name, precision, **kwargs)
        self.shape = list(self.data.shape[:-1])

    def _format(self):
        y = self.data
        # Use an XnorBinary-like representation for the sign
        sign = np.where(y < 0, np.zeros_like(y), np.ones_like(y))
        # Take the logarithm, since this is what we will write to the header
        # for the optimized product using shifts
        y = (np.log2(np.abs(y)) / np.log2(2.0)).astype('int')
        return np.stack((sign, y), axis=-1)

    def __iter__(self):
        data = self._format()
        self._iterator = iter(data.reshape((np.product(data.shape[:-1]), 2)))
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt % value[1]
        return '{%d, %s}' % (value[0], value_fmt)

    next = __next__


# endregion

# region Custom source


class Source:
    """Class representing generated source code blocks.

    Args:
        code (str): Generated source code.
    """

    def __init__(self, code):
        self.code = code

    def __str__(self):
        return str(self.code)


# endregion
