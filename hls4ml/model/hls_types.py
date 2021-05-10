import re
import numpy as np


class Quantizer(object):
    def __init__(self, bits, hls_type):
        self.bits = bits
        self.hls_type = hls_type
    
    def __call__(self, data):
        raise NotImplementedError

class PrecisionType(object):
    def __init__(self, width, signed):
        self.width = width
        self.signed = signed

class IntegerPrecisionType(PrecisionType):
    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)
        self.integer = width
        self.fractional = 0
    
    def __str__(self):
        typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring

    def __eq__(self, other):
        eq = self.width == other.width
        eq = eq and self.signed == other.signed
        # These are probably unnecessary
        eq = eq and self.integer == other.integer
        eq = eq and self.fractional == other.fractional
        return eq

class FixedPrecisionType(PrecisionType):
    def __init__(self, width=16, integer=6, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        super().__init__(width=width, signed=signed)
        self.integer = integer
        self.fractional = width-integer
        self.rounding_mode = rounding_mode
        self.saturation_mode = saturation_mode
        self.saturation_bits = saturation_bits
    
    def __str__(self):
        args = [self.width, self.integer, self.rounding_mode, self.saturation_mode, self.saturation_bits]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
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
    '''
    Convenience class to differentiate 'regular' integers from BNN Xnor ones
    '''
    def __init__(self):
        super().__init__(width=1, signed=False)

class ExponentPrecisionType(IntegerPrecisionType):
    '''
    Convenience class to differentiate 'regular' integers from those which represent exponents, for QKeras po2 quantizers, for example.
    '''
    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)

def find_minimum_width(data, signed=True):
    """
    Helper function to find the minimum integer width to express all entries in the data array
    without saturation / overflow
    """
    maxdata = np.amax(np.abs(data))
    if maxdata == 0.:
        # fringe case (amax(abs(data)) == 0 -> data is uniformly zero)
        return 1

    log2max = np.log2(maxdata)

    iwidth = max(0, int(np.ceil(log2max)))
    if iwidth == int(np.floor(log2max)): # is a power-of-two integer -> need one extra bit
        iwidth += 1

    if signed:
        # add the sign bit
        iwidth += 1

    return iwidth

class HLSType(object):
    def __init__(self, name, precision, **kwargs):
        self.name = name.format(**kwargs)
        self.precision = precision

    def definition_cpp(self):
        return 'typedef {precision} {name};\n'.format(name=self.name, precision=self.precision)

class CompressedType(HLSType):
    def __init__(self, name, precision, index_precision, **kwargs):
        super(CompressedType, self).__init__('compressed_type{index}', precision, **kwargs)
        self.index_precision = index_precision

    def definition_cpp(self):
        cpp_fmt = ('typedef struct {name} {{ '
               '{index} row_index; '
               '{index} col_index; '
               '{precision} weight; }} {name};\n')
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision)

class ExponentType(HLSType):
    def __init__(self, name, precision, **kwargs):
        super(ExponentType, self).__init__('exponent_type{index}', precision, **kwargs)

    def definition_cpp(self):
        cpp_fmt = ('typedef struct {name} {{ '
                   '{sign} sign; '
                   '{precision} weight; }} {name};\n')
        return cpp_fmt.format(name=self.name, precision=self.precision, sign=str(XnorPrecisionType()))

class PackedType(HLSType):
    def __init__(self, name, precision, n_elem, n_pack, **kwargs):
        super(PackedType, self).__init__(name, precision, **kwargs)
        self.n_elem = n_elem
        if n_pack < 0:
            self.n_pack = -n_pack
            self.unpack = True
        else:
            self.n_pack = n_pack
            self.unpack = False

    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(name=self.name, precision=self.precision, n_elem=str(self.n_elem) + n_elem_expr + str(self.n_pack))

class Variable(object):
    def __init__(self, var_name, atype, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = atype
        self.cppname = re.sub(r'\W|^(?=\d)','_', self.name)

class TensorVariable(Variable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, **kwargs):
        super(TensorVariable, self).__init__(var_name, HLSType(type_name, precision, **kwargs), **kwargs)
        self.shape = shape
        self.dim_names = dim_names

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

class InplaceVariable(Variable):
    def __init__(self, shape, dim_names, proxy, **kwargs):
        self.shape = shape
        self.dim_names = dim_names
        self.type = proxy.type
        self.name = proxy.name
        self.size = proxy.size

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def definition_cpp(self):
        return None

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

class WeightVariable(Variable):
    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super(WeightVariable, self).__init__(var_name, HLSType(type_name, precision, **kwargs), **kwargs)
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
                lsb = 2 ** fractional_bits
                if lsb < 1:
                    # Use str to represent the float with digits, get the length
                    # to right of decimal point
                    decimal_spaces = len(str(lsb).split('.')[1])
                else:
                    decimal_spaces = len(str(2**integer_bits)) 
                self.precision_fmt = '%.{}f'.format(decimal_spaces)
            else:
                self.precision_fmt = '%f'

    def definition_cpp(self):
        return '{type} {name}[{size}]'.format(type=self.type.name, name=self.cppname, size=self.data_length)

class CompressedWeightVariable(WeightVariable):
    def __init__(self, var_name, type_name, precision, data, reuse_factor, quantizer=None, **kwargs):
        super(CompressedWeightVariable, self).__init__(var_name, type_name, precision, data, quantizer=quantizer, **kwargs)
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
    def __init__(self, var_name, type_name, precision, data, quantizer, **kwargs):
        super(ExponentWeightVariable, self).__init__(var_name, type_name, precision, data, quantizer, **kwargs)
        '''
        WeightVariable for Exponent aka po2 data. The data should already by quantized by the quantizer.
        '''
        self.type = ExponentType(type_name, precision, **kwargs)
        self.shape = list(self.data.shape[:-1])

    def _format(self):
        y = self.data
        # Use an XnorBinary-like representation for the sign
        sign = np.where(y < 0, np.zeros_like(y), np.ones_like(y))
        # Take the logarithm, since this is what we will write to the header
        # for the optimized product using shifts
        y = (np.log2(np.abs(y)) / np.log2(2.)).astype('int')
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
