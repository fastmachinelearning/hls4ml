import numpy as np

from hls4ml.model.hls_types import FixedPrecisionType, IntegerPrecisionType, Variable, TensorVariable, PackedType

class APIntegerPrecisionType(IntegerPrecisionType):
    def definition_cpp(self):
        typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring
    
    @classmethod
    def from_precision(cls, precision_type):
        return cls(width=precision_type.width, signed=precision_type.signed)

class APFixedPrecisionType(FixedPrecisionType):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def definition_cpp(self):
        args = [self.width, self.integer, self._rounding_mode_cpp(self.rounding_mode), self._saturation_mode_cpp(self.saturation_mode), self.saturation_bits]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring
    
    @classmethod
    def from_precision(cls, precision_type):
        return cls(width=precision_type.width, integer=precision_type.integer, signed=precision_type.signed,
            rounding_mode=precision_type.rounding_mode, saturation_mode=precision_type.saturation_mode, saturation_bits=precision_type.saturation_bits)

class ArrayVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, pragma='partition', **kwargs):
        super(ArrayVariable, self).__init__(shape, dim_names, var_name, type_name, precision, **kwargs)
        self.pragma = pragma

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    @classmethod
    def from_variable(cls, tensor_var, pragma='partition', **kwargs):
        return cls(tensor_var.shape, tensor_var.dim_names, var_name=tensor_var.name, type_name=tensor_var.type.name, precision=tensor_var.type.precision, pragma=pragma)


class StreamVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name='layer{index}', type_name='layer{index}_t', precision=None, n_pack=1, depth=0, **kwargs):
        super(StreamVariable, self).__init__(shape, dim_names, var_name, type_name, precision, **kwargs)
        self.type = PackedType(type_name, precision, shape[-1], n_pack, **kwargs)
        if depth == 0:
            depth = np.prod(shape) // shape[-1]
        self.pragma = ('stream', depth)

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    @classmethod
    def from_variable(cls, tensor_var, n_pack=1, depth=0, **kwargs):
        return cls(tensor_var.shape, tensor_var.dim_names, var_name=tensor_var.name, type_name=tensor_var.type.name, precision=tensor_var.type.precision, n_pack=n_pack, depth=depth, **kwargs)
