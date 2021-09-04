import numpy as np
from numpy.lib.arraysetops import isin

from hls4ml.model.hls_types import CompressedType, NamedType, ExponentType, FixedPrecisionType, IntegerPrecisionType, TensorVariable, PackedType, WeightVariable, XnorPrecisionType

#TODO Rethink if these classes should be built with `from_...(var, ...)` methods or with `__init__(var, ...)`

#region Precision types

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

class ACIntegerPrecisionType(IntegerPrecisionType):
    def definition_cpp(self):
        typestring = 'ac_int<{width}, {signed}>'.format(width=self.width, signed=str(self.signed).lower())
        return typestring
    
    @classmethod
    def from_precision(cls, precision_type):
        return cls(width=precision_type.width, signed=precision_type.signed)

class ACFixedPrecisionType(FixedPrecisionType):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def definition_cpp(self):
        args = [self.width, self.integer, str(self.signed).lower(), self._rounding_mode_cpp(self.rounding_mode), self._saturation_mode_cpp(self.saturation_mode), self.saturation_bits]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ac_fixed<{args}>'.format(args=args)
        return typestring
    
    @classmethod
    def from_precision(cls, precision_type):
        return cls(width=precision_type.width, integer=precision_type.integer, signed=precision_type.signed,
            rounding_mode=precision_type.rounding_mode, saturation_mode=precision_type.saturation_mode, saturation_bits=precision_type.saturation_bits)

class PrecisionTypeConverter(object):
    def convert(self, precision_type):
        raise NotImplementedError

class APTypeConverter(PrecisionTypeConverter):
    def convert(self, precision_type):
        if isinstance(precision_type, IntegerPrecisionType):
            return APIntegerPrecisionType.from_precision(precision_type)
        elif isinstance(precision_type, FixedPrecisionType):
            return APFixedPrecisionType.from_precision(precision_type)
        else:
            raise Exception('Unknown precision type: {}'.format(precision_type.__class__.__name__))

class ACTypeConverter(PrecisionTypeConverter):
    def convert(self, precision_type):
        if isinstance(precision_type, IntegerPrecisionType):
            return ACIntegerPrecisionType.from_precision(precision_type)
        elif isinstance(precision_type, FixedPrecisionType):
            return ACFixedPrecisionType.from_precision(precision_type)
        if isinstance(precision_type, (APIntegerPrecisionType, APFixedPrecisionType)):
            return precision_type
        else:
            raise Exception('Unknown precision type: {}'.format(precision_type.__class__.__name__))

#endregion

#region Data types

class HLSType(NamedType):
    def definition_cpp(self):
        return 'typedef {precision} {name};\n'.format(name=self.name, precision=self.precision.definition_cpp())

    @classmethod
    def from_type(cls, type, precision_converter):
        return cls(
            name=type.name,
            precision=precision_converter.convert(type.precision)
        )

class HLSCompressedType(CompressedType):
    def definition_cpp(self):
        cpp_fmt = (
            'typedef struct {name} {{'
            '{index} row_index;'
            '{index} col_index;'
            '{precision} weight; }} {name};\n'
        )
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision.definition_cpp())

    @classmethod
    def from_type(cls, type, precision_converter):
        return cls(
            name=type.name,
            precision=precision_converter.convert(type.precision),
            index_precision=type.index_precision
        )

class HLSExponentType(ExponentType):
    def definition_cpp(self):
        cpp_fmt = (
            'typedef struct {name} {{'
            '{sign} sign;'
            '{precision} weight; }} {name};\n'
        )
        return cpp_fmt.format(name=self.name, precision=self.precision.definition_cpp(), sign=str(XnorPrecisionType()))

    @classmethod
    def from_type(cls, type, precision_converter):
        return cls(
            name=type.name,
            precision=precision_converter.convert(type.precision)
        )

class HLSPackedType(PackedType):
    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(name=self.name, precision=self.precision.definition_cpp(), n_elem=str(self.n_elem) + n_elem_expr + str(self.n_pack))

    @classmethod
    def from_type(cls, type, precision_converter):
        return cls(
            name=type.name,
            precision=precision_converter.convert(type.precision),
            n_elem=type.n_elem,
            n_pack=type.n_pack
        )

class NamedTypeConverter(object):
    def convert(self, type, precision_converter):
        raise NotImplementedError

class HLSTypeConverter(NamedTypeConverter):
    def convert(self, type, precision_converter):
        if isinstance(type, PackedType):
            return HLSPackedType.from_type(type, precision_converter)
        elif isinstance(type, CompressedType):
            return HLSCompressedType.from_type(type, precision_converter)
        elif isinstance(type, ExponentType):
            return HLSExponentType.from_type(type, precision_converter)
        elif isinstance(type, NamedType):
            return HLSType.from_type(type, precision_converter)
        else:
            raise Exception('Unknown type: {}'.format(type.__class__.__name__))

#endregion

#region Variables

class ArrayVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name, type_name, precision, pragma='partition'):
        super(ArrayVariable, self).__init__(shape, dim_names, var_name, type_name, precision)
        self.type = HLSType(type_name, precision)
        self.pragma = pragma

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    @classmethod
    def from_variable(cls, tensor_var, precision_converter, pragma='partition'):
        return cls(
            tensor_var.shape,
            tensor_var.dim_names,
            var_name=tensor_var.name,
            type_name=tensor_var.type.name,
            precision=precision_converter.convert(tensor_var.type.precision),
            pragma=pragma
        )

class VivadoArrayVariable(ArrayVariable):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(type=self.type.name, name=self.cppname, suffix=name_suffix, shape=self.size_cpp())

class QuartusArrayVariable(ArrayVariable):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}] {pragma}'.format(type=self.type.name, name=self.cppname, suffix=name_suffix, shape=self.size_cpp(), pragma=self.pragma)


class StructMemberVariable(ArrayVariable):
    """Used by Quartus backend for input/output arrays that are members of the inputs/outpus struct"""
    def __init__(self, shape, dim_names, var_name, type_name, precision, pragma='hls_register', struct_name=None):
        super(StructMemberVariable, self).__init__(shape, dim_names, var_name, type_name, precision, pragma)
        assert struct_name is not None, 'struct_name must be provided when creating StructMemberVariable'
        self.struct_name = str(struct_name)
        self.member_name = self.name
        self.name = self.struct_name + '.' + self.member_name

    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(type=self.type.name, name=self.member_name, suffix=name_suffix, shape=self.size_cpp())

    @classmethod
    def from_variable(cls, tensor_var, precision_converter, pragma='partition', struct_name=None):
        return cls(
            tensor_var.shape,
            tensor_var.dim_names,
            var_name=tensor_var.name,
            type_name=tensor_var.type.name,
            precision=precision_converter.convert(tensor_var.type.precision),
            pragma=pragma,
            struct_name=struct_name
        )

class StreamVariable(TensorVariable):
    def __init__(self, shape, dim_names, var_name, type_name, precision, n_pack=1, depth=0):
        super(StreamVariable, self).__init__(shape, dim_names, var_name, type_name, precision)
        self.type = HLSPackedType(type_name, precision, shape[-1], n_pack)
        if depth == 0:
            depth = np.prod(shape) // shape[-1]
        self.pragma = ('stream', depth)

    def get_shape(self):
        return zip(self.dim_names, self.shape)

    def size_cpp(self):
        return '*'.join([str(k) for k in self.dim_names])

    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference: # Function parameter
            return 'hls::stream<{type}> &{name}{suffix}'.format(type=self.type.name, name=self.cppname, suffix=name_suffix)
        else: # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(type=self.type.name, name=self.cppname, suffix=name_suffix)

    @classmethod
    def from_variable(cls, tensor_var, precision_converter,  n_pack=1, depth=0):
        return cls(
            tensor_var.shape,
            tensor_var.dim_names,
            var_name=tensor_var.name,
            type_name=tensor_var.type.name,
            precision=precision_converter.convert(tensor_var.type.precision),
            n_pack=n_pack,
            depth=depth
        )

class StaticWeightVariable(WeightVariable):
    def __init__(self, weight_class, var_name, type_name, precision, data, index_precision=None):
        super(StaticWeightVariable, self).__init__(var_name, type_name, precision, data)
        self.weight_class = weight_class
        if self.weight_class == 'WeightVariable':
            self.type = HLSType(type_name, precision)
        elif self.weight_class == 'ExponentWeightVariable':
            self.type = HLSExponentType(type_name, precision)
        elif self.weight_class == 'CompressedWeightVariable':
            self.type = HLSCompressedType(type_name, precision, index_precision)
        else:
            raise Exception('Cannot create StaticWeightVariable, unknown weight class: {}'.format(self.weight_class))

    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}[{size}]'.format(type=self.type.name, name=self.cppname, size=self.data_length)
    
    @classmethod
    def from_variable(cls, weight_var, precision_converter, index_precision=None):
        return cls(
            weight_var.__class__.__name__,
            var_name=weight_var.name,
            type_name=weight_var.type.name,
            precision=precision_converter.convert(weight_var.type.precision),
            data=weight_var.data,
            index_precision=index_precision
        )

#endregion