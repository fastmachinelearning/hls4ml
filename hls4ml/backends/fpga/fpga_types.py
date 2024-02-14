import numpy as np

from hls4ml.model.types import (
    CompressedType,
    ExponentPrecisionType,
    ExponentType,
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    PackedType,
    XnorPrecisionType,
)

# region Precision types


class PrecisionDefinition:
    def definition_cpp(self):
        raise NotImplementedError


class APIntegerPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = 'ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring


class APFixedPrecisionDefinition(PrecisionDefinition):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AP_' + str(mode)

    def definition_cpp(self):
        args = [
            self.width,
            self.integer,
            self._rounding_mode_cpp(self.rounding_mode),
            self._saturation_mode_cpp(self.saturation_mode),
            self.saturation_bits,
        ]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'ap_{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring


class ACIntegerPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = f'ac_int<{self.width}, {str(self.signed).lower()}>'
        return typestring


class ACFixedPrecisionDefinition(PrecisionDefinition):
    def _rounding_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def _saturation_mode_cpp(self, mode):
        if mode is not None:
            return 'AC_' + str(mode)

    def definition_cpp(self):
        args = [
            self.width,
            self.integer,
            str(self.signed).lower(),
            self._rounding_mode_cpp(self.rounding_mode),
            self._saturation_mode_cpp(self.saturation_mode),
            self.saturation_bits,
        ]
        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = f'ac_fixed<{args}>'
        return typestring


class PrecisionConverter:
    def convert(self, precision_type):
        raise NotImplementedError


class FixedPrecisionConverter(PrecisionConverter):
    def __init__(self, type_map, prefix):
        self.type_map = type_map
        self.prefix = prefix

    def convert(self, precision_type):
        type_cls = type(precision_type)
        type_cls_name = type_cls.__name__

        # If the type is already converted, do nothing
        if type_cls_name.startswith(self.prefix):
            return precision_type

        definition_cls = self.type_map.get(type_cls, None)

        if definition_cls is not None:
            precision_type.__class__ = type(self.prefix + type_cls_name, (type_cls, definition_cls), {})
            return precision_type
        else:
            raise Exception(f'Cannot convert precision type to {self.prefix}: {precision_type.__class__.__name__}')


class APTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: APFixedPrecisionDefinition,
                IntegerPrecisionType: APIntegerPrecisionDefinition,
                ExponentPrecisionType: APIntegerPrecisionDefinition,
                XnorPrecisionType: APIntegerPrecisionDefinition,
            },
            prefix='AP',
        )


class ACTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: ACFixedPrecisionDefinition,
                IntegerPrecisionType: ACIntegerPrecisionDefinition,
                ExponentPrecisionType: ACIntegerPrecisionDefinition,
                XnorPrecisionType: ACIntegerPrecisionDefinition,
            },
            prefix='AC',
        )


# endregion

# region Data types


class TypeDefinition:
    def definition_cpp(self):
        raise NotImplementedError


class TypePrecisionConverter:
    def convert_precision(self, precision_converter):
        self.precision = precision_converter.convert(self.precision)


class NamedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        return f'typedef {self.precision.definition_cpp()} {self.name};\n'


class CompressedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        cpp_fmt = 'typedef struct {name} {{' '{index} row_index;' '{index} col_index;' '{precision} weight; }} {name};\n'
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.index_precision = precision_converter.convert(self.index_precision)


class ExponentTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        cpp_fmt = 'typedef struct {name} {{' '{sign} sign;' '{precision} weight; }} {name};\n'
        return cpp_fmt.format(name=self.name, precision=self.precision.definition_cpp(), sign=self.sign.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.sign = precision_converter.convert(self.sign)


class PackedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(
            name=self.name,
            precision=self.precision.definition_cpp(),
            n_elem=str(self.n_elem) + n_elem_expr + str(self.n_pack),
        )


class HLSTypeConverter:
    def __init__(self, precision_converter):
        self.precision_converter = precision_converter
        self.type_map = {
            NamedType: NamedTypeConverter,
            CompressedType: CompressedTypeConverter,
            ExponentType: ExponentTypeConverter,
            PackedType: PackedTypeConverter,
        }

    def convert(self, atype):
        type_cls = type(atype)
        type_cls_name = type_cls.__name__

        # If the type is already converted, do nothing
        if type_cls_name.startswith('HLS'):
            return atype

        conversion_cls = self.type_map.get(type_cls, None)

        if conversion_cls is not None:
            atype.__class__ = type('HLS' + type_cls_name, (type_cls, conversion_cls), {})
            atype.convert_precision(self.precision_converter)
            return atype
        else:
            raise Exception(f'Cannot convert type: {atype.__class__.__name__}')


# endregion

# region Variables


class VariableDefinition:
    def definition_cpp(self, name_suffix='', as_reference=False):
        raise NotImplementedError


# region ArrayVariable


class VivadoArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class QuartusArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}] {pragma}'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp(), pragma=self.pragma
        )


class VivadoInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class QuartusInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class ArrayVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pragma='partition'):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        tensor_var.type = self.type_converter.convert(tensor_var.type)

        tensor_var.__class__ = type(self.prefix + 'ArrayVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class VivadoArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoArrayVariableDefinition)


class QuartusArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Quartus', definition_cls=QuartusArrayVariableDefinition)


class VivadoInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoInplaceArrayVariableDefinition)


class QuartusInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Quartus', definition_cls=QuartusInplaceArrayVariableDefinition
        )


# endregion

# region StructMemberVariable


class QuartusStructMemberVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.member_name, suffix=name_suffix, shape=self.size_cpp()
        )


class StructMemberVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pragma='partition', struct_name=None):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        tensor_var.type = self.type_converter.convert(tensor_var.type)

        assert struct_name is not None, 'struct_name must be provided when creating a StructMemberVariable'
        tensor_var.struct_name = str(struct_name)
        tensor_var.member_name = tensor_var.name
        tensor_var.name = tensor_var.struct_name + '.' + tensor_var.member_name

        tensor_var.__class__ = type(self.prefix + 'StructMemberVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class QuartusStructMemberVariableConverter(StructMemberVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Quartus', definition_cls=QuartusStructMemberVariableDefinition
        )


# endregion

# region StreamVariable


class VivadoStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class VivadoInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class QuartusStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return f'stream<{self.type.name}> {self.name}{name_suffix}'


class QuartusInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class StreamVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, n_pack=1, depth=0):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        if depth == 0:
            depth = np.prod(tensor_var.shape) // tensor_var.shape[-1]
        tensor_var.pragma = ('stream', depth)
        tensor_var.type = self.type_converter.convert(
            PackedType(tensor_var.type.name, tensor_var.type.precision, tensor_var.shape[-1], n_pack)
        )

        tensor_var.__class__ = type(self.prefix + 'StreamVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class VivadoStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoStreamVariableDefinition)


class QuartusStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Quartus', definition_cls=QuartusStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class InplaceStreamVariableConverter(StreamVariableConverter):
    def convert(self, tensor_var, n_pack=1, depth=0):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = None
        tensor_var.type = self.type_converter.convert(
            PackedType(tensor_var.type.name, tensor_var.type.precision, tensor_var.input_var.shape[-1], n_pack)
        )

        tensor_var.__class__ = type(self.prefix + 'StreamVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class VivadoInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Vivado', definition_cls=VivadoInplaceStreamVariableDefinition
        )


class QuartusInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Quartus', definition_cls=QuartusInplaceStreamVariableDefinition
        )


# endregion

# region WeightsVariable


class StaticWeightVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return f'{self.type.name} {self.name}[{self.data_length}]'


class StaticWeightVariableConverter:
    def __init__(self, type_converter):
        self.type_converter = type_converter

    def convert(self, weight_var):
        if isinstance(weight_var, StaticWeightVariableDefinition):  # Already converted
            return weight_var

        weight_var.weight_class = weight_var.__class__.__name__
        weight_var.storage = 'register'
        weight_var.type = self.type_converter.convert(weight_var.type)

        weight_var.__class__ = type('StaticWeightVariable', (type(weight_var), StaticWeightVariableDefinition), {})
        return weight_var


class BramWeightVariableConverter:
    @classmethod
    def convert(cls, weight_var):
        weight_var.storage = 'bram'
        return weight_var


# endregion

# endregion
