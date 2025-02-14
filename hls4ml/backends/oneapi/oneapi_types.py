'''
This package includes oneAPI-specific customizations to the variable types
'''

import numpy as np

from hls4ml.backends.fpga.fpga_types import (
    ACFixedPrecisionDefinition,
    ACIntegerPrecisionDefinition,
    FixedPrecisionConverter,
    HLSTypeConverter,
    NamedTypeConverter,
    PrecisionDefinition,
    TypeDefinition,
    TypePrecisionConverter,
    VariableDefinition,
)
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
from hls4ml.utils.fixed_point_utils import next_pow2
from hls4ml.utils.string_utils import convert_to_pascal_case


class ACExponentPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = f'std::pair<ac_int<1, false>, ac_int<{self.width}, true>>'
        return typestring


class OneAPIACTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: ACFixedPrecisionDefinition,
                IntegerPrecisionType: ACIntegerPrecisionDefinition,
                ExponentPrecisionType: ACExponentPrecisionDefinition,
                XnorPrecisionType: ACIntegerPrecisionDefinition,
            },
            prefix='AC',
        )


class OneAPICompressedTypeConverter(TypeDefinition, TypePrecisionConverter):
    """Use a tuple for storing a compressed type for oneAPI since it's better supported. (Currently unused)"""

    def definition_cpp(self):
        """tuple format is row_index, col_index, weight"""
        cpp_fmt = 'typedef std::tuple<{index}, {index}, {precision}> {name};\n'
        return cpp_fmt.format(name=self.name, index=self.index_precision, precision=self.precision.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.index_precision = precision_converter.convert(self.index_precision)


class OneAPIExponentTypeConverter(TypeDefinition, TypePrecisionConverter):
    """Use a pair for storing a exponent type for oneAPI since it's better supported"""

    def definition_cpp(self):
        cpp_fmt = 'typedef std::pair<{sign}, {precision}> {name};\n'
        return cpp_fmt.format(name=self.name, precision=self.precision.definition_cpp(), sign=self.sign.definition_cpp())

    def convert_precision(self, precision_converter):
        super().convert_precision(precision_converter)
        self.sign = precision_converter.convert(self.sign)


class OneAPIPackedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        return 'typedef nnet::array<{precision}, {n_elem}> {name};\n'.format(
            name=self.name,
            precision=self.precision.definition_cpp(),
            n_elem=str(self.n_elem) + n_elem_expr + str(self.n_pack),
        )

    def convert_precision(self, precision_converter):
        self.precision = precision_converter.convert(self.precision)


class OneAPIHLSTypeConverter(HLSTypeConverter):
    def __init__(self, precision_converter):
        self.precision_converter = precision_converter
        self.type_map = {
            NamedType: NamedTypeConverter,
            CompressedType: OneAPICompressedTypeConverter,
            ExponentType: OneAPIExponentTypeConverter,
            PackedType: OneAPIPackedTypeConverter,
        }


# region ArrayVarable


class OneAPIArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if self.pragma and not isinstance(self.pragma, tuple):
            return f'[[{self.pragma}]] {self.type.name} {self.name}{name_suffix}'
        else:
            return f'{self.type.name} {self.name}{name_suffix}'


class OneAPIInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class AggregratedArrayVariableConverter:
    """This is a bit of an extension of the standard ArrayVariableConverter"""

    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pragma='', depth=0, n_pack=1):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        if pragma == 'stream':
            if depth == 0:
                depth = np.prod(tensor_var.shape) // tensor_var.shape[-1]
            tensor_var.pragma = ('stream', depth)
            n_elem = tensor_var.shape[-1]
        else:
            tensor_var.pragma = pragma
            n_elem = tensor_var.size()
            n_pack = 1  # ignore any passed value

        tensor_var.type = self.type_converter.convert(
            PackedType(tensor_var.type.name, tensor_var.type.precision, n_elem, n_pack)
        )

        # pipe_name and pipe_id are only used for io_stream and interface variables in io_parallel
        tensor_var.pipe_name = f'{convert_to_pascal_case(tensor_var.name)}Pipe'
        tensor_var.pipe_id = f'{convert_to_pascal_case(tensor_var.name)}PipeID'

        tensor_var.__class__ = type(self.prefix + 'AggregateArrayVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class OneAPIArrayVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIArrayVariableDefinition)


class OneAPIInplaceArrayVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIInplaceArrayVariableDefinition)


# endregion

# region InterfaceMemberVariable


class OneAPIInterfaceVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if self.pragma and not isinstance(self.pragma, tuple):
            return f'[[{self.pragma}]] {self.type.name} {self.name}{name_suffix}'
        else:
            return f'{self.type.name} {self.name}{name_suffix}'

    def declare_cpp(self, pipe_min_size=0, indent=''):
        lines = indent + f'class {self.pipe_id};\n'
        lines += indent + (
            f'using {self.pipe_name} = sycl::ext::intel::experimental::pipe<{self.pipe_id}, '
            + f'{self.type.name}, {pipe_min_size}, PipeProps>;\n'
        )
        return lines


class OneAPIInterfaceVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIInterfaceVariableDefinition)


# endregion


# region StreamVariable
class OneAPIStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=True):
        return f'{self.name}{name_suffix}'

    def declare_cpp(self, indent=''):
        lines = indent + f'class {self.pipe_id};\n'
        lines += indent + (
            f'using {self.pipe_name} = sycl::ext::intel::experimental::pipe<{self.pipe_id}, '
            + f'{self.type.name}, {self.pragma[-1]}>;\n'
        )
        return lines


class OneAPIInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'using {self.name} = {self.input_var.name}'


class OneAPIStreamVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIStreamVariableDefinition)


class OneAPIInplaceStreamVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIInplaceStreamVariableDefinition
        )


# region WeightsVariable


class OneAPIStaticWeightVariableDefinition(VariableDefinition):
    def definition_cpp(self, reuse_factor):
        """Write the appropriate weight definiiton"""
        # first determine whether to store in register or bram (heuristic)
        if reuse_factor == 1 or self.data_length < 2048 or self.type.precision.width < 3:
            attribute = '[[intel::fpga_register]]'
        else:
            # revisit this heuristic
            nbanks = int(2 ** np.ceil(np.log2(self.data_length)) / 2)
            var_width = int(np.ceil(self.type.precision.width / 8))
            bwidth = next_pow2(var_width)
            attribute = (
                f'[[intel::bankwidth({bwidth}), intel::numbanks({nbanks}), '
                'intel::max_replicates(1), intel::fpga_memory("BLOCK_RAM")]]'
            )
        if self.storage == 'register':
            return f'{attribute} static constexpr {self.type.name} {self.name}'
        else:
            return f'{attribute} {self.type.name} {self.name}'


class OneAPIStaticWeightVariableConverter:
    def __init__(self, type_converter):
        self.type_converter = type_converter

    def convert(self, weight_var):
        if isinstance(weight_var, OneAPIStaticWeightVariableDefinition):  # Already converted
            return weight_var

        weight_var.weight_class = weight_var.__class__.__name__
        weight_var.storage = 'register'
        weight_var.type = self.type_converter.convert(
            PackedType(weight_var.name + '_t', weight_var.type.precision, weight_var.data_length, 1)
        )

        weight_var.__class__ = type(
            'OneAPIStaticWeightVariable', (type(weight_var), OneAPIStaticWeightVariableDefinition), {}
        )
        return weight_var


# endregion

# endregion
