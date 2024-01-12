'''
This package includes oneAPI-specific customizations to the variable types
'''
from hls4ml.backends.fpga.fpga_types import ArrayVariableConverter, VariableDefinition

# region ArrayVarable


class OneAPIArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return f'[[{self.pragma}]] std::array<{self.type.name}, {self.size_cpp()}> {self.name}{name_suffix}'


class OneAPIInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class OneAPIArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIArrayVariableDefinition)


class OneAPIInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIInplaceArrayVariableDefinition)


# endregion

# region InterfaceMemberVariable


class OneAPIInterfaceVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return f'[[{self.pragma}]] {self.array_type} {self.name}{name_suffix}'

    def declare_cpp(self, pipe_min_size=0, indent=''):
        lines = indent + f'class {self.pipe_id};\n'
        lines += indent + f'using {self.array_type} = std::array<{self.type.name}, {self.size_cpp()}>;\n'
        lines += indent + (
            f'using {self.pipe_name} = sycl::ext::intel::experimental::pipe<{self.pipe_id}, '
            + f'{self.array_type}, {pipe_min_size}, PipeProps>;\n'
        )
        return lines


class InterfaceVariableConverter:
    def __init__(self, type_converter, prefix, definition_cls):
        self.type_converter = type_converter
        self.prefix = prefix
        self.definition_cls = definition_cls

    def convert(self, tensor_var, pipe_name, pipe_id, array_type, pragma='partition'):
        if isinstance(tensor_var, self.definition_cls):  # Already converted
            return tensor_var

        tensor_var.pragma = pragma
        tensor_var.type = self.type_converter.convert(tensor_var.type)

        tensor_var.pipe_name = pipe_name
        tensor_var.pipe_id = pipe_id
        tensor_var.array_type = array_type

        tensor_var.__class__ = type(self.prefix + 'InterfaceMemberVariable', (type(tensor_var), self.definition_cls), {})
        return tensor_var


class OneAPIInterfaceVariableConverter(InterfaceVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIInterfaceVariableDefinition)


# endregion
