from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    InplaceStreamVariableConverter,
    StreamVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class VivadoArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class VivadoInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class VivadoArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoArrayVariableDefinition)


class VivadoInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoInplaceArrayVariableDefinition)


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


class VivadoStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Vivado', definition_cls=VivadoStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class VivadoInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Vivado', definition_cls=VivadoInplaceStreamVariableDefinition
        )


# endregion
