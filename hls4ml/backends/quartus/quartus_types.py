from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    InplaceStreamVariableConverter,
    StreamVariableConverter,
    StructMemberVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class QuartusArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}] {pragma}'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp(), pragma=self.pragma
        )


class QuartusInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class QuartusArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Quartus', definition_cls=QuartusArrayVariableDefinition)


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


class QuartusStructMemberVariableConverter(StructMemberVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Quartus', definition_cls=QuartusStructMemberVariableDefinition
        )


# endregion

# region StreamVariable


class QuartusStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return f'stream<{self.type.name}> {self.name}{name_suffix}'


class QuartusInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class QuartusStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Quartus', definition_cls=QuartusStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class QuartusInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Quartus', definition_cls=QuartusInplaceStreamVariableDefinition
        )


# endregion
