from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    InplaceStreamVariableConverter,
    StreamVariableConverter,
    StructMemberVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class CatapultArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}] /* {pragma} */'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp(), pragma=self.pragma
        )


class CatapultInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class CatapultArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Catapult', definition_cls=CatapultArrayVariableDefinition)


class CatapultInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Catapult', definition_cls=CatapultInplaceArrayVariableDefinition
        )


# endregion

# region StructMemberVariable


class CatapultStructMemberVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.member_name, suffix=name_suffix, shape=self.size_cpp()
        )


class CatapultStructMemberVariableConverter(StructMemberVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Catapult', definition_cls=CatapultStructMemberVariableDefinition
        )


# endregion

# region StreamVariable


class CatapultStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'ac_channel<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration (string name arg not implemented in ac_channel)
            return 'ac_channel<{type}> {name}{suffix}/*("{name}")*/'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class CatapultStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Catapult', definition_cls=CatapultStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class CatapultInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class CatapultInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Catapult', definition_cls=CatapultInplaceStreamVariableDefinition
        )


# endregion
