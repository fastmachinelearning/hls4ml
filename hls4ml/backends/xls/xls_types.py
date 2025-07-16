from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    VariableDefinition,
)

# region ArrayVariable


class XLSArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return 'multi_dense_fxd::{type}<{width}, 1, {fraction}>'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, width=self.type.precision.width, 
            fraction=self.type.precision.width - self.type.precision.integer
        )


class XLSInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class XLSArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='XLS', definition_cls=XLSArrayVariableDefinition)


class XLSInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='XLS', definition_cls=XLSInplaceArrayVariableDefinition)