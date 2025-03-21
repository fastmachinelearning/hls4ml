from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    ExponentPrecisionType,
    FixedPrecisionConverter,
    FixedPrecisionType,
    InplaceStreamVariableConverter,
    IntegerPrecisionType,
    PrecisionDefinition,
    StreamVariableConverter,
    VariableDefinition,
    XnorPrecisionType,
)

# region ArrayVariable


class LiberoArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class LiberoInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class LiberoArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Libero', definition_cls=LiberoArrayVariableDefinition)


class LiberoInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Libero', definition_cls=LiberoInplaceArrayVariableDefinition)


# endregion

# region StreamVariable


class LiberoStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::FIFO<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::FIFO<{type}> {name}{suffix}({depth})'.format(
                type=self.type.name, name=self.name, depth=self.pragma[1], suffix=name_suffix
            )


class LiberoInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class LiberoStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Libero', definition_cls=LiberoStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class LiberoInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='Libero', definition_cls=LiberoInplaceStreamVariableDefinition
        )


# endregion

# region Precision types


class LAPIntegerPrecisionDefinition(PrecisionDefinition):
    def definition_cpp(self):
        typestring = 'hls::ap_{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring


class LAPFixedPrecisionDefinition(PrecisionDefinition):
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
        ]
        if args[2] == 'AP_TRN' and args[3] == 'AP_WRAP':
            # This is the default, so we won't write the full definition for brevity
            args[2] = args[3] = None

        args = ','.join([str(arg) for arg in args if arg is not None])
        typestring = 'hls::ap_{signed}fixpt<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring


class LAPTypeConverter(FixedPrecisionConverter):
    def __init__(self):
        super().__init__(
            type_map={
                FixedPrecisionType: LAPFixedPrecisionDefinition,
                IntegerPrecisionType: LAPIntegerPrecisionDefinition,
                ExponentPrecisionType: LAPIntegerPrecisionDefinition,
                XnorPrecisionType: LAPIntegerPrecisionDefinition,
            },
            prefix='LAP',
        )


# endregion
