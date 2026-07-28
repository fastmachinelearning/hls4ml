from hls4ml.backends.fpga.fpga_types import (
    ArrayVariableConverter,
    CompressedTypeConverter,
    ExponentTypeConverter,
    HLSTypeConverter,
    InplaceStreamVariableConverter,
    NamedTypeConverter,
    StreamVariableConverter,
    TypeDefinition,
    TypePrecisionConverter,
    VariableDefinition,
)
from hls4ml.model.types import CompressedType, ExponentType, NamedType, PackedType

# region PackedType
#
# Bug A workaround: emit a concrete (non-template) struct per stream payload
# type instead of `typedef nnet::array<T,N> name;`. With the templated
# typedef, Bambu's InterfaceInfer reads the AXIS TDATA Bitwidth from the
# inner element (W) instead of the aggregate (N*W); the simulator's
# per-beat channel layout then doesn't match the C-sim gold reference, and
# cosim aborts with
#     ERROR: MDPI driver: Channel parameter mismatch with respect to gold
# A concrete struct with the same members forces InterfaceInfer to use
# sizeof(struct) for the AXIS Bitwidth.
# (Explicit specialization of `nnet::array` does NOT help — concrete struct
# is the only emission Bambu picks up correctly.)


class BambuPackedTypeConverter(TypeDefinition, TypePrecisionConverter):
    def definition_cpp(self):
        n_elem_expr = '/' if self.unpack else '*'
        n_elem = str(self.n_elem) + n_elem_expr + str(self.n_pack)
        precision = self.precision.definition_cpp()
        name = self.name
        # User-defined element-wise `operator=` (Bug B workaround). Without
        # it, clang lowers `pack = stream.read()` as a single aggregate
        # copy, which Bambu's InterfaceInfer setReadInterface pass refuses
        # to handle:
        #   error -> unexpected condition (gc->args.size() == 2)
        #     void InterfaceInfer::setReadInterface(...)
        # The element-wise loop body lowers to per-element loads/stores
        # that setReadInterface pattern-matches.
        return (
            f'struct {name} {{\n'
            f'    typedef {precision} value_type;\n'
            f'    static const unsigned size = {n_elem};\n'
            f'    {precision} data[{n_elem}];\n'
            f'    {precision} &operator[](size_t pos) {{ return data[pos]; }}\n'
            f'    const {precision} &operator[](size_t pos) const {{ return data[pos]; }}\n'
            f'    {name} &operator=(const {name} &other) {{\n'
            f'        if (&other == this) return *this;\n'
            f'        #pragma clang loop unroll(full)\n'
            f'        for (unsigned i = 0; i < size; i++) data[i] = other.data[i];\n'
            f'        return *this;\n'
            f'    }}\n'
            f'    bool operator==(const {name} &other) const {{\n'
            f'        for (unsigned i = 0; i < size; i++)\n'
            f'            if (data[i] != other.data[i]) return false;\n'
            f'        return true;\n'
            f'    }}\n'
            f'    bool operator!=(const {name} &other) const {{ return !(*this == other); }}\n'
            f'}};\n'
        )


class BambuHLSTypeConverter(HLSTypeConverter):
    def __init__(self, precision_converter):
        self.precision_converter = precision_converter
        self.type_map = {
            NamedType: NamedTypeConverter,
            CompressedType: CompressedTypeConverter,
            ExponentType: ExponentTypeConverter,
            PackedType: BambuPackedTypeConverter,
        }


# endregion

# region ArrayVariable


class BambuArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        return '{type} {name}{suffix}[{shape}]'.format(
            type=self.type.name, name=self.name, suffix=name_suffix, shape=self.size_cpp()
        )


class BambuInplaceArrayVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class BambuArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuArrayVariableDefinition)


class BambuInplaceArrayVariableConverter(ArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuInplaceArrayVariableDefinition)


# endregion

# region StreamVariable


class BambuStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if as_reference:  # Function parameter
            return f'hls::stream<{self.type.name}> &{self.name}{name_suffix}'
        else:  # Declaration
            return 'hls::stream<{type}> {name}{suffix}("{name}")'.format(
                type=self.type.name, name=self.name, suffix=name_suffix
            )


class BambuInplaceStreamVariableDefinition(VariableDefinition):
    def definition_cpp(self):
        return f'auto& {self.name} = {self.input_var.name}'


class BambuStreamVariableConverter(StreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuStreamVariableDefinition)


# endregion

# region InplaceStreamVariable


class BambuInplaceStreamVariableConverter(InplaceStreamVariableConverter):
    def __init__(self, type_converter):
        super().__init__(type_converter=type_converter, prefix='Bambu', definition_cls=BambuInplaceStreamVariableDefinition)


# endregion
