from hls4ml.backends.fpga.fpga_types import VariableDefinition
from hls4ml.backends.oneapi.oneapi_types import AggregratedArrayVariableConverter


# region InterfaceMemberVariable
class OneAPIAcceleratorInterfaceVariableDefinition(VariableDefinition):
    def definition_cpp(self, name_suffix='', as_reference=False):
        if self.pragma and not isinstance(self.pragma, tuple):
            return f'[[{self.pragma}]] {self.type.name} {self.name}{name_suffix}'
        else:
            return f'{self.type.name} {self.name}{name_suffix}'

    # Updated pipe min size to be 32 for simulation.
    def declare_cpp(self, pipe_min_size=32, indent=''):
        # Updated to use streaming beat for restartable streaming kernel.
        # Streaming beat is a wrapper type of the actual type with sideband control signals.
        # Syntax: using BeatT = sycl::ext::intel::experimental::StreamingBeat<DataT, eop, empty>;
        streaming_beat_t = f"{self.pipe_name}BeatT"
        lines = (
            f"{indent}class {self.pipe_id};\n"
            f"{indent}using {streaming_beat_t} = "
            f"sycl::ext::intel::experimental::StreamingBeat<{self.type.name}, true, true>;\n"
            f"{indent}using {self.pipe_name} = sycl::ext::intel::experimental::pipe<"
            f"{self.pipe_id}, {streaming_beat_t}, {pipe_min_size}, HostPipePropertiesT>;\n"
        )
        return lines


class OneAPIAcceleratorInterfaceVariableConverter(AggregratedArrayVariableConverter):
    def __init__(self, type_converter):
        super().__init__(
            type_converter=type_converter, prefix='OneAPI', definition_cls=OneAPIAcceleratorInterfaceVariableDefinition
        )
