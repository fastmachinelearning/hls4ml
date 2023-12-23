from hls4ml.backends.fpga.fpga_types import (
    ACTypeConverter,
    HLSTypeConverter,
    StaticWeightVariableConverter,
)
from hls4ml.backends.oneapi.oneapi_types import (
    OneAPIArrayVariableConverter,
    OneAPIInplaceArrayVariableConverter,
    OneAPIInterfaceVariableConverter
)
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable
from hls4ml.utils.string_utils import convert_to_pascal_case

class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=ACTypeConverter())
        self.array_var_converter = OneAPIArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = OneAPIInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.interface_var_converter = OneAPIInterfaceVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                raise NotImplementedError("io_stream is not yet implemented for oneAPI")
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    new_var = self.interface_var_converter.convert(var, pragma='intel::fpga_register',
                                                                   pipe_name=f'{convert_to_pascal_case(var.name)}Pipe',
                                                                   pipe_id=f'{convert_to_pascal_case(var.name)}PipeID',
                                                                   array_type=f'{var.name}_array_t')
                elif out_name in node.model.outputs:
                    new_var = self.interface_var_converter.convert(var, pragma='intel::fpga_register',
                                                                   pipe_name=f'{convert_to_pascal_case(var.name)}Pipe',
                                                                   pipe_id=f'{convert_to_pascal_case(var.name)}PipeID',
                                                                   array_type=f'{var.name}_array_t')
                elif isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_array_var_converter.convert(var, pragma='')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='intel::fpga_register')
            else:
                raise Exception(f'Unknown IOType {io_type} in {node.name} ({node.class_name})')

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
