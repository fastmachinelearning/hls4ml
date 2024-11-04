from hls4ml.backends.oneapi.oneapi_types import (
    OneAPIACTypeConverter,
    OneAPIArrayVariableConverter,
    OneAPIHLSTypeConverter,
    OneAPIInplaceArrayVariableConverter,
    OneAPIInplaceStreamVariableConverter,
    OneAPIInterfaceVariableConverter,
    OneAPIStaticWeightVariableConverter,
    OneAPIStreamVariableConverter,
)
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable

# from hls4ml.utils.string_utils import convert_to_pascal_case


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = OneAPIHLSTypeConverter(precision_converter=OneAPIACTypeConverter())
        self.array_var_converter = OneAPIArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = OneAPIInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.interface_var_converter = OneAPIInterfaceVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = OneAPIStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = OneAPIInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = OneAPIStaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                if out_name in node.model.inputs:
                    new_var = self.interface_var_converter.convert(var, pragma='stream')
                elif out_name in node.model.outputs:
                    new_var = self.interface_var_converter.convert(var, pragma='stream')
                if isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_stream_var_converter.convert(var, pragma='stream')
                else:
                    new_var = self.stream_var_converter.convert(var, pragma='stream')
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    new_var = self.interface_var_converter.convert(var, pragma='intel::fpga_register')
                elif out_name in node.model.outputs:
                    new_var = self.interface_var_converter.convert(var, pragma='intel::fpga_register')
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
