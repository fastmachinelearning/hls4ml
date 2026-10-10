from hls4ml.backends.altera.altera_types import (
    AlteraACTypeConverter,
    AlteraArrayVariableConverter,
    AlteraHLSTypeConverter,
    AlteraInplaceArrayVariableConverter,
    AlteraInplaceStreamVariableConverter,
    AlteraInterfaceVariableConverter,
    AlteraStaticWeightVariableConverter,
    AlteraStreamVariableConverter,
)
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable

# from hls4ml.utils.string_utils import convert_to_pascal_case


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = AlteraHLSTypeConverter(precision_converter=AlteraACTypeConverter())
        self.array_var_converter = AlteraArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = AlteraInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.interface_var_converter = AlteraInterfaceVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = AlteraStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = AlteraInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = AlteraStaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                if out_name in node.model.inputs:
                    new_var = self.interface_var_converter.convert(var, pragma='stream')
                elif out_name in node.model.outputs:
                    new_var = self.interface_var_converter.convert(var, pragma='stream')
                elif isinstance(var, InplaceTensorVariable):
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
