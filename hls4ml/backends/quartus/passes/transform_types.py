
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceVariable
from hls4ml.backends.fpga.fpga_types import ACTypeConverter, QuartusArrayVariableConverter, HLSTypeConverter, QuartusInplaceVariableConverter, QuartusStructMemberVariableConverter, StaticWeightVariableConverter


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=ACTypeConverter())
        self.array_var_converter = QuartusArrayVariableConverter(type_converter=self.type_converter)
        self.struct_var_converter = QuartusStructMemberVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)
        self.inplace_var_converter = QuartusInplaceVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if isinstance(var, InplaceVariable):
                new_var = self.inplace_var_converter.convert(var, io_type)

            if io_type == 'io_stream':
                raise Exception('Streaming IO is not supported in Quartus.')
            elif io_type == 'io_parallel':
                if node.name in node.model.inputs:
                    new_var = self.struct_var_converter.convert(var, pragma='hls_register', struct_name='inputs')
                elif node.name in node.model.outputs:
                    new_var = self.struct_var_converter.convert(var, pragma='hls_register', struct_name='outputs')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='hls_register')
            else:
                raise Exception('Unknown IOType {} in {} ({})'.format(io_type, node.name, node.class_name))

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = self.weight_var_converter.convert(weight)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
