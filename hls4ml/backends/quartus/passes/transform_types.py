
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.hls_types import CompressedWeightVariable, InplaceVariable
from hls4ml.backends.fpga.fpga_types import ACIntegerPrecisionType, ACTypeConverter, QuartusArrayVariable, HLSTypeConverter, StaticWeightVariable, StreamVariable, StructMemberVariable


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=ACTypeConverter())

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if isinstance(var, InplaceVariable):
                continue

            if io_type == 'io_stream':
                new_var = StreamVariable.from_variable(var)
            elif io_type == 'io_parallel':
                if node.name in node.model.inputs:
                    if isinstance(var, StructMemberVariable):
                        new_var = var
                    else:
                        new_var = StructMemberVariable.from_variable(var, self.type_converter, pragma='hls_register', struct_name='inputs')
                elif node.name in node.model.outputs:
                    if isinstance(var, StructMemberVariable):
                        new_var = var
                    else:
                        new_var = StructMemberVariable.from_variable(var, self.type_converter, pragma='hls_register', struct_name='outputs')
                else:
                    new_var = QuartusArrayVariable.from_variable(var, self.type_converter, pragma='hls_register')
            else:
                raise Exception('Unknown IOType {} in {} ({})'.format(io_type, node.name, node.class_name))

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            new_weight = StaticWeightVariable.from_variable(weight, self.type_converter)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
