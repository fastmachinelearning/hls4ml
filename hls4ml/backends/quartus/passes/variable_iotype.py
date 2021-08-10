
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.hls_types import Variable, TensorVariable
from hls4ml.backends.fpga.fpga_types import ArrayVariable, StreamVariable, StructMemberVariable


class TransformVariables(GlobalOptimizerPass):
    def transform(self, model, node):
        for out_name, var in node.variables.items():
            io_type = node.model.config.get_config_value('IOType')

            if io_type == 'io_stream':
                new_var = StreamVariable.from_variable(var)
            elif io_type == 'io_parallel':
                if node.name in node.model.inputs:
                    if isinstance(var, StructMemberVariable):
                        new_var = var
                    else:
                        new_var = StructMemberVariable.from_variable(var, pragma='hls_register', struct_name='inputs')
                elif node.name in node.model.outputs:
                    if isinstance(var, StructMemberVariable):
                        new_var = var
                    else:
                        new_var = StructMemberVariable.from_variable(var, pragma='hls_register', struct_name='outputs')
                else:
                    new_var = ArrayVariable.from_variable(var, pragma='hls_register')
            else:
                raise Exception('Unknown IOType {} in {} ({})'.format(io_type, node.name, node.class_name))

            node.set_attr(out_name, new_var)    
