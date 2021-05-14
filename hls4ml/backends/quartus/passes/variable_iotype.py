
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.hls_types import Variable, TensorVariable
from hls4ml.backends.fpga.fpga_types import ArrayVariable, StreamVariable


class TransformVariables(GlobalOptimizerPass):
    def transform(self, model, node):
        for out_name, var in node.variables.items():
            io_type = node.model.config.get_config_value('IOType')

            if io_type == 'io_stream':
                new_var = StreamVariable.from_variable(var)
            elif io_type == 'io_parallel':
                if self.name in node.model.inputs:
                    new_var = ArrayVariable.from_variable(var) # TODO replace with struct member variable
                else:
                    new_var = ArrayVariable.from_variable(var, pragma='hls_register')
            else:
                raise Exception('Unknown IOType {} in {} ({})'.format(io_type, node.name, node.__class__.__name__))

            node.set_attr(out_name, new_var)    
