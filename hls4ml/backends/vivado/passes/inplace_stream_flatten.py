from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Reshape
from hls4ml.model.types import InplaceTensorVariable


class InplaceStreamFlatten(OptimizerPass):
    ''' Remove Flatten layer in io_stream '''
    def match(self, node):
        # optimizer pass for a flatten layer (1 output dimension)
        return isinstance(node, Reshape) and len(node.get_output_variable().shape) == 1

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_stream':
            return False

        outvar = node.get_output_variable()
        invar = node.get_input_variable(node.inputs[0])
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        return False
