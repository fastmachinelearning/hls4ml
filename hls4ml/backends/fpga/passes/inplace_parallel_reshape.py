from hls4ml.model.layers import Reshape
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class InplaceParallelReshape(OptimizerPass):
    """
    Replaces the output variable of Reshape layer with an inplace variable when using io_parallel.

    This is done because in io_parallel tensors are stored as flat arrays, requiring no reshaping.
    """

    def match(self, node):
        return isinstance(node, Reshape)

    def transform(self, model, node):
        if model.config.get_config_value('IOType') != 'io_parallel':
            return False

        outvar = node.get_output_variable()
        invar = node.get_input_variable()
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        return False
