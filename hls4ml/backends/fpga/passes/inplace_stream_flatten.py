from hls4ml.model.layers import Reshape
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class InplaceStreamFlatten(OptimizerPass):
    """
    Replaces the output variable of Reshape (flatten) layer with an inplace variable when using io_stream.

    This optimizer avoids the expensive repacking of the stream when Reshape layer flattens the tensor to 1d.
    """

    def match(self, node):
        # Reshape acts as a Flatten layer when the result has 1 dimension
        if not (isinstance(node, Reshape)):
            # Reshape with multiple outputs will be kept as is, or repack cannot handle different shapes
            return False
        if len(node.get_output_variable().shape) + (node.name in node.model.outputs) != 1:
            return False
        io_type = node.model.config.get_config_value('IOType')
        return io_type == 'io_stream'

    def transform(self, model, node):
        outvar = node.get_output_variable()
        invar = node.get_input_variable()
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        return False
