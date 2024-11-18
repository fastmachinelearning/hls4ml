from hls4ml.model.layers import Reshape
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class InplaceStreamFlatten(OptimizerPass):
    """
    Replaces the output variable of Reshape (flatten) layer with an inplace variable when using io_stream.

    This optimizer avoids the expensive repacking of the stream when Reshape layer flattens the tensor to 1d.
    """

    def match(self, node):
        # Layers require flatten data can gather it from the stream, no need for repacking.
        # Reshape acts as a Flatten layer when the result has 1 dimension. Make it a inplace tensor if it happens.

        if node.model.config.get_config_value('IOType') != 'io_stream':
            return False
        if not (isinstance(node, Reshape) and len(node.get_output_variable().shape) == 1):
            # If is not flatten
            return False
        if node.name in node.model.outputs:
            # If used as model output. Output shape shall be preserved in this case.
            return False
        return True

    def transform(self, model, node):
        outvar = node.get_output_variable()
        invar = node.get_input_variable()
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        return False
