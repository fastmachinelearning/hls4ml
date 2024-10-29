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
        if not (isinstance(node, Reshape) and len(node.get_output_variable().shape) == 1):
            # Reshape with multiple outputs will be kept as is, or repack cannot handle different shapes
            return False
        io_type = node.model.config.get_config_value('IOType')
        return io_type == 'io_stream'

    def transform(self, model, node):
        outvar = node.get_output_variable()
        invar = node.get_input_variable()
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        if node.name in model.outputs:
            prev_node = node.get_input_node()
            assert (
                prev_node.name not in model.outputs
            ), f"Cannot output node {prev_node.name}: In io_stream, flatten with a single output is a no-op. As a result, the previous node {prev_node.name}'s output will be used as the output. However, this node is already an output."  # noqa: E501
            model.outputs = [name if name != node.name else prev_node.name for name in model.outputs]
        return False
