from hls4ml.model.layers import Reshape
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class InplaceParallelReshape(OptimizerPass):
    """
    Replaces the output variable of Reshape layer with an inplace variable when using io_parallel.

    This is done because in io_parallel tensors are stored as flat arrays, requiring no reshaping.
    """

    def match(self, node):
        if not isinstance(node, Reshape):
            return False
        return node.model.config.get_config_value('IOType') == 'io_parallel'

    def transform(self, model, node):
        outvar = node.get_output_variable()
        invar = node.get_input_variable()
        newoutvar = InplaceTensorVariable(outvar, invar)
        node.set_attr(node.outputs[0], newoutvar)
        if node.name in model.outputs:
            prev_node = node.get_input_node()
            assert (
                prev_node.name not in model.outputs
            ), f"Cannot output node {prev_node.name}: reshape is a no-op in io_parallel.\
            As a result, the previous node {prev_node.name}'s output will be used as the\
            output. However, this node is already an output."
            model.outputs = [name if name != node.name else prev_node.name for name in model.outputs]
        return False
