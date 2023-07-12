from hls4ml.model.layers import Constant, Reshape
from hls4ml.model.optimizer import OptimizerPass


class ReshapeConstant(OptimizerPass):
    """
    ONNX has the target shape come as an input, not a parameter. This removes
    the Constant input from new shape input. (Non-constant inputs are not supported.)
    The constant value was already used; this is just a cleanup uptimization.
    """

    def match(self, node):
        is_match = isinstance(node, Reshape) and len(node.inputs) > 1 and node.get_input_node(node.inputs[1])

        return is_match

    def transform(self, model, node):
        """
        Remove Constant from new shape input. Note, input shape node is already used on initialize
        """
        shape_node = node.get_input_node(node.inputs[1])
        node.inputs[1] = ''
        if not isinstance(shape_node, Constant):
            raise RuntimeError("Nonconstant shape inputs are not currently supported")
        model.remove_node(shape_node, rewire=False)

        return True
