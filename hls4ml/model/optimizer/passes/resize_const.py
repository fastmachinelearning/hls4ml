from hls4ml.model.layers import Constant, Resize
from hls4ml.model.optimizer import OptimizerPass


class ResizeConstant(OptimizerPass):
    """
    To compute the output shape of resize is necessary to access the scales, that
    are stored as initilizer, later on converted as constant inputs.
    """

    def match(self, node):
        is_match = isinstance(node, Resize) and len(node.inputs) > 1 and node.get_input_node(node.inputs[-1])
        return is_match

    def transform(self, model, node):
        """
        Remove Constant from new shape input. Note, input shape node is already used on initialize
        """
        scales_node = node.get_input_node(node.inputs[-1])
        node.inputs[-1] = ''
        scales_values = scales_node.get_attr('value')
        node.set_attr('out_width', int(node.get_attr('in_width') * scales_values[1]))
        node.set_attr('out_height', int(node.get_attr('in_height') * scales_values[2]))
        if not isinstance(scales_node, Constant):
            raise RuntimeError("Non-constant shape inputs are not supported")
        model.remove_node(scales_node, rewire=False)
        return True
