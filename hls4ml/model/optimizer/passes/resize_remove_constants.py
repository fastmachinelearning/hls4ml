from hls4ml.model.layers import Constant, Resize
from hls4ml.model.optimizer import OptimizerPass


class ResizeRemoveConstants(OptimizerPass):
    """
    This optimizer is intended to clean the Resize node from RoI and Scales parameters that if left cause issues in hls4ml.
    """

    def match(self, node):
        is_match = isinstance(node, Resize) and len(node.inputs) > 1
        return is_match

    def transform(self, model, node):
        """
        Remove RoI and Scale Constant from new shape input.
        """
        # see doc here: https://onnx.ai/onnx/operators/onnx__Resize.html
        scales_idx = 2 if len(node.inputs) == 3 or len(node.inputs) == 4 else 1
        scales_node = node.get_input_node(node.inputs[scales_idx])
        node.inputs[scales_idx] = ''
        if not isinstance(scales_node, Constant):
            raise RuntimeError("Non-constant shape inputs are not supported")
        model.remove_node(scales_node, rewire=False)
        if len(node.inputs) >= 3 and node.inputs[1] != '':
            # RoI is present only if more than 3 inputs are specified
            # RoI position is always 1 when present
            roi_node = node.get_input_node(node.inputs[1])
            node.inputs[1] = ''
            if not isinstance(roi_node, Constant):
                raise RuntimeError("Non-constant RoI inputs are not supported")
            model.remove_node(roi_node, rewire=False)
        if len(node.inputs) == 4:
            # Remove sizes node
            sizes_node = node.get_input_node(node.inputs[-1])
            node.inputs[-1] = ''
            if not isinstance(sizes_node, Constant):
                raise RuntimeError("Non-constant RoI inputs are not supported")
            model.remove_node(sizes_node, rewire=False)
        # Clean all the '' inputs
        node.inputs = list(filter(None, node.inputs))
        return True
