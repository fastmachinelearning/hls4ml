from warnings import warn

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
        roi_index = 1
        scales_idx = 2
        scales_node = node.get_input_node(node.inputs[scales_idx])
        node.inputs[scales_idx] = ''
        if not isinstance(scales_node, Constant):
            raise RuntimeError("Non-constant shape inputs are not supported")
        model.remove_node(scales_node, rewire=False)
        # RoI position is always 1 when present
        roi_node = node.get_input_node(node.inputs[roi_index])
        if roi_node.get_attr('value'):
            warn('RoI value vector is not empty. Consider that RoI is not supported in hls4ml', stacklevel=2)
        node.inputs[roi_index] = ''
        if not isinstance(roi_node, Constant):
            raise RuntimeError("Non-constant RoI inputs are not supported")
        model.remove_node(roi_node, rewire=False)
        # Clean all the '' inputs
        node.inputs = list(filter(None, node.inputs))
        return True
