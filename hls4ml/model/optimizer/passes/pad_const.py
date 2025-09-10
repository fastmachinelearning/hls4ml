from hls4ml.model.layers import Constant, ZeroPadding1D, ZeroPadding2D
from hls4ml.model.optimizer import OptimizerPass


class PaddingConstant(OptimizerPass):
    """
    ONNX has the padding come as an input, not a parameter. This removes the Constant node from the input.
    The constant value was already used; this is just a cleanup uptimization.
    """

    def match(self, node):
        is_match = (
            isinstance(node, (ZeroPadding1D, ZeroPadding2D))
            and len(node.inputs) > 1
            and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )

        return is_match

    def transform(self, model, node):
        """
        Remove Constant node(s) from the graph. Note, padding is already present in ZeroPadding node.
        """
        if len(node.inputs) > 2:
            const_val_node = node.get_input_node(node.inputs[2])
            if not isinstance(const_val_node, Constant):
                raise RuntimeError(f'Non-constant padding inputs are not currently supported ({node.name})')
            model.remove_node(const_val_node)
            node.inputs.pop(2)

        pad_node = node.get_input_node(node.inputs[1])
        if not isinstance(pad_node, Constant):
            raise RuntimeError(f'Non-constant padding inputs are not currently supported ({node.name})')
        model.remove_node(pad_node)
        node.inputs.pop(1)

        return True
