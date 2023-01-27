import numpy as np

from hls4ml.model.layers import Constant, Reshape
from hls4ml.model.optimizer import OptimizerPass


class ReshapeConstant(OptimizerPass):
    """Remove Constant from new shape input"""

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


class ReshapeConstantFusion(OptimizerPass):
    """Remove Constant from new shape input"""

    def match(self, node):
        is_match = (
            isinstance(node, Reshape)
            and len(node.inputs) >= 0
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and (len(node.inputs) == 1 or not node.get_input_node(node.inputs[1]))
        )

        return is_match

    def transform(self, model, node):
        """
        Change the shape of the constant
        """
        const_node = node.get_input_node(node.inputs[0])
        target_shape = node.get_attr('target_shape')
        new_val = np.reshape(const_node.value, target_shape)
        const_node.set_attr('value', new_val)
        const_node.value = new_val
        dims = [f'{const_node.name}_{i}' for i in range(len(target_shape))]
        self.add_output_variable(target_shape, dims, var_name=const_node.name, precision=const_node.get_attr("precision"))

        model.remove_node(node, rewire=True)
        return True
