import numpy as np

from hls4ml.model.layers import Constant, Transpose
from hls4ml.model.optimizer import OptimizerPass


class RemoveUselessTranspose(OptimizerPass):
    def match(self, node):
        is_match = isinstance(node, Transpose) and list(node.get_attr('perm')) == [0]  # Useless transpose
        return is_match

    def transform(self, model, node):
        """
        Remove a transpose layer if it doesn't do anything. i.e 1D input and perm = [0]
        """
        print(f"Unnessary {node.name} in the model, optimizing ...")
        if not node.get_output_nodes():
            print(f"WARNING: {node.name} is the output layer! No rewiring performed.")
            model.remove_node(node, rewire=False)  # Don't rewire if there is no output layer
        else:
            model.remove_node(node, rewire=True)

        return True


class TransposeConstantFusion(OptimizerPass):
    """Remove Constant from new shape input"""

    def match(self, node):
        is_match = (
            isinstance(node, Transpose)
            and len(node.inputs) >= 0
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and list(node.get_attr('perm')) != [0]
        )

        return is_match

    def transform(self, model, node):
        """
        Change the shape of the constant
        """
        const_node = node.get_input_node(node.inputs[0])
        perm = node.get_attr('perm')
        new_val = np.transpose(const_node.value, perm)
        const_node.set_attr('value', new_val)
        const_node.value = new_val
        dims = [f'{const_node.name}_{i}' for i in range(len(perm))]
        self.add_output_variable(new_val.shape, dims, var_name=const_node.name, precision=const_node.get_attr("precision"))

        model.remove_node(node, rewire=True)
        return True
