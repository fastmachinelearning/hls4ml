from hls4ml.model.layers import Input, Reshape
from hls4ml.model.optimizer import OptimizerPass


class AbsorbReshapeIntoInput(OptimizerPass):
    def match(self, node):
        # Looking for a Reshape layer whose input is an Input layer
        if isinstance(node, Reshape):
            inp_node = node.get_input_node()
            if isinstance(inp_node, Input):
                return True
        return False

    def transform(self, model, node):
        # node == Reshape layer that matched
        inp_node = node.get_input_node()
        out_nodes = node.get_output_nodes()
        if len(out_nodes) > 1:
            raise Exception('Reshape node has multiple outputs')

        target_shape = node.get_attr('target_shape')

        input_output_var = inp_node.get_output_variable()
        input_output_var.shape = target_shape

        model.remove_node(node)

        return True  # because we modified the graph
