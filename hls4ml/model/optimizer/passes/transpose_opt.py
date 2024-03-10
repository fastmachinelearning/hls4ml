from hls4ml.model.layers import Input, Transpose
from hls4ml.model.optimizer import OptimizerPass


class RemoveNopTranspose(OptimizerPass):
    """
    Remove a transpose layer if it doesn't do anything to a 1D array. i.e, 1D input and perm = [0]
    """

    def match(self, node):
        is_match = isinstance(node, Transpose) and node.get_attr('perm') == [0]  # Useless transpose
        return is_match

    def transform(self, model, node):
        print(f'Unnecessary transpose node ({node.name}) detected, optimizing ...')
        if not node.get_output_nodes():
            print(f'WARNING: {node.name} is the output layer! No rewiring performed.')
            model.remove_node(node, rewire=False)  # Don't rewire if there is no output layer
        else:
            model.remove_node(node, rewire=True)

        return True


class RemoveSingleChannelTranspose(OptimizerPass):
    """
    Remove transpose of inputs if the number of channels is 1 as for io_parallel this doesn't affect the array
    representation used
    """

    def match(self, node):
        if node.model.config.get_config_value('IOType') != 'io_parallel':
            return False

        return (
            isinstance(node, Transpose)
            and isinstance(node.get_input_node(), Input)
            and node.get_input_variable().shape[0] == 1
        )

    def transform(self, model, node):
        # Adjust the input shape and remove the Transpose node
        input_var = node.get_input_variable()
        input_var.shape.append(input_var.shape.pop(0))
        model.remove_node(node)

        return True
