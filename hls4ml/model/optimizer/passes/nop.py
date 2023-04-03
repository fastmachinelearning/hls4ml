from hls4ml.model.layers import Activation
from hls4ml.model.optimizer import OptimizerPass


class EliminateLinearActivation(OptimizerPass):
    def match(self, node):
        cast = False
        if isinstance(node, Activation):
            cast = node.get_input_variable().type.precision != node.get_output_variable().type.precision
        return isinstance(node, Activation) and node.get_attr('activation') == 'linear' and not cast

    def transform(self, model, node):
        model.remove_node(node)
        return True
