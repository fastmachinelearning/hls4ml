from hls4ml.model.layers import Activation, BatchNormalization, Conv1D, Conv2D, Dense
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import UnspecifiedPrecisionType


class EliminateLinearActivation(OptimizerPass):
    def match(self, node):
        cast = False
        if isinstance(node, Activation):
            cast = node.get_input_variable().type.precision != node.get_output_variable().type.precision
        return isinstance(node, Activation) and node.get_attr('activation') == 'linear' and not cast

    def transform(self, model, node):
        model.remove_node(node)
        return True


class MergeLinearActivation(OptimizerPass):
    '''
    For many objects it's safe to change the output precision independently of the calculation.
    '''

    def match(self, node):
        '''
        Only match if the parent is safe and the precision is not explicitly set.
        '''
        if isinstance(node, Activation) and node.get_attr('activation') == 'linear':
            parent = node.get_input_node(node.inputs[0])
            safe_parent = isinstance(parent, (Dense, Conv1D, Conv2D, BatchNormalization))
            return safe_parent and isinstance(parent.get_output_variable().type.precision, UnspecifiedPrecisionType)
        else:
            return False

    def transform(self, model, node):
        prev_node = node.get_input_node(node.inputs[0])
        quantizer = node.get_attr("quantizer")
        prev_node.set_attr("quantizer", quantizer)
        prev_node.update_output_precision(quantizer.hls_type)
        model.remove_node(node)
        return True