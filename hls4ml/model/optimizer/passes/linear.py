from hls4ml.model.layers import Activation, BatchNormalization, Conv1D, Conv2D, Dense
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


# TODO:  Move migrate this to auto precisoin check from quant precision check
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
            parent_type_fixed = parent.get_attr("quant_precision")
            return safe_parent and not parent_type_fixed
        else:
            return False

    def transform(self, model, node):
        prev_node = node.get_input_node(node.inputs[0])
        quant_precision = node.get_attr("quant_precision")
        prev_node.set_attr("quant_precision", quant_precision)
        prev_node.set_attr("quantizer", node.get_attr("quantizer"))
        prev_node.update_output_precision(quant_precision)
        model.remove_node(node)
        return True
