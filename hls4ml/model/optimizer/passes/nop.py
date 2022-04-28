from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Activation

class EliminateLinearActivation(OptimizerPass):
    def match(self, node):
        cast = False
        if isinstance(node, Activation):
            cast = node.get_input_variable().type.precision != node.get_output_variable().type.precision
            return node.get_attr('activation') == 'linear' and not cast
        else:
            return False
    
    def transform(self, model, node):
        model.remove_node(node)
        return True

class EliminateLinearActivationQuant(OptimizerPass):
    '''
    This is to optimize away lots of linear qantizations in QONNX. May have to restrict it
    more if it causes problems.
    '''
    def match(self, node):
        '''
        Only match if this activation is from quant node and previous node precision is not set  by a quant node already.
        '''
        is_match = (isinstance(node, Activation) and node.get_attr('activation') == 'linear'
                    and node.get_attr("quant_precision")
                    and not node.get_input_node(node.inputs[0]).get_attr("quant_precision"))
        return is_match

    def transform(self, model, node):
        prev_node = node.get_input_node(node.inputs[0]);
        quant_precision = node.get_attr("quant_precision")
        prev_node.set_attr("quant_precision", quant_precision)
        prev_node.set_attr("quantizer", node.get_attr("quantizer"))
        prev_node.update_output_precision(quant_precision)
        model.remove_node(node)
        return True
