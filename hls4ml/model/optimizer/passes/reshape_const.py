import numpy as np
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.hls_layers import Constant

class ReshapeConstant(OptimizerPass):
    """ Remove Constant from new shape input """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Reshape'
                    and len(node.inputs) > 1
                    and node.get_input_node(node.inputs[1]))

        return is_match
    
    def transform(self, model, node):
        """
        Remove Constant from new shape input. Note, input shape node is already used on initialize
        """
        shape_node =  node.get_input_node(node.inputs[1])
        if not isinstance(shape_node, Constant):
            raise "Nonconstant shape inputs are not currently suppoerted"
        model.remove_node(shape_node, rewire=False)
       
        return True