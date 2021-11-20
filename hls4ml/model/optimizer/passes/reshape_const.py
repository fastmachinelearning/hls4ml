import numpy as np
from hls4ml.model.optimizer import OptimizerPass

class ReshapeConstant(OptimizerPass):
    """ Remove Constant from new shape input """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'Reshape'
                    and len(node.inputs) > 1
                    and node.get_input_node(node.inputs[1])
                    and node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant') 

        return is_match
    
    def transform(self, model, node):
        """
        Remove Constant from new shape input
        """
        shape_node =  node.get_input_node(node.inputs[1])
        print(f"Removing {shape_node.name} attached to {node.name}")

        input_shape =  node.get_input_variable(node.inputs[0]).shape
        target_shape = node.infer_shape(input_shape, shape_node.value)

        node.set_attr('target_shape', target_shape)
        node.inputs[1] = ''
        model.remove_node(shape_node, rewire=False)
       
        return True