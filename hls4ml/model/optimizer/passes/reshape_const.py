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
        target_shape = shape_node.value
        if input_shape[0] is None:
            partial_shape = target_shape[1:]
            if -1 in partial_shape:
                print("WARNING: Inferring -1 shape ... ")
                dummy_x = np.ones(input_shape[1:])
                dummy_y = np.reshape(dummy_x, partial_shape)
                partial_shape = list(dummy_y.shape)
            target_shape = input_shape[:1] + partial_shape
        else:
            if -1 in target_shape:  #Need to infer shape for -1
                print("WARNING: Inferring -1 shape ... ")
                dummy_x = np.ones(input_shape)
                dummy_y = np.reshape(dummy_x, target_shape)
                target_shape = list(dummy_y.shape)
        node.set_attr('target_shape', target_shape)
        node.inputs[1] = ''
        model.remove_node(shape_node, rewire=False)
       
        return True