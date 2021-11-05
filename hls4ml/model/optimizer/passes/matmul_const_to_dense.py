import numpy as np
from hls4ml.model.optimizer import OptimizerPass

class MatmulConstToDense(OptimizerPass):
    """ Convert MatMul with constant to a dense layer """
    def match(self, node):
        if node.__class__.__name__ == 'MatMul' and len(node.inputs) == 2:
            if node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant':
                merge_node = node.get_input_node(node.inputs[1]
                if merge_node.__class__.__name__ == 'Merge' 
                    and merge_node.get_input_node(node.inputs[0]).__class__.__name__ == 'Merge':
                    return True
                else:
                    return False
            elif node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant':
                merge_node = node.get_input_node(node.inputs[0]
                if merge_node.__class__.__name__ == 'Merge' 
                    and merge_node.get_input_node(node.inputs[0]).__class__.__name__ == 'Merge':
                    return True
                else:
                    return False
        return False

    def transform(self, model, node):
        """ Substitute Matmul + const + sub + mul for a single dense """
        #determining layer ordering
        matmul_node = node
        sub_node = None
        mul_node = None
        const_node = None
        if matmul_node.get_input_node(matmul_node.inputs[0]).__class__.__name__ == 'Constant':
            const_node = matmul_node.get_input_node(matmul_node.inputs[0])
            sub_node = matmul_node.get_input_node(matmul_node.inputs[1])
        else:
            const_node = matmul_node.get_input_node(matmul_node.inputs[1])
            sub_node = matmul_node.get_input_node(matmul_node.inputs[0])
        mul_node = sub_node.get_input_node(sub_node.inputs[0])
        
        #creating the attributes
        input_shape = sub_node.get_input_variable().shape
        output_shape = matmul_node.get_output_variable().shape
        attributes = {
            'n_in': input_shape,
            'n_out': output_shape,
            }

        #making new node, takes inputs from mul node, replaces matmul node, no need to rewire
        new_dense = model.make_node("Dense", matmul_node.name, attributes, mul_node.inputs.copy())
        new_dense.weights['weight'] = const_node.value
        #TODO: how to integrate factor and subtrahend into this?
        
        #removing and replacing old nodes
        model.replace_node(matmul_node, new_dense)
        model.remove_node(sub_node, rewire=False)
        model.remove_node(const_node, rewire=False)

        return True
