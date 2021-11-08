import numpy as np
from hls4ml.model.optimizer import OptimizerPass

class MatmulConstToDense(OptimizerPass):
    """ Convert MatMul with constant to a dense layer """
    def match(self, node):
        is_match = (node.__class__.__name__ == 'MatMul' and len(node.inputs) == 2 and 
                    (node.get_input_node(node.inputs[0]).__class__.__name__ == 'Constant' or 
                        node.get_input_node(node.inputs[1]).__class__.__name__ == 'Constant'))
        return is_match

    def transform(self, model, node):
        """ Substitute Matmul + Constant for a single dense """
        #determining Constant layer input
        matmul_node = node
        const_node = None
        const_inp_idx = 0
        other_inp_idx = 1
        if matmul_node.get_input_node(matmul_node.inputs[0]).__class__.__name__ == 'Constant':
            const_node = matmul_node.get_input_node(matmul_node.inputs[0])
            other_node = matmul_node.get_input_node(matmul_node.inputs[1])
        else:
            const_node = matmul_node.get_input_node(matmul_node.inputs[1])
            other_node = matmul_node.get_input_node(matmul_node.inputs[0])
            const_inp_idx = 1
            other_inp_idx = 0
        
        #creating the attributes
        #TODO: what other attributes need to be set here?
        #get input shape by taking the output shape of the non constant input node        
        input_shape = matmul_node.get_input_node(matmul_node.inputs[1-const_inp_idx]).get_output_variable().shape 
        output_shape = matmul_node.get_output_variable().shape
        attributes = {
            'n_in': input_shape,
            'n_out': output_shape,
            "weights_data": const_node.value,
            "weights_precision": const_node.get_attr("quant_precision"), 
            "weights_quantizer": const_node.get_attr("quantizer"),
            "precision": other_node.get_attr("quant_precision"), 
            "quantizer": other_node.get_attr("quantizer"),
            "omit_bias": True
        }

        #making new node
        new_dense = model.make_node("Dense", f"Dense_{matmul_node.name}", attributes, [matmul_node.inputs[other_inp_idx]], matmul_node.outputs)

        #removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        model.replace_node(matmul_node, new_dense)

        return True
