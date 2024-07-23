import numpy as np

from hls4ml.model.layers import Constant, Dense, MatMul
from hls4ml.model.optimizer import OptimizerPass


class MatmulConstToDense(OptimizerPass):
    """
    Convert MatMul with constant to a dense layer. Note, this only supports the second input
    being the constant. If needed, one could add transposes to make that be the case in
    other yet to be written optimizers.
    """

    def match(self, node):
        is_match = (
            isinstance(node, MatMul) and len(node.inputs) == 2 and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )
        return is_match

    def transform(self, model, node):
        """Substitute Matmul + Constant for a single dense"""
        # determining Constant layer input
        const_node = node.get_input_node(node.inputs[1])
        other_var = node.get_input_variable(node.inputs[0])

        weight_data = const_node.attributes['value']
        weight_quantizer = const_node.get_attr('quantizer')

        # get the configuration name
        config = model.config.get_layer_config(node)
        new_name = f'Dense_{node.name}'
        model.config.set_name_config(new_name, config)
        model.config.parse_name_config(new_name, config)

        in_shape = other_var.shape
        n_in = np.prod(in_shape)
        out_shape = list(in_shape[:-1]) + [weight_data.shape[-1]]
        n_out = np.prod(out_shape)

        # creating the attributes
        attributes = {
            'weight_data': weight_data,
            'weight_quantizer': weight_quantizer,
            'bias_data': np.zeros(out_shape),
            'use_bias': False,
            'n_in': n_in,
            'n_out': n_out,
        }

        # making new node
        new_dense = model.make_node(Dense, new_name, attributes, [node.inputs[0]], [x for x in node.outputs])

        # removing and replacing old nodes
        model.remove_node(const_node, rewire=False)
        del node.inputs[1]
        model.replace_node(node, new_dense)

        return True
